"""
optimize_ic_lora.py — Joint LoRA + IC optimization for GraphCast

Optimizes low-rank (LoRA) weight corrections and/or initial condition (IC)
perturbations to minimize GraphCast's training loss against ERA5 over an
autoregressive forecast window. Modes (config key joint_mode):

  'simultaneous' — one forward/backward pass gives gradients for both
                   LoRA factors and IC; both updated each step with
                   separate Adam optimizers.
  'lora_then_ic' — Phase 1: optimize LoRA (IC frozen).
                   Phase 2: freeze LoRA, optimize IC.
  'alternating'  — Alternate K-step blocks: K steps LoRA, K steps IC.
  'lora_only'    — optimize LoRA only (IC unchanged).
  'ic_only'      — optimize IC only (weights unchanged).

The run is configured by a JSON file whose path is given in the j_path
environment variable (example: configs/lora_only_2017-01-01T12.json). Main fields:
  init_date, pred_steps:  forecast start and window length (6-hr steps)
  lora_rank, lora_alpha, lora_sigma, lora_lr, lora_target, correction_mode
  ic_lr:          learning rate for IC optimizer (default 1e-3)
  n_optim_steps:  total steps when expanding_windows is not set
  expanding_windows, steps_per_window: train on successively longer windows
  patience:       early stopping patience per phase
  block_size:     K for alternating mode (default 10)
  lora_products_path: (optional) path to pre-trained LoRA products to
                      warm-start LoRA (useful for lora_then_ic mode)
  save_root:      output directory; results are written to {save_root}/{date}/
"""

import sys, os, json, time
import numpy as np
import jax
import jax.numpy as jnp
from jax import config as jconfig
from jax.random import PRNGKey as jaxkey
from optax import adam, apply_updates, linear_schedule
from logging import info, basicConfig, INFO, StreamHandler
import xarray
from xarray import load_dataset as xld
from dataclasses import asdict
from numpy import datetime64, arange
from numpy import random as npr
from random import seed as rseed
import warnings

# --- Paths --- (repository root, so the bundled graphcast/ package is importable)
path_to_add = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if path_to_add not in sys.path:
    sys.path.append(path_to_add)
warnings.filterwarnings("ignore", message="endian-ness of dtype")

# --- Reproducibility ---
def set_deterministic(seed=0):
    key = jaxkey(seed)
    rseed(seed)
    npr.seed(seed)
    return key

key = set_deterministic()

# --- JAX config ---
# Must set x64 BEFORE any JAX operations (including imports that trigger JAX init)
_pre_config = json.load(open(os.getenv('j_path'), 'r'))
_use_x64 = _pre_config.get('use_x64', False)
jconfig.update("jax_debug_nans", True)
jconfig.update("jax_enable_x64", _use_x64)
if _use_x64:
    print(f'F64 mode enabled')

# --- Logging ---
basicConfig(
    level=INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[StreamHandler()],
    force=True,
)

# --- Load model and config ---
from load_model import model_config, task_config, params
from config import dataset_file_path as _default_dataset_path, drop_time_dim_if_present
import jitted
from graphcast import data_utils
from graphcast import xarray_jax  # unwrap_data for the IC penalties inside the jitted loss

if _use_x64:
    # Upcast model params to float64
    params = jax.tree_util.tree_map(
        lambda x: jnp.array(x, dtype=jnp.float64) if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating) else x,
        params
    )

j_path = os.getenv('j_path')
with open(j_path, 'r') as f:
    config = json.load(f)

init_date = config['init_date']
pred_steps = config['pred_steps']
just = config.get('justify', 'left')

# Joint optimization config
joint_mode = config.get('joint_mode', 'simultaneous')
assert joint_mode in ('simultaneous', 'lora_then_ic', 'alternating',
                      'lora_only', 'ic_only'), \
    f"Unknown joint_mode '{joint_mode}'"

# LoRA config
correction_mode = config.get('correction_mode', 'lora')
rank = config.get('lora_rank', 4)
alpha = config.get('lora_alpha', 4)
sigma = config.get('lora_sigma', 0.01 / rank**0.5)
lora_lr = config.get('lora_lr', 1e-3)
lora_target = config.get('lora_target', 'processor_all')

# IC config
ic_lr = config.get('ic_lr', 1e-3)

# Optimization config
n_optim_steps = config.get('n_optim_steps', 100)
patience = config.get('patience', 20)
block_size = config.get('block_size', 10)  # for alternating mode
dataset_file_path = config.get('dataset_file_path', _default_dataset_path)
lora_products_path = config.get('lora_products_path', None)

# --- IC/LoRA trade-off dial (normalized weak-constraint penalty; "Alpha-dial" in the log) ---
# J = L_MSE + lambda_IC*(S_IC/S_IC_joint) + lambda_W*(S_W/S_W_joint)
# Single dial rho with geometric-mean strength Lambda:
#   lambda_IC = Lambda*exp(-rho/2), lambda_W = Lambda*exp(+rho/2)
# rho<0 -> IC penalized -> toward LoRA-only ; rho>0 -> LoRA penalized -> toward IC-only.
rho        = float(config.get('rho', 0.0))
Lambda     = float(config.get('Lambda', 0.0))   # default 0 -> no penalty (plain joint optimization)
S_IC_joint = float(config.get('S_IC_joint', 489.5))  # normalizer: _S_IC of an unpenalized joint solution
S_W_joint  = float(config.get('S_W_joint', 0.446))   # normalizer: _S_W of an unpenalized joint solution
lambda_IC = Lambda * float(np.exp(-rho / 2.0))
lambda_W  = Lambda * float(np.exp(+rho / 2.0))
# Laplacian smoothness penalty on the IC delta (on by default):
#   penalty = ic_laplacian_weight * sum_var w_var * sum_grid cos(lat)*(grad^2 dx_var)^2
# in z-scored space (z-scoring already makes variables comparable). The default weights
# w_var emphasize the large-scale mass field (geopotential x3) and de-emphasize the
# noisier fields (winds / MSLP / precip x0.1); all others 1.0. It depends only on the IC
# delta, so it does not change the LoRA updates in 'lora_only' mode.
_ZBOOST_WEIGHTS = {
    'geopotential': 3.0,
    'u_component_of_wind': 0.1, 'v_component_of_wind': 0.1,
    '10m_u_component_of_wind': 0.1, '10m_v_component_of_wind': 0.1,
    'mean_sea_level_pressure': 0.1, 'total_precipitation_6hr': 0.1,
}
ic_laplacian_weight = float(config.get('ic_laplacian_weight', 5e-4))         # 0.0 disables
laplacian_var_weights = config.get('laplacian_var_weights', _ZBOOST_WEIGHTS)  # None -> flat (all 1.0)
log_every = int(config.get('log_every', 10))  # print loss every N steps (1 = every epoch)
per_ts_every = int(config.get('per_ts_every', 1))  # compute per-timestep eval every N steps (>1 saves walltime)

# Expanding window config
expanding_windows = config.get('expanding_windows', None)  # e.g. [8,16,24,32,40]
steps_per_window = config.get('steps_per_window', 50)  # int, or a list with one int per expanding window
loss_skip_steps = config.get('loss_skip_steps', 0)  # skip first N timesteps in loss
scale_lr_by_window = config.get('scale_lr_by_window', True)  # per-window LR = base/(max(ps/4,1))**lr_window_exponent; DEFAULT ON
# Exponent of the per-window LR shrinkage (0.5 = square root, 1.0 = linear in window
# length). With Adam, each window's first step moves every parameter by ~LR, while the
# rollout-loss sensitivity to that step grows with window length, so longer windows
# need a smaller LR to avoid a loss overshoot when the window widens. lora_lr and ic_lr
# are scaled by the same factor.
lr_window_exponent = float(config.get('lr_window_exponent', 1.0))
# Extra LR multiplier applied to the LARGEST window only (both learning rates). 1.0 = off.
largest_window_lr_mult = float(config.get('largest_window_lr_mult', 1.0))
# Per-window LR warmup. The optimizer is re-initialized at every window widening, so
# Adam's first bias-corrected update is m_hat/(sqrt(v_hat)+eps) ~ sign(g): every
# parameter moves by the full LR in the same step, and that parameter-space jump scales
# as lr*sqrt(N_params), i.e. it grows with LoRA rank. A short linear ramp removes the
# start-of-window spike without a global LR cut. 0 = off (constant LR in each window).
lr_warmup_steps = int(config.get('lr_warmup_steps', 0))


def _win_lr(base_lr):
    """Per-window LR: a constant, or a linear ramp over the first lr_warmup_steps steps.

    optax counts from 0 at each optimizer.init(), and the optimizer is re-initialized
    once per window, so the ramp automatically restarts at every window widening.
    """
    if lr_warmup_steps > 0:
        return linear_schedule(init_value=base_lr / lr_warmup_steps,
                               end_value=base_lr,
                               transition_steps=lr_warmup_steps)
    return base_lr
if expanding_windows:
    expanding_windows = sorted(expanding_windows)
    # Filter out windows that are entirely within the skip region
    if loss_skip_steps > 0:
        expanding_windows = [w for w in expanding_windows if w > loss_skip_steps]
    pred_steps = max(expanding_windows)  # extract data for largest window
    info(f'Expanding window mode: {[f"ps={w}" for w in expanding_windows]}')

# Per-window step counts: a steps_per_window list is aligned one-to-one with
# expanding_windows (ascending). A single int applies to every window.
if isinstance(steps_per_window, (list, tuple)):
    assert expanding_windows and loss_skip_steps == 0, \
        'a steps_per_window list needs expanding_windows and loss_skip_steps=0'
    assert list(config['expanding_windows']) == expanding_windows, \
        'give expanding_windows in ascending order when steps_per_window is a list'
    assert len(steps_per_window) == len(expanding_windows), \
        f'steps_per_window {steps_per_window} needs one entry per window {expanding_windows}'
    steps_per_window = [int(s) for s in steps_per_window]
    info(f'Per-window steps: {dict(zip(expanding_windows, steps_per_window))}')


def _steps_for_window(win_idx):
    """Steps in expanding window win_idx: the shared int, or that window's list entry."""
    return steps_per_window[win_idx] if isinstance(steps_per_window, list) else steps_per_window

# --- Load dataset and extract inputs/targets/forcings ---
init_date_dt = datetime64(init_date)
inter = xld(dataset_file_path, decode_timedelta=False).compute()
inter = inter.assign(
    geopotential_at_surface=drop_time_dim_if_present(inter, 'geopotential_at_surface'),
    land_sea_mask=drop_time_dim_if_present(inter, 'land_sea_mask'),
)

for j, i in enumerate(inter.datetime.values[0]):
    if i == init_date_dt:
        start_index = j - 1
        break

example_batch = inter.isel(time=slice(start_index, None))
date_train = str(example_batch.datetime.values[0][1])[:13]

num_time_steps = len(example_batch.time)
new_time = arange(
    0, num_time_steps * 6 * 3600 * 10**9, 6 * 3600 * 10**9, dtype='timedelta64[ns]'
)
example_batch = example_batch.assign_coords(time=new_time)

train_inputs, train_targets, train_forcings = data_utils.extract_inputs_targets_forcings(
    example_batch,
    target_lead_times=slice("6h", f"{pred_steps * 6}h"),
    **asdict(task_config),
    justify=just,
)

# F64 mode: upcast all data to float64 before any JAX operations
if _use_x64:
    train_inputs = train_inputs.astype('float64')
    train_targets = train_targets.astype('float64')
    train_forcings = train_forcings.astype('float64')

# Keep a frozen copy of original inputs for IC perturbation reference
orig_inputs = jax.tree_util.tree_map(lambda x: x, train_inputs)

# =====================================================================
# Normalize IC inputs (z-score) so all variables are on the same footing
# =====================================================================
# IC optimization operates in normalized space (zero mean, unit variance).
# Inside the loss function, we denormalize before passing to the model.

_STATIC_VARS = [
    'year_progress_cos', 'year_progress_sin', 'day_progress_cos',
    'day_progress_sin', 'toa_incident_solar_radiation',
    'geopotential_at_surface', 'land_sea_mask',
]

def _normalize_32(ds):
    """Normalize dynamic variables to zero mean, unit variance."""
    _dtype = 'float64' if _use_x64 else 'float32'
    ds_copy = ds.copy()
    stats = {}
    norm_dims = ['lon', 'lat', 'batch']
    for var_name, da in ds.data_vars.items():
        if var_name not in _STATIC_VARS:
            dims_to_reduce = [d for d in norm_dims if d in da.dims]
            mean = da.mean(dim=dims_to_reduce, keep_attrs=True).astype(_dtype)
            std = da.std(dim=dims_to_reduce, keep_attrs=True).astype(_dtype)
            ds_copy[var_name] = ((da.astype(_dtype) - mean) / std)
            stats[var_name] = {'mean': mean, 'std': std}
        else:
            ds_copy[var_name] = da.astype(_dtype)
    return ds_copy, stats

norm_inputs, norm_stats = _normalize_32(train_inputs)
orig_norm_inputs = jax.tree_util.tree_map(lambda x: x, norm_inputs)

# Precision-aware denormalize (avoids jitted.denormalize_32 which hardcodes float32)
def _denormalize(norm_ds, stats):
    """Denormalize using per-sample stats. Preserves input dtype (F32 or F64)."""
    dt_copy = norm_ds.copy()
    for var_name in stats:
        da = dt_copy[var_name]
        mean = stats[var_name]['mean']
        std = stats[var_name]['std']
        dt_copy[var_name] = da * std + mean
    return dt_copy

info(f'Date: {date_train}  Lead: {pred_steps} steps ({pred_steps / 4:.1f} days)')
info(f'Joint mode: {joint_mode}  Correction: {correction_mode}')
info(f'LoRA — rank: {rank}  alpha: {alpha}  lr: {lora_lr}  target: {lora_target}')
info(f'IC — lr: {ic_lr}')
info(f'Steps: {n_optim_steps}  Patience: {patience}  Block size: {block_size}')
info(f'Alpha-dial: rho={rho:+.3f} Lambda={Lambda:.4g} -> lambda_IC={lambda_IC:.4g} lambda_W={lambda_W:.4g} '
     f'(S_IC_joint={S_IC_joint:.4g} S_W_joint={S_W_joint:.4g})')

# =====================================================================
# Initialize LoRA corrections
# =====================================================================
from lora_utils import (
    get_target_keys, init_lora_factors, apply_lora,
    save_lora_products, save_lora_factors, load_lora_products,
    load_lora_factors, verify_init,
)

target_keys = get_target_keys(params, target=lora_target)

if correction_mode == 'lora':
    trainable = init_lora_factors(params, target_keys, rank, sigma)
    assert verify_init(params, trainable, alpha, rank), "Gate 0 FAILED"

    def _apply_corrections(trainable, base_params):
        return apply_lora(base_params, trainable, alpha, rank)

else:
    raise ValueError(f"Only 'lora' correction_mode supported in joint IC. Got '{correction_mode}'.")

# Warm-start LoRA from pre-trained products if provided
if lora_products_path is not None:
    info(f'Warm-starting LoRA from {lora_products_path}')
    products = load_lora_products(lora_products_path)
    # Continued optimization needs factors, but products store only deltaW:
    # keep the freshly initialized A and solve for B.
    scale = float(alpha) / float(rank) ** 0.5
    for key in trainable:
        if key in products:
            deltaW = jnp.array(products[key], dtype=jnp.float32)
            _, A = trainable[key]
            # Set B = deltaW @ pinv(A) / scale so that scale * B @ A ≈ deltaW
            A_pinv = jnp.linalg.pinv(A)
            B_new = (deltaW @ A_pinv) / scale
            trainable[key] = (B_new, A)
    info(f'  Loaded {len(products)} LoRA products as warm-start')

# Resume from saved checkpoint (factors + normalized IC delta)
resume_from = config.get('resume_from', None)
if resume_from is not None:
    factors_path = resume_from['factors_path']
    ic_norm_delta_path = resume_from['ic_norm_delta_path']
    info(f'Resuming from checkpoint:')
    info(f'  Factors: {factors_path}')
    info(f'  IC norm delta: {ic_norm_delta_path}')
    loaded_factors = load_lora_factors(factors_path)
    for key in trainable:
        if key in loaded_factors:
            trainable[key] = loaded_factors[key]
    info(f'  Loaded {len(loaded_factors)} LoRA factor pairs')
    # Apply saved normalized IC delta to norm_inputs
    ic_delta_data = np.load(ic_norm_delta_path)
    for var in ic_delta_data:
        if var in norm_inputs:
            norm_inputs[var] = norm_inputs[var] + jnp.array(ic_delta_data[var], dtype=jnp.float32)
    info(f'  Applied IC norm delta ({len(ic_delta_data.files)} vars)')

# =====================================================================
# Build zero-grad mask for IC (static/forcing variables)
# =====================================================================
from xarray import DataArray as xda
from numpy import zeros_like

def _make_fixed_grad_values(inputs):
    """Create zero-gradient masks for static/forcing vars in inputs."""
    fixed = {}
    static_vars = [
        'toa_incident_solar_radiation', 'land_sea_mask',
        'geopotential_at_surface', 'day_progress_sin', 'day_progress_cos',
        'year_progress_cos', 'year_progress_sin',
    ]
    for var_name in static_vars:
        if var_name in inputs:
            fixed[var_name] = xda(
                zeros_like(inputs[var_name].values),
                coords=inputs[var_name].coords,
                dims=inputs[var_name].dims,
            )
    return fixed

fixed_grad_values = _make_fixed_grad_values(norm_inputs)

def zero_grads_ic(grads):
    """Zero out gradients for static/forcing variables in IC."""
    for var_name, fixed_value in fixed_grad_values.items():
        if var_name in grads:
            grads[var_name] = fixed_value
    return grads


# =====================================================================
# Dial penalty pieces (cos-lat IC norm + LoRA Frobenius norm)
# =====================================================================
# Dynamic (optimized) vars only — excludes the 7 static/forcing vars. Fixed at trace time.
_DYN_VARS = [v for v in norm_inputs.data_vars if v not in fixed_grad_values]
# Bare cos(lat) weight (NO mean-normalization), shape (lat,1) -> broadcasts over (...,lat,lon).
# Raw cos + jnp.sum is the same convention as the S_IC_joint normalizer.
_COSLAT_W = jnp.asarray(
    np.cos(np.deg2rad(np.asarray(norm_inputs['lat'].values, dtype=np.float32)))[:, None]
)
_LORA_SCALE2 = jnp.float32((alpha / (rank ** 0.5)) ** 2)


def _S_IC(norm_ic):
    """cos-lat-weighted SUM of squared z-score IC delta over dynamic vars (no level/var weight)."""
    total = jnp.array(0.0, dtype=jnp.float32)
    for var in _DYN_VARS:
        diff = (norm_ic[var] - orig_norm_inputs[var]).astype('float32')
        sq = xarray_jax.unwrap_data(diff ** 2, require_jax=True)
        total = total + jnp.sum(sq * _COSLAT_W)
    return total


def _S_IC_laplacian(norm_ic):
    """cos-lat-weighted SUM of squared discrete Laplacian of the z-score IC delta.

    5-point ∇² on the (lat,lon) grid: periodic in lon (jnp.roll), and lat uses the
    same roll — the wrap-around error lives only in the pole rows, which carry ~0
    cos-lat weight, so the penalty is effectively a spherical roughness measure.
    Penalizing this drives δx toward smooth/large-scale corrections.
    """
    total = jnp.array(0.0, dtype=jnp.float32)
    for var in _DYN_VARS:
        diff = (norm_ic[var] - orig_norm_inputs[var]).astype('float32')
        d = xarray_jax.unwrap_data(diff, require_jax=True)  # (..., lat, lon)
        lap = (jnp.roll(d, 1, axis=-1) + jnp.roll(d, -1, axis=-1) - 2.0 * d) \
            + (jnp.roll(d, 1, axis=-2) + jnp.roll(d, -1, axis=-2) - 2.0 * d)
        vw = float(laplacian_var_weights.get(var, 1.0)) if laplacian_var_weights else 1.0
        total = total + jnp.float32(vw) * jnp.sum((lap ** 2) * _COSLAT_W)
    return total


def _S_W(trainable):
    """scale^2 * sum_k sum((B_k @ A_k)^2) = sum of squared effective weight delta."""
    total = jnp.array(0.0, dtype=jnp.float32)
    for _key, (B, A) in trainable.items():
        BA = B @ A
        total = total + jnp.sum(BA ** 2)
    return _LORA_SCALE2 * total


# =====================================================================
# Gradient functions for each mode
# =====================================================================

def _joint_loss(trainable, base_params, norm_ic, targets, forcings):
    """Loss function differentiable w.r.t. both trainable (LoRA) and norm_ic (normalized IC).

    norm_ic is in z-scored space. We denormalize before the forward pass so the
    model sees raw inputs, but gradients flow back through denormalization,
    giving well-scaled ∂loss/∂norm_ic.
    """
    effective_params = _apply_corrections(trainable, base_params)
    denorm_ic = _denormalize(norm_ic, norm_stats)
    if loss_skip_steps > 0:
        # Run full forward pass but only count loss from step loss_skip_steps onward
        (per_ts, diagnostics), next_state = jitted.timestep_loss_fn.apply(
            effective_params, {}, jax.random.PRNGKey(0),
            model_config, task_config,
            denorm_ic, targets, forcings,
        )
        loss = per_ts[loss_skip_steps:].mean()
    else:
        (loss, diagnostics), next_state = jitted.loss_fn.apply(
            effective_params, {}, jax.random.PRNGKey(0),
            model_config, task_config,
            denorm_ic, targets, forcings,
        )
    # --- dial penalty (F32; differentiable: penIC->norm_ic arg2, penW->trainable arg0) ---
    S_IC = _S_IC(norm_ic)
    S_W = _S_W(trainable)
    penIC = jnp.float32(lambda_IC) * S_IC / jnp.float32(S_IC_joint)
    penW = jnp.float32(lambda_W) * S_W / jnp.float32(S_W_joint)
    # Optional Laplacian smoothness penalty on δx (raw weight; 0 -> off, no effect on aux/dial).
    pen_lap = jnp.float32(ic_laplacian_weight) * _S_IC_laplacian(norm_ic) \
        if ic_laplacian_weight != 0.0 else jnp.float32(0.0)
    total_loss = loss.astype('float32') + penIC + penW + pen_lap
    return total_loss, (diagnostics, next_state, S_IC, S_W, penIC, penW)


# --- Simultaneous: gradients w.r.t. both LoRA and IC in one pass ---
@jax.jit
def simultaneous_step(trainable, base_params, inputs, targets, forcings):
    (loss, (diagnostics, _ns, S_IC, S_W, penIC, penW)), (lora_grads, ic_grads) = jax.value_and_grad(
        _joint_loss, argnums=(0, 2), has_aux=True,
    )(trainable, base_params, inputs, targets, forcings)
    return loss, diagnostics, lora_grads, ic_grads, S_IC, S_W, penIC, penW


# --- LoRA-only step (for lora_then_ic phase 1, alternating LoRA blocks) ---
@jax.jit
def lora_only_step(trainable, base_params, inputs, targets, forcings):
    (loss, (diagnostics, _ns, _sic, _sw, _pic, _pw)), lora_grads = jax.value_and_grad(
        _joint_loss, argnums=0, has_aux=True,
    )(trainable, base_params, inputs, targets, forcings)
    return loss, diagnostics, lora_grads


# --- IC-only step (for lora_then_ic phase 2, alternating IC blocks) ---
@jax.jit
def ic_only_step(trainable, base_params, inputs, targets, forcings):
    (loss, (diagnostics, _ns, _sic, _sw, _pic, _pw)), ic_grads = jax.value_and_grad(
        _joint_loss, argnums=2, has_aux=True,
    )(trainable, base_params, inputs, targets, forcings)
    return loss, diagnostics, ic_grads


# --- Per-timestep evaluation (for diagnostics) ---
@jax.jit
def per_ts_eval(trainable, base_params, norm_ic, targets, forcings):
    effective_params = _apply_corrections(trainable, base_params)
    denorm_ic = _denormalize(norm_ic, norm_stats)
    (per_ts_losses, _), _ = jitted.timestep_loss_fn.apply(
        effective_params, {}, jax.random.PRNGKey(0),
        model_config, task_config,
        denorm_ic, targets, forcings,
    )
    return per_ts_losses


# =====================================================================
# Save paths — defined early so loss_checkpoint.npy works during optimization
# =====================================================================
save_root = config.get('save_root', 'outputs')
save_dir = f'{save_root}/{date_train}'
os.makedirs(save_dir, exist_ok=True)
prefix = f'{date_train}_r{rank}_{joint_mode}'


# =====================================================================
# Control baseline
# =====================================================================
info('Computing control per-timestep loss baseline...')
_identity_trainable = jax.tree_util.tree_map(jnp.zeros_like, trainable)
control_per_ts = np.array(per_ts_eval(
    _identity_trainable, params, norm_inputs, train_targets, train_forcings
))
control_loss = float(control_per_ts.squeeze().mean())
info(f'Control mean loss: {control_loss:.4f}')


# =====================================================================
# Optimization loops
# =====================================================================

loss_history = []
per_ts_history = []
lora_norm_history = []
ic_norm_history = []
penIC_history = []
penW_history = []
S_IC_history = []
S_W_history = []

best_loss = float('inf')
best_trainable = trainable
best_norm_inputs = norm_inputs
best_step = 0

# --- Optional per-window IC snapshots (diagnostic; default OFF) ---
# When True (ic_only expanding runs), dump the committed best IC delta at each window
# boundary to <save_dir>/window_ps<T>/ plus a forward eval of each snapshot in the
# ps_eval block. Pure no-op unless enabled: does NOT touch optimization/RNG/optimizer/loss.
save_window_snapshots = bool(config.get('save_window_snapshots', False))
window_ic_snapshots = []  # (win_ps, ic_delta_norm_pytree) captured at each window's best-commit

# --- Optional per-epoch IC snapshots (diagnostic; default OFF) ---
# When True (ic_only single-window), dump the LIVE post-update IC delta each epoch to
# <save_dir>/epoch_<gstep>/ and record a per-var magnitude trace every epoch. Log-spaced
# full-IC dumps if epoch_save_log>0 (dense early), else every epoch_save_every. Pure no-op
# unless enabled: does NOT touch optimization/RNG/optimizer/loss.
save_epoch_snapshots = bool(config.get('save_epoch_snapshots', False))
epoch_save_every     = int(config.get('epoch_save_every', 1))   # used only if epoch_save_log == 0
epoch_save_log       = int(config.get('epoch_save_log', 0))     # >0 => save at this many log-spaced epochs
if save_epoch_snapshots and epoch_save_log > 0:
    _Nepoch = max(int(n_optim_steps), 2)
    _epoch_save_set = {int(round(x)) for x in np.logspace(0, np.log10(_Nepoch), epoch_save_log)}
    _epoch_save_set.add(_Nepoch)
else:
    _epoch_save_set = None


def _log_step(step, loss_val, tag='', win_targets=None, win_forcings=None):
    """Log a step with optional improvement marker."""
    global best_loss, best_trainable, best_norm_inputs, best_step
    loss_history.append(loss_val)

    _tgt = win_targets if win_targets is not None else train_targets
    _frc = win_forcings if win_forcings is not None else train_forcings
    # per_ts_eval is an extra forward each step; gate it (default per_ts_every=1 -> every step,
    # unchanged). Larger values skip the forward to save walltime; loss_history is still per-step.
    if step % per_ts_every == 0:
        per_ts = np.array(per_ts_eval(trainable, params, norm_inputs, _tgt, _frc))
        per_ts_history.append(per_ts)

    is_best = loss_val < best_loss
    if is_best:
        best_loss = loss_val
        best_trainable = trainable
        best_norm_inputs = norm_inputs
        best_step = step

    if is_best or step % log_every == 0:
        marker = ' *' if is_best else ''
        info(f'{tag}{step:4d}  loss={loss_val:.6f}{marker}')

    # Checkpoint: flush loss history every step so preemption doesn't lose data
    try:
        np.save(f'{save_dir}/{prefix}_loss_checkpoint.npy', np.array(loss_history))
    except Exception:
        pass  # save_dir/prefix may not yet be defined on first call; fail silently

    return is_best


def _compute_ic_delta_norm(current_inputs, original_inputs):
    """Compute the L2 norm of the IC perturbation (dynamic vars only)."""
    total_sq = 0.0
    for var in current_inputs:
        if var not in fixed_grad_values:  # skip static vars
            diff = np.array(current_inputs[var].values) - np.array(original_inputs[var].values)
            total_sq += float(np.sum(diff**2))
    return float(np.sqrt(total_sq))


def _save_window_ic_snapshot(win_ps, snap_norm_inputs, win_best_loss):
    """Dump the committed best IC delta for one expanding window to window_ps<T>/.
    Additive: writes raw ic_delta.npz (denorm space, same recipe as the final save)
    + a small meta.json.
    Returns the normalized IC delta pytree for the later ps_eval forward pass."""
    w_dir = f'{save_dir}/window_ps{win_ps}'
    os.makedirs(w_dir, exist_ok=True)
    snap_raw = _denormalize(snap_norm_inputs, norm_stats)
    orig_raw = _denormalize(orig_norm_inputs, norm_stats)
    delta = {v: (np.array(snap_raw[v].values) - np.array(orig_raw[v].values)).astype(np.float32)
             for v in snap_raw if v not in fixed_grad_values}
    np.savez(f'{w_dir}/{prefix}_ic_delta.npz', **delta)
    with open(f'{w_dir}/{prefix}_meta.json', 'w') as _f:
        json.dump({'init_date': date_train, 'joint_mode': joint_mode, 'window_ps': int(win_ps),
                   'horizon_days': win_ps * 6 / 24, 'window_best_loss': float(win_best_loss),
                   'rank': rank, 'ic_lr': ic_lr, 'lr_window_exponent': lr_window_exponent}, _f, indent=2)
    return jax.tree_util.tree_map(lambda b, o: b - o, snap_norm_inputs, orig_norm_inputs)


# --- Per-epoch magnitude trace + live-IC snapshot (host-side numpy; inert unless save_epoch_snapshots) ---
epoch_mag_steps = []
epoch_mag_cos = {v: [] for v in _DYN_VARS}   # cos-lat z-score RMS per dynamic var
epoch_mag_raw = {v: [] for v in _DYN_VARS}   # raw-unit RMS per dynamic var
_epoch_coslat = np.cos(np.deg2rad(np.asarray(norm_inputs['lat'].values, dtype=np.float64)))[:, None]


def _record_epoch_magnitude(gstep, live_norm_inputs):
    """cos-lat-weighted RMS of the LIVE IC delta (z-score + raw units) per dynamic var."""
    epoch_mag_steps.append(int(gstep))
    live_raw = _denormalize(live_norm_inputs, norm_stats)
    orig_raw = _denormalize(orig_norm_inputs, norm_stats)
    for v in _DYN_VARS:
        dz = (np.asarray(live_norm_inputs[v].values, dtype=np.float64)
              - np.asarray(orig_norm_inputs[v].values, dtype=np.float64))
        w = np.broadcast_to(_epoch_coslat, dz.shape)
        epoch_mag_cos[v].append(float(np.sqrt(np.sum((dz ** 2) * w) / np.sum(w))))
        dr = (np.asarray(live_raw[v].values, dtype=np.float64)
              - np.asarray(orig_raw[v].values, dtype=np.float64))
        epoch_mag_raw[v].append(float(np.sqrt(np.sum((dr ** 2) * w) / np.sum(w))))


def _save_epoch_ic_snapshot(gstep, win_ps, live_norm_inputs):
    """Dump the LIVE post-update IC delta for one epoch to epoch_<gstep>/ (raw units,
    same recipe as _save_window_ic_snapshot)."""
    e_dir = f'{save_dir}/epoch_{gstep:04d}'
    os.makedirs(e_dir, exist_ok=True)
    live_raw = _denormalize(live_norm_inputs, norm_stats)
    orig_raw = _denormalize(orig_norm_inputs, norm_stats)
    delta = {v: (np.asarray(live_raw[v].values) - np.asarray(orig_raw[v].values)).astype(np.float32)
             for v in live_raw if v not in fixed_grad_values}
    np.savez(f'{e_dir}/{prefix}_ic_delta.npz', **delta)
    with open(f'{e_dir}/{prefix}_meta.json', 'w') as _f:
        json.dump({'init_date': date_train, 'joint_mode': joint_mode, 'epoch': int(gstep),
                   'window_ps': int(win_ps), 'ic_lr': ic_lr,
                   'scale_lr_by_window': bool(scale_lr_by_window)}, _f, indent=2)


info(f'\n{"="*60}')
info(f'Starting {joint_mode} optimization ({n_optim_steps} steps)')
info(f'{"="*60}')
t_start = time.time()


if joint_mode == 'simultaneous':
    # ------------------------------------------------------------------
    # SIMULTANEOUS: one pass, both gradients, both updates
    # Supports expanding window mode via expanding_windows config.
    # ------------------------------------------------------------------
    window_list = expanding_windows if expanding_windows else [pred_steps]
    global_step = 0

    for win_idx, win_ps in enumerate(window_list):
        is_expanding = len(window_list) > 1
        win_steps = _steps_for_window(win_idx) if is_expanding else n_optim_steps
        win_patience = patience

        if is_expanding:
            info(f'\n--- Window {win_idx+1}/{len(window_list)}: ps={win_ps} ({win_ps*6/24:.0f} days), {win_steps} steps ---')

        # Slice targets/forcings for this window
        win_targets = train_targets.isel(time=slice(0, win_ps))
        win_forcings = train_forcings.isel(time=slice(0, win_ps))

        # Compute control loss for this window
        _zero_t = jax.tree_util.tree_map(jnp.zeros_like, trainable)
        win_ctrl = np.array(per_ts_eval(
            _zero_t, params, orig_norm_inputs, win_targets, win_forcings
        ))
        win_ctrl_loss = float(win_ctrl.squeeze().mean())
        if is_expanding:
            info(f'  Window control loss: {win_ctrl_loss:.4f}')

        # Fresh optimizers each window (momentum from old window hurts).
        # Per-window LR scaling: window_lr = base / max(ps/4, 1)**lr_window_exponent  (default ON).
        _wscale = (max(win_ps / 4.0, 1.0)) ** lr_window_exponent if scale_lr_by_window else 1.0
        _extra = largest_window_lr_mult if (scale_lr_by_window and len(window_list) > 1 and win_ps == window_list[-1]) else 1.0
        lora_optimizer = adam(learning_rate=_win_lr(lora_lr / _wscale * _extra))
        ic_optimizer = adam(learning_rate=_win_lr(ic_lr / _wscale * _extra))
        lora_opt_state = lora_optimizer.init(trainable)
        ic_opt_state = ic_optimizer.init(norm_inputs)
        if is_expanding:
            info(f'  Window LR /{_wscale:.3f} (exp={lr_window_exponent}, xtra={_extra}): lora_lr={lora_lr/_wscale*_extra:.6f} ic_lr={ic_lr/_wscale*_extra:.6f}')

        # Reset best tracking for this window
        best_loss = float('inf')
        no_improve = 0

        for step in range(win_steps):
            loss, diagnostics, lora_grads, ic_grads, S_IC, S_W, penIC, penW = simultaneous_step(
                trainable, params, norm_inputs, win_targets, win_forcings,
            )
            loss_val = float(loss)
            penIC_history.append(float(penIC)); penW_history.append(float(penW))
            S_IC_history.append(float(S_IC)); S_W_history.append(float(S_W))

            # Zero-out IC gradients for static/forcing variables
            ic_grads = zero_grads_ic(ic_grads)

            # Track gradient norms
            lora_norm = float(jnp.sqrt(sum(
                jnp.sum(gB**2) + jnp.sum(gA**2)
                for gB, gA in lora_grads.values()
            )))
            ic_norm = float(jnp.sqrt(sum(
                jnp.sum(v**2) for v in jax.tree_util.tree_leaves(ic_grads)
            )))
            lora_norm_history.append(lora_norm)
            ic_norm_history.append(ic_norm)

            tag = f'[W{win_idx+1}] ' if is_expanding else '[SIM] '
            is_best = _log_step(global_step, loss_val, tag=tag,
                                win_targets=win_targets, win_forcings=win_forcings)
            if is_best:
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= win_patience:
                info(f'  Early stopping at step {step} (patience={win_patience})')
                break

            # Update both parameter sets
            lora_updates, lora_opt_state = lora_optimizer.update(
                lora_grads, lora_opt_state, trainable
            )
            trainable = apply_updates(trainable, lora_updates)

            ic_updates, ic_opt_state = ic_optimizer.update(
                ic_grads, ic_opt_state, norm_inputs
            )
            norm_inputs = apply_updates(norm_inputs, ic_updates)

            global_step += 1

        win_improve = (1 - best_loss / win_ctrl_loss) * 100
        info(f'  Window best: {best_loss:.4f} ({win_improve:.1f}% vs window ctrl)')

    # Update control_loss to reflect the final (largest) window
    control_loss = win_ctrl_loss

    ic_delta = _compute_ic_delta_norm(best_norm_inputs, orig_norm_inputs)
    info(f'Final IC perturbation norm: {ic_delta:.6f}')


elif joint_mode in ('lora_only', 'ic_only'):
    # ------------------------------------------------------------------
    # STANDALONE: optimize exactly one parameter set, the other frozen at 0.
    #   lora_only -> update trainable (LoRA); IC stays at orig (delta = 0)
    #   ic_only   -> update norm_inputs (IC); LoRA stays at 0
    # Identical expanding-window / flat-LR / eval machinery as 'simultaneous',
    # so standalone curves are directly comparable to the joint run.
    # ------------------------------------------------------------------
    window_list = expanding_windows if expanding_windows else [pred_steps]
    global_step = 0

    for win_idx, win_ps in enumerate(window_list):
        is_expanding = len(window_list) > 1
        win_steps = _steps_for_window(win_idx) if is_expanding else n_optim_steps
        win_patience = patience

        if is_expanding:
            info(f'\n--- Window {win_idx+1}/{len(window_list)}: ps={win_ps} ({win_ps*6/24:.0f} days), {win_steps} steps ---')

        win_targets = train_targets.isel(time=slice(0, win_ps))
        win_forcings = train_forcings.isel(time=slice(0, win_ps))

        # Window control loss
        _zero_t = jax.tree_util.tree_map(jnp.zeros_like, trainable)
        win_ctrl = np.array(per_ts_eval(
            _zero_t, params, orig_norm_inputs, win_targets, win_forcings
        ))
        win_ctrl_loss = float(win_ctrl.squeeze().mean())
        if is_expanding:
            info(f'  Window control loss: {win_ctrl_loss:.4f}')

        # Fresh optimizer each window (momentum from old window hurts).
        # Per-window LR scaling: window_lr = base / max(ps/4, 1)**lr_window_exponent  (default ON).
        _wscale = (max(win_ps / 4.0, 1.0)) ** lr_window_exponent if scale_lr_by_window else 1.0
        _extra = largest_window_lr_mult if (scale_lr_by_window and len(window_list) > 1 and win_ps == window_list[-1]) else 1.0
        if joint_mode == 'lora_only':
            optimizer = adam(learning_rate=_win_lr(lora_lr / _wscale * _extra))
            opt_state = optimizer.init(trainable)
        else:  # ic_only
            optimizer = adam(learning_rate=_win_lr(ic_lr / _wscale * _extra))
            opt_state = optimizer.init(norm_inputs)
        if is_expanding:
            info(f'  Window LR /{_wscale:.3f} (exp={lr_window_exponent}, xtra={_extra}): ic_lr={ic_lr/_wscale*_extra:.6f}')

        best_loss = float('inf')
        no_improve = 0

        for step in range(win_steps):
            if joint_mode == 'lora_only':
                loss, diagnostics, grads = lora_only_step(
                    trainable, params, norm_inputs, win_targets, win_forcings,
                )
                lora_norm = float(jnp.sqrt(sum(
                    jnp.sum(gB**2) + jnp.sum(gA**2)
                    for gB, gA in grads.values()
                )))
                lora_norm_history.append(lora_norm)
                ic_norm_history.append(0.0)
            else:  # ic_only
                loss, diagnostics, grads = ic_only_step(
                    trainable, params, norm_inputs, win_targets, win_forcings,
                )
                grads = zero_grads_ic(grads)
                ic_norm = float(jnp.sqrt(sum(
                    jnp.sum(v**2) for v in jax.tree_util.tree_leaves(grads)
                )))
                lora_norm_history.append(0.0)
                ic_norm_history.append(ic_norm)

            loss_val = float(loss)
            tag = f'[W{win_idx+1}] ' if is_expanding else f'[{joint_mode[:3].upper()}] '
            is_best = _log_step(global_step, loss_val, tag=tag,
                                win_targets=win_targets, win_forcings=win_forcings)
            if is_best:
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= win_patience:
                info(f'  Early stopping at step {step} (patience={win_patience})')
                break

            updates, opt_state = optimizer.update(
                grads, opt_state, trainable if joint_mode == 'lora_only' else norm_inputs
            )
            if joint_mode == 'lora_only':
                trainable = apply_updates(trainable, updates)
            else:  # ic_only
                norm_inputs = apply_updates(norm_inputs, updates)

            global_step += 1

            # --- per-epoch snapshot: record LIVE (post-update) IC. Inert unless save_epoch_snapshots. ---
            if save_epoch_snapshots and joint_mode == 'ic_only':
                _record_epoch_magnitude(global_step, norm_inputs)
                _do_ep = ((global_step in _epoch_save_set) if _epoch_save_set is not None
                          else (global_step % epoch_save_every == 0))
                if _do_ep:
                    _save_epoch_ic_snapshot(global_step, win_ps, norm_inputs)

        win_improve = (1 - best_loss / win_ctrl_loss) * 100
        info(f'  Window best: {best_loss:.4f} ({win_improve:.1f}% vs window ctrl)')

        if save_window_snapshots:
            _snap = _save_window_ic_snapshot(win_ps, best_norm_inputs, best_loss)
            window_ic_snapshots.append((int(win_ps), _snap))
            info(f'  [window snapshot] IC snapshot -> window_ps{win_ps}/')

    control_loss = win_ctrl_loss

    ic_delta = _compute_ic_delta_norm(best_norm_inputs, orig_norm_inputs)
    info(f'Final IC perturbation norm: {ic_delta:.6f}')


elif joint_mode == 'lora_then_ic':
    # ------------------------------------------------------------------
    # LORA THEN IC: Phase 1 = LoRA only, Phase 2 = IC only (LoRA frozen)
    # ------------------------------------------------------------------
    phase1_steps = config.get('phase1_steps', n_optim_steps // 2)
    phase2_steps = n_optim_steps - phase1_steps

    # --- Phase 1: LoRA optimization (IC frozen) ---
    info(f'\n--- Phase 1: LoRA optimization ({phase1_steps} steps) ---')
    lora_optimizer = adam(learning_rate=lora_lr)
    lora_opt_state = lora_optimizer.init(trainable)
    no_improve = 0

    for step in range(phase1_steps):
        loss, diagnostics, lora_grads = lora_only_step(
            trainable, params, norm_inputs, train_targets, train_forcings,
        )
        loss_val = float(loss)

        lora_norm = float(jnp.sqrt(sum(
            jnp.sum(gB**2) + jnp.sum(gA**2)
            for gB, gA in lora_grads.values()
        )))
        lora_norm_history.append(lora_norm)
        ic_norm_history.append(0.0)

        is_best = _log_step(step, loss_val, tag='[P1-LoRA] ')
        if is_best:
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            info(f'Phase 1 early stopping at step {step}')
            break

        lora_updates, lora_opt_state = lora_optimizer.update(
            lora_grads, lora_opt_state, trainable
        )
        trainable = apply_updates(trainable, lora_updates)

    # Freeze LoRA at best
    trainable = best_trainable
    phase1_best_loss = best_loss
    info(f'Phase 1 complete. Best loss: {phase1_best_loss:.6f}')

    # --- Phase 2: IC optimization (LoRA frozen) ---
    info(f'\n--- Phase 2: IC optimization ({phase2_steps} steps) ---')
    ic_optimizer = adam(learning_rate=ic_lr)
    ic_opt_state = ic_optimizer.init(norm_inputs)
    no_improve = 0
    # Don't reset best — let IC improve from LoRA's best

    for step in range(phase1_steps, phase1_steps + phase2_steps):
        loss, diagnostics, ic_grads = ic_only_step(
            trainable, params, norm_inputs, train_targets, train_forcings,
        )
        loss_val = float(loss)

        ic_grads = zero_grads_ic(ic_grads)

        ic_norm = float(jnp.sqrt(sum(
            jnp.sum(v**2) for v in jax.tree_util.tree_leaves(ic_grads)
        )))
        lora_norm_history.append(0.0)
        ic_norm_history.append(ic_norm)

        is_best = _log_step(step, loss_val, tag='[P2-IC] ')
        if is_best:
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            info(f'Phase 2 early stopping at step {step}')
            break

        ic_updates, ic_opt_state = ic_optimizer.update(
            ic_grads, ic_opt_state, norm_inputs
        )
        norm_inputs = apply_updates(norm_inputs, ic_updates)

    ic_delta = _compute_ic_delta_norm(best_norm_inputs, orig_norm_inputs)
    info(f'Final IC perturbation norm: {ic_delta:.6f}')


elif joint_mode == 'alternating':
    # ------------------------------------------------------------------
    # ALTERNATING: K steps LoRA, K steps IC, repeat.
    # Supports expanding window mode via expanding_windows config.
    # block_size controls how many steps each component runs before switching.
    # ------------------------------------------------------------------
    window_list = expanding_windows if expanding_windows else [pred_steps]
    global_step = 0

    for win_idx, win_ps in enumerate(window_list):
        is_expanding = len(window_list) > 1
        win_steps = _steps_for_window(win_idx) if is_expanding else n_optim_steps
        win_patience = patience

        if is_expanding:
            info(f'\n--- Window {win_idx+1}/{len(window_list)}: ps={win_ps} ({win_ps*6/24:.0f} days), {win_steps} steps ---')

        win_targets = train_targets.isel(time=slice(0, win_ps))
        win_forcings = train_forcings.isel(time=slice(0, win_ps))

        # Window control loss
        _zero_t = jax.tree_util.tree_map(jnp.zeros_like, trainable)
        win_ctrl = np.array(per_ts_eval(
            _zero_t, params, orig_norm_inputs, win_targets, win_forcings
        ))
        win_ctrl_loss = float(win_ctrl.squeeze().mean())
        if is_expanding:
            info(f'  Window control loss: {win_ctrl_loss:.4f}')

        # Fresh optimizers each window (momentum from prior window hurts)
        lora_optimizer = adam(learning_rate=_win_lr(lora_lr))
        ic_optimizer = adam(learning_rate=_win_lr(ic_lr))
        lora_opt_state = lora_optimizer.init(trainable)
        ic_opt_state = ic_optimizer.init(norm_inputs)

        # Reset best tracking for this window (mirrors simultaneous mode)
        best_loss = float('inf')
        no_improve = 0
        win_step = 0

        while win_step < win_steps:
            # --- LoRA block ---
            block_steps = min(block_size, win_steps - win_step)
            for _ in range(block_steps):
                loss, diagnostics, lora_grads = lora_only_step(
                    trainable, params, norm_inputs, win_targets, win_forcings,
                )
                loss_val = float(loss)

                lora_norm = float(jnp.sqrt(sum(
                    jnp.sum(gB**2) + jnp.sum(gA**2)
                    for gB, gA in lora_grads.values()
                )))
                lora_norm_history.append(lora_norm)
                ic_norm_history.append(0.0)

                tag = f'[W{win_idx+1}-L] ' if is_expanding else '[ALT-L] '
                is_best = _log_step(global_step, loss_val, tag=tag,
                                    win_targets=win_targets, win_forcings=win_forcings)
                if is_best:
                    no_improve = 0
                else:
                    no_improve += 1

                lora_updates, lora_opt_state = lora_optimizer.update(
                    lora_grads, lora_opt_state, trainable
                )
                trainable = apply_updates(trainable, lora_updates)
                global_step += 1
                win_step += 1

            if no_improve >= win_patience or win_step >= win_steps:
                break

            # --- IC block ---
            block_steps = min(block_size, win_steps - win_step)
            for _ in range(block_steps):
                loss, diagnostics, ic_grads = ic_only_step(
                    trainable, params, norm_inputs, win_targets, win_forcings,
                )
                loss_val = float(loss)

                ic_grads = zero_grads_ic(ic_grads)

                ic_norm = float(jnp.sqrt(sum(
                    jnp.sum(v**2) for v in jax.tree_util.tree_leaves(ic_grads)
                )))
                lora_norm_history.append(0.0)
                ic_norm_history.append(ic_norm)

                tag = f'[W{win_idx+1}-I] ' if is_expanding else '[ALT-I] '
                is_best = _log_step(global_step, loss_val, tag=tag,
                                    win_targets=win_targets, win_forcings=win_forcings)
                if is_best:
                    no_improve = 0
                else:
                    no_improve += 1

                ic_updates, ic_opt_state = ic_optimizer.update(
                    ic_grads, ic_opt_state, norm_inputs
                )
                norm_inputs = apply_updates(norm_inputs, ic_updates)
                global_step += 1
                win_step += 1

            if no_improve >= win_patience:
                info(f'Alternating early stopping at step {global_step} (window {win_idx+1})')
                break

        if is_expanding:
            win_improve = (1 - best_loss / win_ctrl_loss) * 100 if win_ctrl_loss > 0 else 0.0
            info(f'  Window best: {best_loss:.4f} ({win_improve:.1f}% vs window ctrl)')
            control_loss = win_ctrl_loss  # final window control becomes the reference

    ic_delta = _compute_ic_delta_norm(best_norm_inputs, orig_norm_inputs)
    info(f'Final IC perturbation norm: {ic_delta:.6f}')


# =====================================================================
# Summary
# =====================================================================
elapsed = time.time() - t_start
if loss_history:
    reduction = (1 - best_loss / loss_history[0]) * 100 if loss_history[0] > 0 else 0.0
    ctrl_improve = (1 - best_loss / control_loss) * 100 if control_loss > 0 else 0.0
else:
    reduction = 0.0
    ctrl_improve = 0.0
info(f'\nBest loss: {best_loss:.6f} at step {best_step} ({reduction:.1f}% reduction from initial)')
info(f'Control mean loss: {control_loss:.4f}')
info(f'Improvement over control: {ctrl_improve:.1f}%')
info(f'Elapsed: {elapsed:.1f}s ({elapsed / max(len(loss_history), 1):.1f}s/step)')


# =====================================================================
# Save results
# =====================================================================
# (save_dir, prefix defined early above for loss_checkpoint.npy)

# LoRA products
product_path = f'{save_dir}/{prefix}_lora_products.npz'
save_lora_products(best_trainable, product_path, alpha, rank)
info(f'Saved LoRA products -> {product_path}')

factors_path = f'{save_dir}/{prefix}_lora_factors.npz'
save_lora_factors(best_trainable, factors_path)
info(f'Saved LoRA factors -> {factors_path}')

# IC perturbation — denormalize best and orig back to raw space, save delta
best_raw = _denormalize(best_norm_inputs, norm_stats)
orig_raw = _denormalize(orig_norm_inputs, norm_stats)
ic_delta_dict = {}
for var in best_raw:
    if var not in fixed_grad_values:  # dynamic vars only
        delta = np.array(best_raw[var].values) - np.array(orig_raw[var].values)
        ic_delta_dict[var] = delta.astype(np.float32)
ic_path = f'{save_dir}/{prefix}_ic_delta.npz'
np.savez(ic_path, **ic_delta_dict)
info(f'Saved IC perturbation -> {ic_path}')

# Normalized IC delta (for clean resume — stays in per-sample-norm space)
ic_norm_delta_dict = {}
for var in best_norm_inputs:
    if var not in fixed_grad_values:
        delta_vals = np.array(best_norm_inputs[var].values) - np.array(orig_norm_inputs[var].values)
        ic_norm_delta_dict[var] = delta_vals.astype(np.float32)
ic_norm_path = f'{save_dir}/{prefix}_ic_norm_delta.npz'
np.savez(ic_norm_path, **ic_norm_delta_dict)
info(f'Saved IC norm delta -> {ic_norm_path}')

# Loss history
loss_path = f'{save_dir}/{prefix}_loss.npy'
np.save(loss_path, np.array(loss_history))

per_ts_path = f'{save_dir}/{prefix}_per_ts_loss.npy'
try:
    np.save(per_ts_path, np.array(per_ts_history))
except ValueError:
    # Expanding window: per-ts arrays have different shapes per window
    import pickle
    with open(per_ts_path.replace('.npy', '.pkl'), 'wb') as pf:
        pickle.dump(per_ts_history, pf)

ctrl_ts_path = f'{save_dir}/{prefix}_control_per_ts_loss.npy'
np.save(ctrl_ts_path, control_per_ts)

# Gradient norm history
norms_path = f'{save_dir}/{prefix}_grad_norms.npz'
np.savez(norms_path,
         lora_norms=np.array(lora_norm_history),
         ic_norms=np.array(ic_norm_history))
info(f'Saved gradient norms -> {norms_path}')

if save_epoch_snapshots and epoch_mag_steps:
    _emag_path = f'{save_dir}/{prefix}_epoch_magnitudes.npz'
    np.savez(_emag_path,
             steps=np.array(epoch_mag_steps, dtype=np.int32),
             dyn_vars=np.array(_DYN_VARS),
             cos_rms=np.stack([np.array(epoch_mag_cos[v]) for v in _DYN_VARS], axis=1),
             raw_rms=np.stack([np.array(epoch_mag_raw[v]) for v in _DYN_VARS], axis=1))
    info(f'Saved per-epoch magnitude trace -> {_emag_path}')

# --- Dial: final penalty quantities on best solution (numpy, host-side) ---
_coslat_np = np.cos(np.deg2rad(np.asarray(norm_inputs['lat'].values, dtype=np.float64)))[:, None]
final_S_IC = 0.0
for _var in _DYN_VARS:
    _d = (np.asarray(best_norm_inputs[_var].values, dtype=np.float64)
          - np.asarray(orig_norm_inputs[_var].values, dtype=np.float64))
    final_S_IC += float(np.sum((_d ** 2) * _coslat_np))
final_S_W = 0.0
for _k, (B, A) in best_trainable.items():
    _BA = np.asarray(B, dtype=np.float64) @ np.asarray(A, dtype=np.float64)
    final_S_W += float(np.sum(_BA ** 2))
final_S_W *= (alpha / (rank ** 0.5)) ** 2
final_P_IC = final_S_IC / S_IC_joint
final_P_W = final_S_W / S_W_joint
info(f'Alpha-dial final: S_IC={final_S_IC:.4g} (P_IC={final_P_IC:.3f})  '
     f'S_W={final_S_W:.4g} (P_W={final_P_W:.3f})  '
     f'penIC={lambda_IC*final_P_IC:.4g} penW={lambda_W*final_P_W:.4g}')

pen_hist_path = f'{save_dir}/{prefix}_penalty_history.npz'
np.savez(pen_hist_path,
         penIC=np.array(penIC_history), penW=np.array(penW_history),
         S_IC=np.array(S_IC_history), S_W=np.array(S_W_history))
info(f'Saved penalty history -> {pen_hist_path}')

# Metadata
meta = {
    'init_date': date_train,
    'joint_mode': joint_mode,
    'correction_mode': correction_mode,
    'rank': rank,
    'alpha': alpha,
    'sigma': sigma,
    'lora_lr': lora_lr,
    'ic_lr': ic_lr,
    'lr_window_exponent': lr_window_exponent,
    'lr_warmup_steps': lr_warmup_steps,
    'largest_window_lr_mult': largest_window_lr_mult,
    'ic_laplacian_weight': ic_laplacian_weight,
    'laplacian_var_weights': laplacian_var_weights,
    'target': lora_target,
    'pred_steps': int(pred_steps),
    'n_steps_run': len(loss_history),
    'best_step': best_step,
    'best_loss': float(best_loss),
    'initial_loss': float(loss_history[0]) if loss_history else None,
    'control_loss': control_loss,
    'reduction_pct': float(reduction),
    'reduction_vs_control_pct': float(ctrl_improve),
    'elapsed_s': float(elapsed),
    'block_size': block_size if joint_mode == 'alternating' else None,
    'lora_products_path': lora_products_path,
    'expanding_windows': expanding_windows,
    'steps_per_window': steps_per_window if expanding_windows else None,
    'rho': rho,
    'Lambda': Lambda,
    'lambda_IC': lambda_IC,
    'lambda_W': lambda_W,
    'S_IC_joint': S_IC_joint,
    'S_W_joint': S_W_joint,
    'final_S_IC': final_S_IC,
    'final_S_W': final_S_W,
    'final_P_IC': final_P_IC,
    'final_P_W': final_P_W,
    'final_penIC': lambda_IC * final_P_IC,
    'final_penW': lambda_W * final_P_W,
}
if joint_mode == 'lora_then_ic':
    meta['phase1_steps'] = config.get('phase1_steps', n_optim_steps // 2)
    meta['phase1_best_loss'] = float(phase1_best_loss)

meta_path = f'{save_dir}/{prefix}_meta.json'
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)

info(f'Saved metadata -> {meta_path}')

# =====================================================================
# Extended evaluation (drift check) — run best result out to ps_eval
# =====================================================================
ps_eval = config.get('ps_eval', None)

if ps_eval is not None and ps_eval > pred_steps:
    info(f'\n--- Extended evaluation: {ps_eval} steps ({ps_eval/4:.0f} days) ---')

    # Re-extract data for the longer window
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("6h", f"{ps_eval * 6}h"),
        **asdict(task_config),
        justify=just,
    )

    # Normalize eval inputs with the SAME stats as training
    eval_norm, _ = _normalize_32(eval_inputs)

    # Apply best IC perturbation in normalized space
    best_ic_delta_norm = jax.tree_util.tree_map(
        lambda best, orig: best - orig, best_norm_inputs, orig_norm_inputs
    )
    # The delta is for the training window; eval inputs have the same IC structure
    eval_joint = jax.tree_util.tree_map(lambda e, d: e + d, eval_norm, best_ic_delta_norm)

    @jax.jit
    def eval_per_ts(trainable, base_params, norm_ic, targets, forcings):
        effective_params = _apply_corrections(trainable, base_params)
        denorm_ic = _denormalize(norm_ic, norm_stats)
        (per_ts_losses, _), _ = jitted.timestep_loss_fn.apply(
            effective_params, {}, jax.random.PRNGKey(0),
            model_config, task_config,
            denorm_ic, targets, forcings,
        )
        return per_ts_losses

    # 1. Control: base params, original (un-perturbed) normalized eval inputs
    info('  Running control...')
    _zero_trainable = jax.tree_util.tree_map(jnp.zeros_like, best_trainable)
    ctrl_eval = np.array(eval_per_ts(
        _zero_trainable, params, eval_norm, eval_targets, eval_forcings
    ))
    info(f'  Control done: total={ctrl_eval.sum():.4f}')

    # 2. Joint-LoRA-only: jointly-trained LoRA, original IC (no IC perturbation)
    info('  Running joint-LoRA-only (no IC pert)...')
    joint_lora_eval = np.array(eval_per_ts(
        best_trainable, params, eval_norm, eval_targets, eval_forcings
    ))
    info(f'  Joint-LoRA-only done: total={joint_lora_eval.sum():.4f}')

    # 3. IC-only: no LoRA, best IC perturbation
    info('  Running IC-only (no LoRA)...')
    ic_only_eval = np.array(eval_per_ts(
        _zero_trainable, params, eval_joint, eval_targets, eval_forcings
    ))
    info(f'  IC-only done: total={ic_only_eval.sum():.4f}')

    # 4. Joint: best LoRA + best IC
    info('  Running joint (LoRA + IC)...')
    joint_eval = np.array(eval_per_ts(
        best_trainable, params, eval_joint, eval_targets, eval_forcings
    ))
    info(f'  Joint done: total={joint_eval.sum():.4f}')

    # --- Per-window IC snapshots: forward-eval each to ps_eval (only if save_window_snapshots) ---
    if save_window_snapshots and window_ic_snapshots:
        info(f'\n  --- Per-window snapshot evals ({len(window_ic_snapshots)} windows) ---')
        for _wps, _snap in window_ic_snapshots:
            _ev = np.array(eval_per_ts(
                _zero_trainable, params,
                jax.tree_util.tree_map(lambda e, d: e + d, eval_norm, _snap),
                eval_targets, eval_forcings))
            _wd = f'{save_dir}/window_ps{_wps}'
            os.makedirs(_wd, exist_ok=True)
            np.savez(f'{_wd}/{prefix}_eval_{ps_eval}step.npz',
                     control=ctrl_eval, ic_only=_ev, ps_eval=ps_eval, ps_train=_wps)
            info(f'  [window snapshot] window_ps{_wps}: ic_only={_ev.squeeze().mean():.4f} '
                 f'({(1 - _ev.squeeze().mean() / ctrl_eval.squeeze().mean()) * 100:.1f}% vs ctrl)')

    # 5. Independently-trained LoRAs (optional; config key indep_lora_path)
    #    Apply deltaW products by walking the original param tree — preserves Haiku nesting.
    #    Supports single path (string) or multiple paths (list of {label, path} dicts).
    from collections.abc import Mapping as _Mapping

    def _apply_products_to_params(base_params, products):
        """Walk param tree, add deltaW to matching leaves."""
        def _walk(d, prefix=""):
            result = {}
            for k, v in d.items():
                full_key = f"{prefix}/{k}" if prefix else k
                if isinstance(v, _Mapping):
                    result[k] = _walk(v, full_key)
                elif full_key in products:
                    result[k] = v + jnp.array(products[full_key], dtype=jnp.float32)
                else:
                    result[k] = v
            return result
        return _walk(base_params)

    @jax.jit
    def eval_indep_per_ts(corrected_params, norm_ic, targets, forcings):
        denorm_ic = _denormalize(norm_ic, norm_stats)
        (per_ts_losses, _), _ = jitted.timestep_loss_fn.apply(
            corrected_params, {}, jax.random.PRNGKey(0),
            model_config, task_config,
            denorm_ic, targets, forcings,
        )
        return per_ts_losses

    # Build list of independent LoRA configs
    indep_lora_configs = []
    indep_lora_path = config.get('indep_lora_path', None)
    if isinstance(indep_lora_path, str):
        indep_lora_configs.append({'label': 'indep_lora', 'path': indep_lora_path})
    elif isinstance(indep_lora_path, list):
        indep_lora_configs = indep_lora_path

    indep_results = {}
    from lora_utils import load_lora_products as _load_lp
    for lora_cfg in indep_lora_configs:
        lbl = lora_cfg['label']
        pth = lora_cfg['path']
        if not os.path.exists(pth):
            info(f'  Skipping {lbl}: {pth} not found')
            continue
        info(f'  Loading {lbl} from {pth}')
        products = _load_lp(pth)
        corrected = _apply_products_to_params(params, products)
        info(f'  Running {lbl}...')
        result = np.array(eval_indep_per_ts(
            corrected, eval_norm, eval_targets, eval_forcings
        ))
        indep_results[lbl] = result
        info(f'  {lbl} done: total={result.sum():.4f}')

    # 6. Handoff experiment: joint LoRA+IC for ps_handoff steps, then base GC
    ps_handoff = config.get('ps_handoff', None)
    handoff_eval = None
    if ps_handoff is not None and ps_handoff < ps_eval:
        info(f'\n  --- Handoff at step {ps_handoff} ---')
        ps_remaining = ps_eval - ps_handoff

        # Slice eval-window data into seg1 and seg2 (guarantees seg1+seg2 = ps_eval)
        seg1_targets  = eval_targets.isel(time=slice(0, ps_handoff))
        seg1_forcings = eval_forcings.isel(time=slice(0, ps_handoff))
        seg2_targets  = eval_targets.isel(time=slice(ps_handoff, None))
        seg2_forcings = eval_forcings.isel(time=slice(ps_handoff, None))

        # Seg1 inputs: eval IC with joint IC perturbation (in norm space)
        seg1_denorm = _denormalize(eval_joint, norm_stats)

        # Per-ts losses for segment 1 (LoRA+IC)
        eff_params_seg1 = _apply_corrections(best_trainable, params)
        (seg1_per_ts, _), _ = jitted.timestep_loss_fn.apply(
            eff_params_seg1, {}, jax.random.PRNGKey(0),
            model_config, task_config,
            seg1_denorm, seg1_targets, seg1_forcings,
        )
        seg1_per_ts = np.array(seg1_per_ts)
        info(f'  Seg1 (LoRA+IC, {ps_handoff} steps): mean={seg1_per_ts.squeeze().mean():.4f}')

        # Forward pass to get state at handoff point
        seg1_preds, _ = jitted.run_forward.apply(
            eff_params_seg1, {}, jax.random.PRNGKey(0),
            model_config, task_config,
            seg1_denorm, seg1_targets, seg1_forcings,
        )

        # Construct handoff inputs: append the predictions, keep the last num_inputs time steps
        num_inputs = eval_inputs.sizes['time']
        pred_keys = list(set(seg1_preds.data_vars).intersection(set(eval_inputs.data_vars)))
        advanced = xarray.concat(
            [eval_inputs[pred_keys], seg1_preds[pred_keys]], dim='time'
        ).tail(time=num_inputs)
        handoff_inputs = eval_inputs.copy(deep=True)
        for key in pred_keys:
            handoff_inputs[key].values[:] = advanced[key].values

        # Segment 2: base GC from handoff state
        (seg2_per_ts, _), _ = jitted.timestep_loss_fn.apply(
            params, {}, jax.random.PRNGKey(0),
            model_config, task_config,
            handoff_inputs, seg2_targets, seg2_forcings,
        )
        seg2_per_ts = np.array(seg2_per_ts)
        info(f'  Seg2 (base GC, {ps_remaining} steps): mean={seg2_per_ts.squeeze().mean():.4f}')

        # Concatenate — should be exactly ps_eval total
        handoff_eval = np.concatenate([seg1_per_ts.squeeze(), seg2_per_ts.squeeze()])
        assert len(handoff_eval) == ps_eval, f'Handoff length {len(handoff_eval)} != ps_eval {ps_eval}'
        info(f'  Handoff total: mean={handoff_eval.mean():.4f} ({(1 - handoff_eval.mean()/ctrl_eval.squeeze().mean())*100:.1f}% vs ctrl)')

    # Save extended eval
    eval_path = f'{save_dir}/{prefix}_eval_{ps_eval}step.npz'
    save_dict = dict(
        control=ctrl_eval,
        joint_lora_only=joint_lora_eval,
        ic_only=ic_only_eval,
        joint=joint_eval,
        ps_eval=ps_eval,
        ps_train=pred_steps,
    )
    for lbl, arr in indep_results.items():
        save_dict[lbl] = arr
    if handoff_eval is not None:
        save_dict['joint_handoff'] = handoff_eval
        save_dict['ps_handoff'] = ps_handoff
    np.savez(eval_path, **save_dict)
    info(f'Saved extended eval -> {eval_path}')

    # Summary
    ctrl_mean = ctrl_eval.squeeze().mean()
    def _pct(arr):
        return (1 - arr.squeeze().mean() / ctrl_mean) * 100
    info(f'  Control mean:      {ctrl_mean:.4f}')
    info(f'  Joint-LoRA-only:   {joint_lora_eval.squeeze().mean():.4f} ({_pct(joint_lora_eval):.1f}% vs ctrl)')
    info(f'  IC-only:           {ic_only_eval.squeeze().mean():.4f} ({_pct(ic_only_eval):.1f}% vs ctrl)')
    info(f'  Joint (LoRA+IC):   {joint_eval.squeeze().mean():.4f} ({_pct(joint_eval):.1f}% vs ctrl)')
    for lbl, arr in indep_results.items():
        info(f'  {lbl:20s} {arr.squeeze().mean():.4f} ({_pct(arr):.1f}% vs ctrl)')

info('// Complete //')
