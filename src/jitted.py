# jitted.py — GraphCast forward, loss and gradient functions (Haiku-transformed, JAX-jitted).
import functools
from load_model import params, model_config, task_config
from load_model import diffs_stddev_by_level, mean_by_level, stddev_by_level

from graphcast import graphcast
from graphcast import casting
from graphcast import normalization
from graphcast import autoregressive
from graphcast import xarray_tree
from graphcast import xarray_jax

import jax
import xarray as xr
import jax.numpy as jnp
from graphcast import xarray_jax
from optax import apply_updates
from functools import partial
from haiku import transform_with_state as tws
from os import getenv
from json import load as jload
import numpy as np
import haiku as hk

# Access the j_path environment variable and get configs
j_path = getenv('j_path')
with open(j_path, 'r') as f:
    config = jload(f)

selected_times = config['selected_times']
use_f32 = config.get('use_f32', False)  # BF16 by default; set True in config for F32

state={} #Needed for final functions on this page

from collections.abc import Mapping

def _to_mutable(tree):
    if hasattr(hk.data_structures, "to_mutable_dict"):
        return hk.data_structures.to_mutable_dict(tree)
    return tree

def _flatten_nested(d, prefix=()):
    for k, v in d.items():
        if isinstance(v, Mapping):
            yield from _flatten_nested(v, prefix + (k,))
        else:
            key = "/".join(prefix + (k,))
            yield key, np.asarray(v)

def denormalize_64(dt, stats):
    """
    Denormalize an xarray.Dataset using provided statistics.
    Uses float64 precision for arithmetic operations to maintain numerical stability.
    Parameters:
    - dt (xr.Dataset): The dataset to denormalize.
    - stats (dict): A dictionary containing 'mean' and 'std' for each variable.
    Returns:
    - xr.Dataset: The denormalized dataset in float32.
    """
    dt_copy = dt.copy()  # Keep data in its original dtype (probably float32)

    dynamic_vars = [
        "geopotential", "temperature", "u_component_of_wind", "v_component_of_wind",
        "specific_humidity", "vertical_velocity", "2m_temperature", "mean_sea_level_pressure",
        "10m_v_component_of_wind", "10m_u_component_of_wind"
    ]
    time_vars =['toa_incident_solar_radiation','day_progress_sin','day_progress_cos','year_progress_cos','year_progress_sin'
    ]

    # Include 'total_precipitation_6hr' if present
    if "total_precipitation_6hr" in dt_copy.data_vars:
        dynamic_vars.append("total_precipitation_6hr")

    for var_name in time_vars:
       dt_copy[var_name] = dt_copy[var_name].astype('float64')

    for var_name in dynamic_vars:
      da = dt_copy[var_name]
      mean = stats[var_name]['mean'].astype('float64')
      std = stats[var_name]['std'].astype('float64')

      # Temporarily cast data to float64 for computation
      da_64 = da.astype('float64')
      denormalized = da_64 * std + mean

      # Cast back to float32
      dt_copy[var_name] = denormalized

    return dt_copy

def denormalize_32(dt, stats):
    """
    Denormalize an xarray.Dataset using provided statistics.
    Uses float64 precision for arithmetic operations to maintain numerical stability.
    Parameters:
    - dt (xr.Dataset): The dataset to denormalize.
    - stats (dict): A dictionary containing 'mean' and 'std' for each variable.
    Returns:
    - xr.Dataset: The denormalized dataset in float32.
    """
    dt_copy = dt.copy()  # Keep data in its original dtype (probably float32)

    dynamic_vars = [
        "geopotential", "temperature", "u_component_of_wind", "v_component_of_wind",
        "specific_humidity", "vertical_velocity", "2m_temperature", "mean_sea_level_pressure",
        "10m_v_component_of_wind", "10m_u_component_of_wind"]

    # Include 'total_precipitation_6hr' if present
    if "total_precipitation_6hr" in dt_copy.data_vars:
        dynamic_vars.append("total_precipitation_6hr")

    for var_name in dynamic_vars:
      da = dt_copy[var_name]
      mean = stats[var_name]['mean'].astype('float32')
      std = stats[var_name]['std'].astype('float32')

      # Temporarily cast data to float64 for computation
      da_32 = da.astype('float32')
      denormalized = da_32 * std + mean

      # Cast back to float32
      dt_copy[var_name] = denormalized.astype('float32')

    return dt_copy

def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig,
    force_f32: bool = False):
  """Constructs and wraps the GraphCast Predictor.

  force_f32: if True, disables the BFloat16 cast regardless of the global
  `use_f32` config setting. Used by evaluation paths (e.g. multi_rmse_fn)
  where reproducible per-variable RMSE matters more than throughput.
  """
  # Deeper one-step predictor.
  predictor = graphcast.GraphCast(model_config, task_config)

  # Modify inputs/outputs to `graphcast.GraphCast` to handle conversion to
  # from/to float32 to/from BFloat16. With force_f32=True the BF16 cast is
  # disabled even when the global use_f32 flag is False.
  bf16_enabled = (not use_f32) and (not force_f32)
  predictor = casting.Bfloat16Cast(predictor, enabled=bf16_enabled)

  # Modify inputs/outputs to `casting.Bfloat16Cast` so the casting to/from
  # BFloat16 happens after applying normalization to the inputs/targets.
  predictor = normalization.InputsAndResiduals(
      predictor,
      diffs_stddev_by_level=diffs_stddev_by_level,
      mean_by_level=mean_by_level,
      stddev_by_level=stddev_by_level)

  # Wraps everything so the one-step model can produce trajectories.
  predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True) 
  return predictor

@tws
def run_forward(model_config, task_config, inputs, targets_template, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config)
  return predictor(inputs, targets_template=targets_template, forcings=forcings)

#Original
@tws 
def loss_fn(model_config, task_config, inputs, targets, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config) 
  loss, diagnostics = predictor.loss(inputs, targets, forcings) #defines loss -- subfunction of predictor
  return xarray_tree.map_structure(
      lambda x: xarray_jax.unwrap_data(x.mean(), require_jax=True),
      (loss, diagnostics))

@tws 
def timestep_loss_fn(model_config, task_config, inputs, targets, forcings):
  predictor = construct_wrapped_graphcast(model_config, task_config) # per-timestep loss (see autoregressive.py)
  per_timestep_losses, per_timestep_diagnostics = predictor.per_timestep_loss_and_diagnostics(inputs, targets, forcings) #defines loss -- subfunction of predictor
  return per_timestep_losses, per_timestep_diagnostics

#Original
def grads_fn(params, state, model_config, task_config, inputs, targets, forcings):
  def _aux(params, state, i, t, f):
    (loss, diagnostics), next_state = loss_fn.apply(
        params, state, jax.random.PRNGKey(0), model_config, task_config,
        i, t, f)
    return loss, (diagnostics, next_state)
  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True)(params, state, inputs, targets, forcings)
  return loss, diagnostics, next_state, grads

def norm_grads64_fn(params, state, model_config, task_config, inputs, targets, forcings, stats, rng):
  def _aux(params, state, i, s, t, f):
    denorm_input = denormalize_64(i,s)
    (loss, diagnostics), next_state = loss_fn.apply( 
        params, state, rng, model_config, task_config,
        denorm_input, t, f)
    return loss, (diagnostics, next_state)
  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True, argnums=2)(params, state, inputs, stats, targets, forcings) #argnum=2 for respect to inputs
  return loss, diagnostics, grads

def norm_grads32_fn(params, state, model_config, task_config, inputs, targets, forcings, stats, rng):
  def _aux(params, state, i, s, t, f):
    denorm_input = denormalize_32(i,s)
    (loss, diagnostics), next_state = loss_fn.apply( 
        params, state, rng, model_config, task_config,
        denorm_input, t, f)
    return loss, (diagnostics, next_state)
  (loss, (diagnostics, next_state)), grads = jax.value_and_grad(
      _aux, has_aux=True, argnums=2)(params, state, inputs, stats, targets, forcings) #argnum=2 for respect to inputs
  return loss, diagnostics, grads

def update_inputs(opt_update, grads, opt_state, inputs):
    updates, new_opt_state = opt_update(grads, opt_state, inputs)
    new_inputs = apply_updates(inputs, updates)
    return new_inputs, new_opt_state

def update_params(opt_update, grads, opt_state, params):
    """Standard parameter update function (GraphCast style)."""
    updates, new_opt_state = opt_update(grads, opt_state, params)
    new_params = apply_updates(params, updates)
    return new_params, new_opt_state


@tws
def mse_fn(
    model_config,
    task_config,
    inputs,
    targets,
    forcings,
    var_name: str = "geopotential",
    level=500,
    use_index: bool = False,
    area_weighted: bool = True,
):
  """
  Per-timestep MSE for a single variable/level, averaged over space (and batch if present).
  RETURNS: jax.Array with shape (time,), NOT an xarray.DataArray (to stay JIT-safe).
  Assumes dims like: (batch?, time, [level], lat, lon)
  """
  predictor = construct_wrapped_graphcast(model_config, task_config)
  out = predictor(inputs, targets_template=targets, forcings=forcings)
  preds = out[0] if isinstance(out, tuple) else out

  if var_name not in preds or var_name not in targets:
    raise ValueError(f"Variable '{var_name}' must exist in both predictions and targets.")

  pred_var = preds[var_name]
  targ_var = targets[var_name]

  if "time" not in pred_var.dims:
    raise ValueError(
      f"Variable '{var_name}' does not have a 'time' dimension in predictions; "
      "it looks static or not predicted. Choose a dynamic variable."
    )

  # Level selection if present
  if "level" in pred_var.dims:
    if use_index:
      pred_sel = pred_var.isel(level=level)
      targ_sel = targ_var.isel(level=level)
    else:
      pred_sel = pred_var.sel(level=level)
      targ_sel = targ_var.sel(level=level)
  else:
    pred_sel = pred_var
    targ_sel = targ_var

  # Squared error (xarray) with dims e.g. (batch?, time, lat, lon)
  se = (pred_sel - targ_sel) ** 2

  # Expect spatial dims
  if not all(d in se.dims for d in ("lat", "lon")):
    raise ValueError("Expected 'lat' and 'lon' dims for spatial averaging.")

  # === Switch to pure JAX and operate by axis indices ===
  dims = list(se.dims)
  se_vals = xarray_jax.unwrap_data(se, require_jax=True)  # jax.Array

  # Reduce any non-spatial, non-time dims (e.g., batch)
  keep = {"time", "lat", "lon"}
  reduce_axes = [dims.index(d) for d in dims if d not in keep]
  # reduce higher axes first to keep indices valid
  for ax in sorted(reduce_axes, reverse=True):
    se_vals = jnp.nanmean(se_vals, axis=ax)
    dims.pop(ax)

  # Mean over longitude
  lon_ax = dims.index("lon")
  se_vals = jnp.nanmean(se_vals, axis=lon_ax)
  dims.pop(lon_ax)  # now dims should be ['time','lat'] in that order (usually)

  # At this point: se_vals has dims == dims (likely ['time','lat'])
  # Area weighting over lat (optional)


  # area weighting over lat (optional)
  # Area weighting over lat (optional)
  if area_weighted:
      try:
          lat_da = se["lat"]
          lat_host = xarray_jax.unwrap_data(lat_da, require_jax=False)
          lat_vals = jnp.asarray(lat_host, dtype=se_vals.dtype)

          # compute cos(lat) weights
          w_lat = jnp.cos(jnp.deg2rad(lat_vals))
          lat_ax = dims.index("lat")

          # broadcast to match se_vals shape
          if lat_ax == 1:
              w_b = w_lat[None, ...]  # (1, lat)
          else:
              w_b = w_lat  # (..., lat)

          mask = jnp.isfinite(se_vals).astype(se_vals.dtype)
          num = jnp.nansum(se_vals * w_b, axis=lat_ax)
          den = jnp.nansum(w_b * mask, axis=lat_ax)
          den = jnp.where(den == 0, 1.0, den)
          mse_time = num / den

      except Exception as e:
          import logging
          logger = logging.getLogger(__name__)
          logger.error(f"[mse_fn] Area weighting failed: {type(e).__name__}: {e}")
          raise RuntimeError("Area weighting computation failed.") from e

  else:
      lat_ax = dims.index("lat")
      mse_time = jnp.nanmean(se_vals, axis=lat_ax)



  # dims should now be ['time']
  return mse_time  # jax.Array (time,)


@tws
def multi_rmse_fn(
    model_config,
    task_config,
    inputs,
    targets,
    forcings,
    var_levels: tuple = (("geopotential", 500),),
    use_index: bool = False,
    area_weighted: bool = True,
):
  """
  Per-timestep cos-lat-weighted RMSE for multiple (var_name, level) pairs,
  computed in a SINGLE forward pass. Returns shape (n_pairs, time).

  - Always runs the forward pass in F32 (force_f32=True) for reproducible
    per-variable error values, regardless of the global use_f32 setting.
  - Returns RMSE (not MSE) in each variable's native units, except geopotential
    which is divided by g=9.80665 to give RMSE in meters of geopotential height.

  var_levels: tuple of (var_name, level) tuples (hashable, static at JIT time), e.g.
     (("geopotential", 500), ("temperature", 500), ("u_component_of_wind", 500),
      ("geopotential", 1000), ("temperature", 1000), ("u_component_of_wind", 1000))
  """
  predictor = construct_wrapped_graphcast(model_config, task_config, force_f32=True)
  out = predictor(inputs, targets_template=targets, forcings=forcings)
  preds = out[0] if isinstance(out, tuple) else out
  G_STD = 9.80665  # m/s^2; geopotential -> geopotential height in meters

  results = []
  for var_name, level in var_levels:
    if var_name not in preds or var_name not in targets:
      raise ValueError(f"Variable '{var_name}' must exist in both predictions and targets.")

    pred_var = preds[var_name]
    targ_var = targets[var_name]

    if "time" not in pred_var.dims:
      raise ValueError(
        f"Variable '{var_name}' does not have a 'time' dimension in predictions; "
        "it looks static or not predicted. Choose a dynamic variable."
      )

    if "level" in pred_var.dims:
      if use_index:
        pred_sel = pred_var.isel(level=level)
        targ_sel = targ_var.isel(level=level)
      else:
        pred_sel = pred_var.sel(level=level)
        targ_sel = targ_var.sel(level=level)
    else:
      pred_sel = pred_var
      targ_sel = targ_var

    se = (pred_sel - targ_sel) ** 2
    if not all(d in se.dims for d in ("lat", "lon")):
      raise ValueError("Expected 'lat' and 'lon' dims for spatial averaging.")

    dims = list(se.dims)
    se_vals = xarray_jax.unwrap_data(se, require_jax=True)

    # Reduce non-spatial, non-time dims (e.g., batch).
    keep = {"time", "lat", "lon"}
    reduce_axes = [dims.index(d) for d in dims if d not in keep]
    for ax in sorted(reduce_axes, reverse=True):
      se_vals = jnp.nanmean(se_vals, axis=ax)
      dims.pop(ax)

    # Mean over longitude.
    lon_ax = dims.index("lon")
    se_vals = jnp.nanmean(se_vals, axis=lon_ax)
    dims.pop(lon_ax)

    # Cos-lat weighted mean over latitude.
    if area_weighted:
      lat_da = se["lat"]
      lat_host = xarray_jax.unwrap_data(lat_da, require_jax=False)
      lat_vals = jnp.asarray(lat_host, dtype=se_vals.dtype)
      w_lat = jnp.cos(jnp.deg2rad(lat_vals))
      lat_ax = dims.index("lat")
      w_b = w_lat[None, ...] if lat_ax == 1 else w_lat
      mask = jnp.isfinite(se_vals).astype(se_vals.dtype)
      num = jnp.nansum(se_vals * w_b, axis=lat_ax)
      den = jnp.nansum(w_b * mask, axis=lat_ax)
      den = jnp.where(den == 0, 1.0, den)
      mse_time = num / den
    else:
      lat_ax = dims.index("lat")
      mse_time = jnp.nanmean(se_vals, axis=lat_ax)

    # MSE -> RMSE; geopotential -> meters of geopotential height
    rmse_time = jnp.sqrt(mse_time)
    if var_name == "geopotential":
      rmse_time = rmse_time / G_STD

    results.append(rmse_time)

  return jnp.stack(results, axis=0)  # (n_pairs, time)


def _grads_fn_bound(params, state, inputs, targets, forcings):
    # Call the original function with ALL 7 args explicitly.
    return grads_fn(params, state, model_config, task_config, inputs, targets, forcings)


# Jax doesn't seem to like passing configs as args through the jit. Passing it
# in via partial (instead of capture by closure) forces jax to invalidate the
# jit cache if you change configs.

def with_configs(fn):
  return partial(
      fn, model_config=model_config, task_config=task_config)

# Always pass params and state, so the usage below are simpler
def with_params(fn):
  return partial(fn, params=params, state=state)

# Our models aren't stateful, so the state is always empty, so just return the
# predictions. This is requiredy by our rollout code, and generally simpler.
def drop_state(fn):
  return lambda **kw: fn(**kw)[0]

init_jitted = jax.jit(with_configs(run_forward.init))
loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(loss_fn.apply))))
timestep_loss_fn_jitted = drop_state(with_params(jax.jit(with_configs(timestep_loss_fn.apply))))

# runtime signature will be: (params, state, inputs, targets, forcings)
grads_fn_jitted = jax.jit(_grads_fn_bound)
import inspect
assert len(inspect.signature(_grads_fn_bound).parameters) == 5

norm_grads32_fn_jitted = with_params(jax.jit(with_configs(norm_grads32_fn)))
norm_grads64_fn_jitted = with_params(jax.jit(with_configs(norm_grads64_fn)))

mse_fn_jitted = drop_state(
    with_params(
        jax.jit(
            with_configs(mse_fn.apply),
            static_argnames=("var_name", "level", "use_index", "area_weighted"),
        )
    )
)

# Multi-(var, level) variant: one forward pass, returns (n_pairs, time) RMSE in F32.
# Geopotential is returned as RMSE of geopotential height (meters).
multi_rmse_fn_jitted = drop_state(
    with_params(
        jax.jit(
            with_configs(multi_rmse_fn.apply),
            static_argnames=("var_levels", "use_index", "area_weighted"),
        )
    )
)

run_forward_jitted = drop_state(with_params(jax.jit(with_configs(
    run_forward.apply))))