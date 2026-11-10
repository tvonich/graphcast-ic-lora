"""
lora_utils.py — LoRA weight-correction utilities for GraphCast

Convention (rsLoRA, Kalajdzievski 2023):
    deltaW = (alpha / sqrt(rank)) * B @ A,  W_eff = W_base + deltaW
    B: (d_out, rank) initialized to zeros; A: (rank, d_in) ~ N(0, sigma)
    At init: B = 0 → deltaW = 0. rsLoRA scaling keeps gradients stable
    across ranks.

Identifiability: save B@A products, not factors separately — BA = (BR)(R⁻¹A).
Initialization check: call verify_init() to confirm deltaW=0, shapes, rank consistency.
"""

import numpy as np
import jax.numpy as jnp
from collections.abc import Mapping


# ---------------------------------------------------------------------------
# Internal helpers — param dict flattening/unflattening
# ---------------------------------------------------------------------------

def flatten_params(params):
    """
    Flatten nested Haiku param dict to {"/".join(key_path): array}.

    Same convention as hk.data_structures.to_flat_dict() but returns
    '/' separated string keys (not tuples) for easier filtering.

    Example: {'GraphCast': {'~': {'linear': {'w': array}}}}
             → {'GraphCast/~/linear/w': array}
    """
    result = {}

    def _recurse(d, prefix):
        for k, v in d.items():
            full_key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, Mapping):
                _recurse(v, full_key)
            else:
                result[full_key] = v

    _recurse(params, "")
    return result


def _unflatten_params(flat_dict):
    """
    Rebuild nested dict from flat dict with '/' separated keys.

    Inverse of flatten_params(). Used in apply_lora() to return a param
    dict with the same nesting structure that loss_fn.apply() expects.
    """
    result = {}
    for key, value in flat_dict.items():
        parts = key.split("/")
        d = result
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return result


# ---------------------------------------------------------------------------
# Core LoRA functions
# ---------------------------------------------------------------------------

def get_target_keys(params, target='processor_all', verbose=True):
    """
    Select weight matrix keys for LoRA based on a target configuration.

    Targets:
        'processor_all'         — 64 keys: all mesh_gnn /w (16 steps × 4 MLPs)
        'processor_w0_only'     — 32 keys: linear_0/w only (input projections)
        'processor_and_decoder' — 74 keys: processor + all mesh2grid_gnn /w
        'decoder_only'          — 10 keys: all mesh2grid_gnn /w

    Note: decoder has a (4, 512) matrix; max usable rank = 4 for any target
    set containing decoder keys. Returns sorted list of flat '/' keys.
    """
    flat = flatten_params(params)

    # --- mesh_gnn processor: the 16-step message-passing core ---
    # Keys: mesh_gnn/~_networks_builder/processor_{edges|nodes}_{0-15}_...mlp/~/linear_{0|1}/w
    processor_keys = sorted([
        k for k in flat
        if k.startswith("mesh_gnn/")
        and "processor_" in k
        and k.endswith("/w")
    ])

    if len(processor_keys) != 64:
        raise ValueError(
            f"Expected 64 mesh_gnn processor /w keys, found {len(processor_keys)}.\n"
            "Checkpoint structure may differ from GraphCast_small.\n"
            "Call print_param_keys(params, 'processor') to inspect."
        )

    # --- mesh2grid_gnn decoder: all weight matrices in the decoder block ---
    # Includes edge encoder, single-step processor, and output node MLP.
    decoder_keys = sorted([
        k for k in flat
        if k.startswith("mesh2grid_gnn/")
        and k.endswith("/w")
    ])

    if target == 'processor_all':
        keys = processor_keys

    elif target == 'processor_w0_only':
        keys = [k for k in processor_keys if "linear_0/w" in k]
        if len(keys) != 32:
            raise ValueError(f"Expected 32 W0-only keys, found {len(keys)}.")

    elif target == 'node_w0_only':
        # Node W₀ (linear_0/w) across all 16 processor steps = 16 keys.
        # Shape: (1024, 512) — input projection reading [node_feat ∥ aggregated_edge_feat].
        # Motivation: test whether regime corrections concentrate in node input weighting.
        keys = [k for k in processor_keys if "nodes" in k and "linear_0" in k]
        if len(keys) != 16:
            raise ValueError(f"Expected 16 node_w0_only keys, found {len(keys)}.")

    elif target == 'l0_only':
        # All 4 matrices in processor step 0 only (2 edge + 2 node MLPs = 4 keys).
        # Motivation: test whether early-layer corrections alone drive improvement.
        keys = [k for k in processor_keys if "_edges_0_" in k or "_nodes_0_" in k]
        if len(keys) != 4:
            raise ValueError(f"Expected 4 l0_only keys, found {len(keys)}.")

    elif target == 'edges_l0_l15':
        # Edge W₀ + W₁ at processor steps 0 and 15 only = 4 keys.
        # Tests whether first/last layer edge corrections capture regime signal.
        keys = [k for k in processor_keys if "edges" in k
                and ("_edges_0_" in k or "_edges_15_" in k)]
        if len(keys) != 4:
            raise ValueError(f"Expected 4 edges_l0_l15 keys, found {len(keys)}.")

    elif target == 'nodes_l0_l15':
        # Node W₀ + W₁ at processor steps 0 and 15 only = 4 keys.
        keys = [k for k in processor_keys if "nodes" in k
                and ("_nodes_0_" in k or "_nodes_15_" in k)]
        if len(keys) != 4:
            raise ValueError(f"Expected 4 nodes_l0_l15 keys, found {len(keys)}.")

    elif target == 'w0_l0_l15':
        # Edge W₀ + Node W₀ at processor steps 0 and 15 only = 4 keys.
        # Covers both correction channels (message construction + integration)
        # but only input projections at boundary layers.
        keys = [k for k in processor_keys if "linear_0/w" in k
                and ("_0_" in k or "_15_" in k)]
        if len(keys) != 4:
            raise ValueError(f"Expected 4 w0_l0_l15 keys, found {len(keys)}.")

    elif target == 'nodes_even':
        # Node matrices at even-numbered processor steps only (0,2,4,...,14).
        # 8 steps × 2 matrices (linear_0, linear_1) = 16 keys.
        # At r=4 this matches processor_nodes r=2 param count (breadth vs depth test).
        import re as _re
        keys = []
        for k in processor_keys:
            if "nodes" not in k:
                continue
            m = _re.search(r'processor_nodes_(\d+)_', k)
            if m and int(m.group(1)) % 2 == 0:
                keys.append(k)
        if len(keys) != 16:
            raise ValueError(f"Expected 16 nodes_even keys, found {len(keys)}.")

    elif target == 'edges_even':
        # Edge matrices at even-numbered processor steps only (0,2,4,...,14).
        # 8 steps × 2 matrices (linear_0, linear_1) = 16 keys.
        # Symmetric counterpart to nodes_even.
        import re as _re
        keys = []
        for k in processor_keys:
            if "edges" not in k:
                continue
            m = _re.search(r'processor_edges_(\d+)_', k)
            if m and int(m.group(1)) % 2 == 0:
                keys.append(k)
        if len(keys) != 16:
            raise ValueError(f"Expected 16 edges_even keys, found {len(keys)}.")

    elif target == 'processor_edges':
        keys = [k for k in processor_keys if "edges" in k]
        if len(keys) != 32:
            raise ValueError(f"Expected 32 processor_edges keys, found {len(keys)}.")

    elif target == 'processor_nodes':
        keys = [k for k in processor_keys if "nodes" in k]
        if len(keys) != 32:
            raise ValueError(f"Expected 32 processor_nodes keys, found {len(keys)}.")

    elif target == 'l0_node_w0':
        # THE SMALLEST MEANINGFUL ADAPTER: one matrix, the node-MLP input projection of
        # processor step 0. Shape (1024, 512) -> 1,536 params at r=1, i.e. 1.7% of
        # processor_all and 27% of l0_only. Bottom rung of the depth/width ladder; below
        # this you leave low-rank entirely (diagonal, scalar-per-layer, bias-only).
        keys = [k for k in processor_keys if "_nodes_0_" in k and "linear_0" in k]
        if len(keys) != 1:
            raise ValueError(f"Expected 1 l0_node_w0 key, found {len(keys)}.")

    elif target in ('layers_first_half', 'layers_first_quarter'):
        # Depth-truncated processor: all 4 matrices (2 edge + 2 node MLPs) of the FIRST
        # n message-passing steps only. Gives a parameter ladder at fixed rank that is
        # orthogonal to the rank ladder -- 'how deep does the correction need to reach'
        # rather than 'how many directions per layer'.
        #   layers_first_quarter -> steps 0-3   = 16 keys
        #   layers_first_half    -> steps 0-7   = 32 keys
        # (steps 0-0 = 4 keys is the existing 'l0_only'; all 16 steps = 'processor_all')
        n = 4 if target == 'layers_first_quarter' else 8
        want = {f"_edges_{i}_" for i in range(n)} | {f"_nodes_{i}_" for i in range(n)}
        keys = [k for k in processor_keys if any(w in k for w in want)]
        if len(keys) != 4 * n:
            raise ValueError(f"Expected {4*n} {target} keys, found {len(keys)}.")

    elif target == 'processor_and_decoder':
        keys = processor_keys + decoder_keys

    elif target == 'decoder_only':
        keys = decoder_keys

    else:
        raise ValueError(
            f"Unknown target '{target}'. Options: 'processor_all', "
            "'processor_edges', 'processor_nodes', 'nodes_even', "
            "'processor_w0_only', 'node_w0_only', 'l0_only', "
            "'edges_l0_l15', 'nodes_l0_l15', 'w0_l0_l15', "
            "'edges_even', 'nodes_even', "
            "'processor_and_decoder', 'decoder_only'."
        )

    if verbose:
        print(f"Target: '{target}' → {len(keys)} weight matrices")
        proc_in_keys = [k for k in keys if k.startswith("mesh_gnn/")]
        dec_in_keys  = [k for k in keys if k.startswith("mesh2grid_gnn/")]
        if proc_in_keys:
            edge_k = [k for k in proc_in_keys if "edges" in k]
            node_k = [k for k in proc_in_keys if "nodes" in k]
            print(f"  Processor:  {len(proc_in_keys)}  (edge={len(edge_k)}, node={len(node_k)})")
        if dec_in_keys:
            print(f"  Decoder:    {len(dec_in_keys)}")
        total_params = sum(flat[k].size for k in keys)
        print(f"  Total weight params covered: {total_params:,}")

    return keys


def init_lora_factors(base_params, target_keys, rank, sigma, seed=42):
    """
    Initialize LoRA B and A matrices for each target weight matrix.

    B = zeros (deltaW = 0 at init); A ~ N(0, sigma) (breaks symmetry for gradients).
    Default in optimize_ic_lora.py: sigma = 0.01/sqrt(rank).

    Returns {flat_key: (B, A)} — JAX pytree for optax. A matrices drawn from
    seeded np.random.default_rng; sorted target_keys ensure stable ordering.
    """
    rng = np.random.default_rng(seed)
    flat_base = flatten_params(base_params)
    lora_factors = {}

    for key in target_keys:
        if key not in flat_base:
            raise KeyError(
                f"Target key '{key}' not found in base_params.\n"
                "Re-run get_target_keys(params) to get valid keys."
            )

        W = flat_base[key]

        if W.ndim != 2:
            raise ValueError(
                f"Key '{key}' has shape {W.shape} — LoRA requires 2D weight "
                "matrices (ndim=2). Remove this key from target_keys."
            )

        d_out, d_in = W.shape
        B = jnp.zeros((d_out, rank), dtype=jnp.float32)
        A = jnp.array(rng.normal(0.0, sigma, (rank, d_in)), dtype=jnp.float32)
        lora_factors[key] = (B, A)

    return lora_factors


def compute_lora_corrections(lora_factors, alpha, rank):
    """
    Compute deltaW = (alpha / sqrt(rank)) * B @ A for each target matrix.

    rsLoRA scaling keeps gradient magnitude stable across ranks.
    At init (B=0): every deltaW is exactly 0.0.
    Called inside apply_lora() inside the JIT-traced gradient function.
    """
    scale = jnp.float32(alpha / (rank ** 0.5))
    return {
        key: scale * (B @ A)
        for key, (B, A) in lora_factors.items()
    }


def apply_lora(base_params, lora_factors, alpha, rank):
    """
    Compute W_eff = W_base + deltaW for each target matrix.

    IMPORTANT: call INSIDE jax.value_and_grad — calling outside treats
    lora_factors as constants (zero gradients). Corrections in F32;
    BF16 cast happens inside loss_fn.apply() downstream.

    Walks the original param tree and replaces only corrected leaves,
    preserving the exact nesting structure that loss_fn.apply() expects.
    (Haiku module names contain '/' — cannot split on '/' to unflatten.)
    """
    corrections = compute_lora_corrections(lora_factors, alpha, rank)

    def _apply(d, prefix=""):
        result = {}
        for k, v in d.items():
            full_key = f"{prefix}/{k}" if prefix else k
            if isinstance(v, Mapping):
                result[k] = _apply(v, full_key)
            elif full_key in corrections:
                result[k] = v + corrections[full_key]
            else:
                result[k] = v
        return result

    return _apply(base_params)


def save_lora_products(lora_factors, path, alpha, rank):
    """
    Save deltaW products (not A/B separately — identifiability) to .npz.
    Key encoding: '/' → '__'. ~1.4 MB at rank=4, 64 matrices.
    """
    # Guard: '/' → '__' encoding breaks if keys naturally contain '__'
    for key in lora_factors:
        if "__" in key:
            raise ValueError(
                f"Key '{key}' contains '__', which collides with the "
                "save encoding. Use a different separator in save/load."
            )

    scale = float(alpha) / float(rank) ** 0.5
    products = {}
    for key, (B, A) in lora_factors.items():
        deltaW = (scale * (np.array(B) @ np.array(A))).astype(np.float32)
        safe_key = key.replace("/", "__")
        products[safe_key] = deltaW
    np.savez(path, **products)


def save_lora_factors(lora_factors, path):
    """
    Save raw B and A matrices separately (not the product).
    Key encoding: '/' → '__', suffix '__B' or '__A'.
    Shapes at rank=4: B is (m, 4), A is (4, n).
    """
    for key in lora_factors:
        if "__" in key:
            raise ValueError(
                f"Key '{key}' contains '__', which collides with the "
                "save encoding. Use a different separator in save/load."
            )
    arrays = {}
    for key, (B, A) in lora_factors.items():
        safe_key = key.replace("/", "__")
        arrays[f'{safe_key}__B'] = np.array(B, dtype=np.float32)
        arrays[f'{safe_key}__A'] = np.array(A, dtype=np.float32)
    np.savez(path, **arrays)


def load_lora_factors(path):
    """
    Load raw B and A matrices saved by save_lora_factors().

    Returns {key: (B, A)} with '/' separator restored.
    Values are JAX arrays (float32).
    """
    data = np.load(path)
    factors = {}
    seen = set()
    for k in data.files:
        if k.endswith('__B'):
            base = k[:-3]  # strip __B
            if base not in seen:
                seen.add(base)
                B = data[f'{base}__B']
                A = data[f'{base}__A']
                orig_key = base.replace("__", "/")
                factors[orig_key] = (jnp.array(B, dtype=jnp.float32),
                                     jnp.array(A, dtype=jnp.float32))
    return factors


def load_lora_products(path):
    """
    Load deltaW products saved by save_lora_products().

    Returns {flat_key: deltaW_array} with '/' separator restored.
    Values are numpy float32 arrays.
    """
    data = np.load(path)
    return {k.replace("__", "/"): data[k] for k in data.files}


# ---------------------------------------------------------------------------
# Diagnostics — initialization check ("Gate 0") and analysis helpers
# ---------------------------------------------------------------------------

def print_param_keys(params, filter_str=None):
    """
    Print all flattened param keys and shapes.

    Use when get_target_keys() fails — inspect actual key names to fix filters.
    filter_str: if provided, only print keys containing this string.
    """
    flat = flatten_params(params)
    count = 0
    for k, v in sorted(flat.items()):
        if filter_str is None or filter_str in k:
            print(f"  {k:80s}  {v.shape}")
            count += 1
    print(f"\nShowing {count} / {len(flat)} total leaves")


def verify_init(base_params, lora_factors, alpha, rank):
    """
    Gate 0: verify deltaW=0 at init, deltaW shape matches W_base,
    B/A shapes consistent with rank. Returns True if all pass.
    """
    corrections = compute_lora_corrections(lora_factors, alpha, rank)
    flat_base = flatten_params(base_params)

    print(f"\nGate 0 — LoRA init verification (rank={rank}, alpha={alpha})")
    print(f"Target matrices: {len(corrections)}")
    all_pass = True

    for key, deltaW in corrections.items():
        # Check 1: zero at init
        max_val = float(jnp.max(jnp.abs(deltaW)))
        if max_val > 1e-10:
            print(f"  FAIL [zero init]:  {key}")
            print(f"    max|deltaW| = {max_val:.2e}  (expected 0.0)")
            all_pass = False

        # Check 2: shape matches base
        W_base = flat_base[key]
        if deltaW.shape != W_base.shape:
            print(f"  FAIL [shape mismatch]:  {key}")
            print(f"    deltaW={deltaW.shape}  W_base={W_base.shape}")
            all_pass = False

        # Check 3: A and B shapes
        B, A = lora_factors[key]
        d_out, d_in = W_base.shape
        if B.shape != (d_out, rank):
            print(f"  FAIL [B shape]:  {key}  B={B.shape}  expected=({d_out},{rank})")
            all_pass = False
        if A.shape != (rank, d_in):
            print(f"  FAIL [A shape]:  {key}  A={A.shape}  expected=({rank},{d_in})")
            all_pass = False

    if all_pass:
        n = len(corrections)
        total_lora_params = sum(B.size + A.size for B, A in lora_factors.values())
        total_base_params = sum(v.size for v in flat_base.values())
        print(f"  PASS: {n} matrices verified.")
        print(f"  LoRA params:  {total_lora_params:>10,}")
        print(f"  Base params:  {total_base_params:>10,}  (frozen)")
        print(f"  LoRA/Base:    {total_lora_params/total_base_params:.2%}")
    else:
        print("\n  Some checks FAILED — do not proceed to GPU until resolved.")

    return all_pass
