# GraphCast Initial Condition and LoRA Optimization

> **Built on [GraphCast](https://github.com/google-deepmind/graphcast) by Google DeepMind.**
> This repository is a research extension of GraphCast and includes a modified copy of its code.
> If you use this code, please also cite the GraphCast paper (see [Citations](#citations)).

Gradient-based optimization of the **initial conditions (IC)** and of **low-rank (LoRA) weight corrections** for DeepMind's [GraphCast](https://github.com/google-deepmind/graphcast) weather model (GraphCast_small: 1.0°, 13 pressure levels). The optimizer minimizes GraphCast's own training loss (a weighted mean-squared error against ERA5) over an autoregressive forecast. It can adjust a perturbation to the initial condition, a rank-*r* correction to the model weights, or both at once. This repository ships one worked example: a **LoRA-only** optimization of the 10-day forecast started at **2017-01-01 12 UTC**. It is not affiliated with or endorsed by Google DeepMind.

This is research code developed on NCAR's Derecho supercomputer, but it runs on any machine with a CUDA 12 GPU and ERA5 data in GraphCast's format.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Quick Start](#quick-start)
4. [Configuration Reference](#configuration-reference)
5. [Output Files](#output-files)
6. [ERA5 Data Format](#era5-data-format)
7. [Design Notes](#design-notes)
8. [Reproducibility](#reproducibility)
9. [Changes to GraphCast](#changes-to-graphcast)
10. [Citations](#citations)
11. [License](#license)
12. [For Agents](#for-agents)

---

## Requirements

| Item | Requirement |
|------|-------------|
| GPU | NVIDIA GPU with CUDA 12. The example (float32, 10-day window) was tested on an A100 with 40 GB. |
| Host memory | The whole input file is loaded into memory. Peak use was about 100 GB with the 30 GB one-year file used for testing; a file trimmed to the needed time steps (see [ERA5 Data Format](#era5-data-format)) needs much less. |
| Python | 3.10, in the conda environment from `environment.yml` |
| CUDA | A CUDA 12 toolkit on the machine (see the note at the top of `environment.yml`); cuDNN 8.9 comes with the environment |
| Run time | About 25 minutes for the example on one A100 (150 optimization steps, then an 11-day evaluation) |

---

## Setup

### 1. Clone this repository

```bash
git clone https://github.com/tvonich/graphcast-ic-lora
cd graphcast-ic-lora
```

The modified GraphCast code is included in `graphcast/` (see [Changes to GraphCast](#changes-to-graphcast)). Do not clone or install DeepMind's `graphcast` package as well: the bundled copy must be the one that is imported, and `submit_jobs/run_lora_only.sh` puts it first on `PYTHONPATH`.

### 2. Download model weights and normalization statistics

These are published by DeepMind (license: CC BY-NC-SA 4.0) and are not included here. Download them into `params/` and `stats/`:

```bash
mkdir -p params stats
B=https://storage.googleapis.com/dm_graphcast
curl -o params/params_GraphCast_small.npz \
  "$B/params/GraphCast_small%20-%20ERA5%201979-2015%20-%20resolution%201.0%20-%20pressure%20levels%2013%20-%20mesh%202to5%20-%20precipitation%20input%20and%20output.npz"
for f in diffs_stddev_by_level mean_by_level stddev_by_level; do
  curl -o stats/$f.nc "$B/stats/$f.nc"
done
```

These are byte-identical to the files used for the results (checked by size and MD5). To keep them elsewhere, set `GRAPHCAST_PARAMS_PATH` and `GRAPHCAST_STATS_PATH`.

### 3. Create the conda environment

```bash
conda env create -f environment.yml
conda activate graphcast-ic-lora
```

This installs Python 3.10, JAX/jaxlib 0.4.28 (CUDA 12, cuDNN 8.9), dm-haiku 0.0.11, optax, xarray and the other dependencies at the tested versions. The pinned jaxlib also needs a CUDA 12 toolkit installed on the machine (see the note at the top of `environment.yml`; on HPC systems this is usually `module load cuda`).

### 4. Prepare the ERA5 input file

The example needs ERA5 on GraphCast's 1.0° grid from 2017-01-01 06 UTC to 2017-01-12 12 UTC, in the layout described in [ERA5 Data Format](#era5-data-format). Set `dataset_file_path` in `configs/lora_only_2017-01-01T12.json` to your file (default: `data/era5_1deg_13lvl_2017-01.nc`).

---

## Quick Start

After completing setup steps 1–4:

```bash
# On any machine with a GPU, from the repository root:
bash submit_jobs/run_lora_only.sh                      # the example configuration
bash submit_jobs/run_lora_only.sh path/to/other.json   # any other configuration

# Or as a PBS job (e.g. on NCAR Derecho), submitted from the repository root.
# First set your project code in the "#PBS -A" line of the script (or use qsub -A).
mkdir -p logs
qsub submit_jobs/run_lora_only.sh
qsub -v RUN_CONFIG=path/to/other.json submit_jobs/run_lora_only.sh
```

The script activates the conda environment named by `CONDA_ENV` (default `graphcast-ic-lora`; set `CONDA_ENV=""` to use the active environment as is). It then stops if JAX cannot see a GPU, sets the environment variables listed under [For Agents](#environment-variables), and runs `batch_modules/optimize_ic_lora.py` from the repository root.

A config path given on the command line is taken relative to the current directory. Relative paths inside the config (`dataset_file_path`, `save_root`) and the default `params/` and `stats/` locations are taken relative to the repository root. With PBS, the job log goes to `logs/`. A successful run ends its log with `// Complete //`.

---

## Configuration Reference

A run is configured by one JSON file. The table lists the keys of the example, `configs/lora_only_2017-01-01T12.json`, plus three optional keys. Steps are 6-hour forecast steps.

| Key | Example | Description |
|-----|---------|-------------|
| `init_date` | `"2017-01-01T12"` | Forecast start (`YYYY-MM-DDTHH`): the later of GraphCast's two input times. It must appear in the file's `datetime` coordinate, preceded by the time 6 h earlier. |
| `pred_steps` | `40` | Optimization window in steps (40 = 10 days). With `expanding_windows`, the largest window is used instead. |
| `justify` | `"left"` | Leave at `"left"`: the two input times are `init_date` − 6 h and `init_date`. |
| `joint_mode` | `"lora_only"` | What is optimized: `lora_only`, `ic_only`, `simultaneous` (IC and LoRA from one gradient pass), `lora_then_ic` or `alternating`. |
| `correction_mode` | `"lora"` | Type of weight correction; `"lora"` is the only one implemented. |
| `lora_rank`, `lora_alpha` | `2`, `2` | Rank *r* and scale α of ΔW = (α/√*r*)·B·A. |
| `lora_sigma` | (optional; default 0.01/√*r*) | Standard deviation of the initial entries of A. |
| `lora_target` | `"processor_all"` | Weight matrices that receive a correction: all 64 weight matrices of the 16 processor layers. Other choices are listed in `get_target_keys` in `batch_modules/lora_utils.py`. |
| `lora_lr`, `ic_lr` | `0.001`, `0.0005` | Base Adam learning rates for the LoRA factors and for the IC perturbation (`ic_lr` is unused in `lora_only`). |
| `expanding_windows` | `[8, 16, 24, 32, 40]` | Successive optimization windows in steps. Omit to optimize a single window of `pred_steps`. |
| `steps_per_window` | `30` | Optimization steps per window: one integer, or a list with one entry per window. |
| `n_optim_steps` | `150` | Number of steps when `expanding_windows` is not set (also used by `lora_then_ic`). |
| `patience` | `1000` | Stop a window after this many steps without a new lowest loss (1000: never, here). |
| `scale_lr_by_window`, `lr_window_exponent` | `true`, `0.5` | Learning rate in a window of T steps = base rate / max(T/4, 1)^exponent, i.e. divided by √(window length in days) here. |
| `largest_window_lr_mult` | `1.0` | Extra learning-rate factor for the last window (1.0 = none). |
| `lr_warmup_steps` | `0` | Length of a linear learning-rate ramp at the start of each window (0 = none). |
| `rho`, `Lambda` | `0.0`, `0.0` | Size penalty that trades IC against weight corrections: λ_IC = Λ·e^(−ρ/2) and λ_W = Λ·e^(ρ/2) multiply S_IC/`S_IC_joint` and S_W/`S_W_joint`. `Lambda` = 0 switches it off. |
| `S_IC_joint`, `S_W_joint` | (optional; defaults 489.5, 0.446) | Normalizers of the size penalty. Set them to `final_S_IC` and `final_S_W` from `meta.json` of an unpenalized `simultaneous` run for the same date. |
| `ic_laplacian_weight` | `0.0005` | Weight of the Laplacian smoothness penalty on the IC perturbation (no effect in `lora_only`). |
| `laplacian_var_weights` | `null` | Per-variable weights of that penalty; `null` = equal weights. If the key is left out, built-in weights are used instead (geopotential 3; winds, mean sea-level pressure and precipitation 0.1). |
| `log_every`, `per_ts_every` | `1`, `10` | Log the loss every N steps; record the loss at each lead time every N steps. |
| `ps_eval` | `44` | After optimization, evaluate the saved corrections over this many steps (must be larger than the window; `null` skips the evaluation). |
| `dataset_file_path` | `"data/era5_1deg_13lvl_2017-01.nc"` | ERA5 input file. |
| `save_root` | `"outputs/lora_only"` | Output directory. |
| `selected_vars`, `selected_lvls`, `selected_region` | `"all"` | Restrict the loss to a list of variables, a list of pressure levels (hPa), or a region `[lat_min, lat_max, lon_min, lon_max]` (latitude −90 to 90, longitude 0 to 359, minimum ≤ maximum). |
| `selected_times` | `"all"` | Not used: every step of the window counts in the loss. |
| `use_f32` | `true` | Run GraphCast in float32 (`false`: bfloat16, GraphCast's default). |
| `save_epoch_snapshots`, `epoch_save_log`, `epoch_save_every` | `false`, `0`, `1` | Optional IC snapshots during optimization (`ic_only` mode). |

**Variable names (GraphCast 1.0°, 13 levels):**
```
Pressure-level: geopotential, temperature, specific_humidity, u_component_of_wind,
                v_component_of_wind, vertical_velocity
Surface:        2m_temperature, mean_sea_level_pressure, 10m_u_component_of_wind,
                10m_v_component_of_wind, total_precipitation_6hr
```

**Pressure levels:** `[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]` hPa

---

## Output Files

Results go to `{save_root}/{init_date}/` (example: `outputs/lora_only/2017-01-01T12/`), with file names starting `2017-01-01T12_r2_lora_only_`:

| File | Contents |
|---|---|
| `lora_products.npz` | the weight corrections ΔW = (α/√r)·B·A, one array per targeted weight matrix (64 here) |
| `lora_factors.npz` | the B and A factors |
| `ic_delta.npz`, `ic_norm_delta.npz` | the IC perturbation in physical and z-scored units (all zero in `lora_only` mode) |
| `loss.npy` | loss at every optimization step (`loss_checkpoint.npy`: the same, written during the run) |
| `eval_44step.npz` | per-step loss over a 44-step (11-day) forecast for: the unmodified model (`control`), the corrected weights (`joint_lora_only`), the perturbed IC (`ic_only`) and both (`joint`) |
| `control_per_ts_loss.npy` | per-step loss of the unmodified model over the training forecast (40 steps) |
| `per_ts_loss.pkl` | per-step loss over the current window, every `per_ts_every` optimization steps (a pickled list of arrays, because the windows differ in length; saved as `.npy` when there is a single window) |
| `grad_norms.npz` | gradient norm at every optimization step (`lora_norms`, `ic_norms`) |
| `penalty_history.npz` | penalty terms at every step (joint modes only; empty in `lora_only` mode) |
| `meta.json` | run settings and summary numbers |

The saved corrections are those with the lowest training loss in the last (longest) window; `eval_44step.npz` evaluates them.

In `lora_only` mode the IC is not changed, so `ic_only` repeats `control` and `joint` repeats `joint_lora_only`. They still differ slightly (by 0.03 % and 0.3 % in the 44-step mean loss of the release check), because GPU computations are not bit-for-bit reproducible (see [Reproducibility](#reproducibility)).

**Loading the corrections:**
```python
import numpy as np
dW = np.load("outputs/lora_only/2017-01-01T12/2017-01-01T12_r2_lora_only_lora_products.npz")
# Keys are GraphCast parameter names with "/" replaced by "__"; the last "__" separates
# the module from the parameter (e.g. "...__linear_0__w" is parameter "w" of ".../linear_0").
```

---

## ERA5 Data Format

The script reads one netCDF file in GraphCast's dataset layout (the same layout as DeepMind's example files in `gs://dm_graphcast/dataset/`):

| Property | Requirement |
|----------|-------------|
| Grid | 1.0° latitude/longitude: `lat` 181 values from −90 to 90, `lon` 360 values from 0 to 359 |
| Pressure levels | `level`: the 13 levels listed above |
| Time step | 6-hourly |
| Time coverage | `init_date` − 6 h through `init_date` + `ps_eval` × 6 h. For the example: 2017-01-01 06 UTC to 2017-01-12 12 UTC (46 times). |
| Dimensions | `batch` (size 1), `time`, `level`, `lat`, `lon` |
| Times | a `datetime(batch, time)` coordinate, in which `init_date` is looked up (the script re-indexes `time` itself) |
| Variables | the 11 variables listed above |
| Static fields | `geopotential_at_surface`, `land_sea_mask` (a time dimension, if present, is removed) |
| Forcings | not needed: GraphCast's code computes `toa_incident_solar_radiation` and the time-of-day and time-of-year features from `datetime`, as it did for the example run |
| Format | netCDF, readable by `xarray.load_dataset()`; the whole file is read into memory |

---

## Design Notes

### Method

* **LoRA.** Each targeted weight matrix W of the processor (all 64 weight matrices of the 16 message-passing layers) is replaced by W + ΔW, with ΔW = (α/√r)·B·A (rank-stabilized LoRA). B starts at zero and the entries of A are drawn from N(0, (0.01/√r)²) with a fixed seed, so ΔW = 0 at the start and the first forecast is GraphCast's own. Only A and B are trained; the GraphCast weights stay fixed.
* **IC.** The perturbation is optimized in z-scored space (each input variable standardized by the mean and standard deviation of the initial condition itself over all grid points, separately for each pressure level and input time) and converted back before the forward pass. Both input times (`init_date` − 6 h and `init_date`) are perturbed; the static fields and forcings are not.
* **Loss.** GraphCast's training loss: the mean over forecast steps and grid points of the squared error, normalized by the standard deviation of 6-hour differences and weighted by grid-cell area, pressure level and variable. The per-variable weights are DeepMind's, except that precipitation is weighted 0 (DeepMind: 0.1).
* **Penalties** (joint modes). The size penalty (`rho`, `Lambda`) uses S_IC, the cos(latitude)-weighted sum of the squared z-scored IC perturbation, and S_W, the sum of the squared entries of all ΔW. The smoothness penalty (`ic_laplacian_weight`) is the cos(latitude)-weighted sum of the squared five-point Laplacian of the z-scored IC perturbation, computed on the grid. Both are off or have no effect in the example.
* **Precision.** `use_f32: true` runs the model in float32; the default, `false`, uses GraphCast's bfloat16.

### Optimization Windows

Forecast error grows with lead time, so a loss over a long window is dominated by its last steps, and fitting a 10-day trajectory from the start is hard. The optimization therefore runs on successively longer windows (here 8, 16, 24, 32 and 40 steps, i.e. 2 to 10 days). Each window starts from the final state of the previous one with a fresh Adam optimizer (default β₁ = 0.9, β₂ = 0.999). In joint modes, the saved state is the one with the lowest objective (loss plus penalties) in the last window.

### Learning Rate

The learning rate in each window is the base rate (`lora_lr`, `ic_lr`) divided by max(T/4, 1)^`lr_window_exponent`, where T is the window in steps. The example uses the square root: 1e-3/√2 ≈ 7.1e-4 for the 2-day window down to 1e-3/√10 ≈ 3.2e-4 for the 10-day window. As the window grows, the loss becomes more sensitive to each update, and a smaller rate avoids overshooting when the window widens.

---

## Reproducibility

Runs are not bit-for-bit reproducible on GPUs, because some GPU operations (for example the scatter-adds in GraphCast's message passing) add numbers in a non-deterministic order. The optimization amplifies these last-digit differences. We checked this release against the original research code on the example, with one configuration on one NCAR Derecho node (A100 40 GB GPUs, float32):

| Run | Training-loss reduction vs. control (10-day window) |
|-----|------------------------------------------------------|
| Original code, run A | 71.3 % |
| Original code, run B | 69.5 % |
| This repository, fresh clone | 71.5 % |
| Original code, archived research run (June 2026) | 70.2 % |

All four runs agree to about 1 part in 10⁵ at the first step. The two runs of the original code end 6 % apart in loss; the release run is within that spread (0.8 % from run A). The individual ΔW matrices differ by about 20 % (median relative Frobenius norm) between any two runs, including the two runs of the same code: the optimization finds different, equally good corrections. XLA's `--xla_gpu_deterministic_ops=true` flag is meant to remove this non-determinism; it was not used here.

---

## Changes to GraphCast

`graphcast/` is DeepMind's code at commit `8debd72` with small changes:
* a per-timestep loss;
* a loss with optional selection of variables, pressure levels and a lat/lon region (with everything selected, as here, it computes the same weighted MSE as GraphCast's `weighted_mse_per_level`);
* the precipitation loss weight set to 0;
* weighted per-variable loss diagnostics;
* rematerialization of the processor layers to save memory (same math);
* a `justify` option for slicing inputs and targets from the dataset;
* compatibility fixes for newer xarray and Python 3.9.

Every changed file says so at the top; `NOTICE` lists the changes file by file.

`graphcast/graphcast.py` reads the loss selection from the config named by the `j_path` environment variable when it is imported. `submit_jobs/run_lora_only.sh` sets `j_path`; to use these modules from your own code, set it first.

---

## Citations

If you use this code, please cite it (`CITATION.cff`) and GraphCast:

```bibtex
@software{vonich_graphcast_ic_lora,
  author = {Vonich, P. Trent},
  title  = {graphcast-ic-lora: joint optimization of {GraphCast} initial conditions
            and low-rank weight corrections},
  url    = {https://github.com/tvonich/graphcast-ic-lora},
  year   = {2026}
}

@article{lam2023learning,
  title   = {Learning skillful medium-range global weather forecasting},
  author  = {Lam, Remi and Sanchez-Gonzalez, Alvaro and Willson, Matthew and
             Wirnsberger, Peter and Fortunato, Meire and Alet, Ferran and
             Ravuri, Suman and Ewalds, Timo and Eaton-Rosen, Zach and Hu, Weihua and
             Merose, Alexander and Hoyer, Stephan and Holland, George and
             Vinyals, Oriol and Stott, Jacklynn and Pritzel, Alexander and
             Mohamed, Shakir and Battaglia, Peter},
  journal = {Science},
  volume  = {382},
  number  = {6677},
  pages   = {1416--1421},
  year    = {2023},
  doi     = {10.1126/science.adi2336}
}
```

---

## License

Code: Apache License 2.0 (`LICENSE`, `NOTICE`), including the modified GraphCast code (© 2023 DeepMind Technologies Limited). The GraphCast weights are CC BY-NC-SA 4.0; weight corrections made with this code derive from them and carry the same license.

---

## For Agents

This section provides structured information for automated tools or agents working with this codebase.

### Pipeline Execution Order

```
submit_jobs/run_lora_only.sh [config.json]      (bash, or qsub from the repository root)
  → locates the repository; resolves the config path (argument, RUN_CONFIG, or the example)
  → activates CONDA_ENV (default graphcast-ic-lora; "" = keep the active environment)
  → stops unless JAX sees a GPU
  → exports j_path, GRAPHCAST_PARAMS_PATH, GRAPHCAST_STATS_PATH, PYTHONPATH,
            XLA_PYTHON_CLIENT_MEM_FRACTION
  → python -u batch_modules/optimize_ic_lora.py   (working directory: repository root)
      → reads the config named by j_path
      → imports config.py, load_model.py (weights, normalization statistics), jitted.py
        (GraphCast loss and forward functions), lora_utils.py, and graphcast/
      → loads dataset_file_path into memory and takes the data from init_date − 6 h on
      → computes the control loss, then runs Adam over the windows
      → saves outputs to {save_root}/{init_date}/
      → evaluates the saved corrections over ps_eval steps; logs "// Complete //"
```

### Environment Variables

| Variable | Set by | Consumed by | Description |
|----------|--------|-------------|-------------|
| `j_path` | run script | `optimize_ic_lora.py`, `jitted.py`, `graphcast/graphcast.py` (at import) | Path to the JSON config |
| `GRAPHCAST_PARAMS_PATH` | run script (default `params/params_GraphCast_small.npz`) | `load_model.py` | GraphCast_small weights |
| `GRAPHCAST_STATS_PATH` | run script (default `stats`) | `load_model.py` | Directory with the three `*_by_level.nc` files |
| `PYTHONPATH` | run script | Python | Repository root first, so that `import graphcast` finds the bundled copy |
| `XLA_PYTHON_CLIENT_MEM_FRACTION` | run script (default 0.90) | JAX | Fraction of GPU memory JAX reserves |
| `RUN_CONFIG`, `CONDA_ENV` | user (optional) | run script | Config path; conda environment (name or path, `""` = none) |
| `GRAPHCAST_DATA_PATH` | user (optional) | `config.py` | Input file used when the config has no `dataset_file_path` |

### Key Files

| File | Role |
|------|------|
| `batch_modules/optimize_ic_lora.py` | Main entry point: all optimization modes, saving, evaluation |
| `batch_modules/lora_utils.py` | LoRA factors: target matrices, initialization, applying ΔW, save/load |
| `batch_modules/jitted.py` | GraphCast forward, loss and per-timestep loss functions (Haiku + JAX) |
| `batch_modules/load_model.py` | Loads the weights and normalization statistics at import time |
| `batch_modules/config.py` | Small helpers and the default dataset path |
| `graphcast/` | Modified copy of DeepMind's GraphCast code (see `NOTICE`) |
| `configs/lora_only_2017-01-01T12.json` | Example configuration |
| `submit_jobs/run_lora_only.sh` | Run script (plain bash or PBS) |
| `environment.yml` | Conda environment (tested versions) |
| `params/`, `stats/` | Weights and statistics (download separately, see Setup) |

### Success Criteria

A successful run:
1. ends its log with `// Complete //` and exits with status 0;
2. writes `meta.json`, `lora_products.npz`, `lora_factors.npz`, `loss.npy` and `eval_44step.npz` to `{save_root}/{init_date}/`;
3. for the example, logs `Improvement over control:` of about 70 % (69.5–71.5 % in the release check).

### Common Failure Modes

| Symptom | Likely cause |
|---------|--------------|
| `JAX found no GPU` | No CUDA 12 toolkit visible to the pinned jaxlib (see `environment.yml`) or no GPU on the node |
| `FileNotFoundError` for `params/...` or `stats/...` | Weights or statistics not downloaded, or `GRAPHCAST_PARAMS_PATH` / `GRAPHCAST_STATS_PATH` wrong |
| `NameError: name 'start_index' is not defined` | `init_date` is not in the file's `datetime` coordinate |
| `TypeError: expected str, bytes or os.PathLike object, not NoneType` at import | `j_path` is not set |
| `Repository not found` | `qsub` was not run from the repository root |
| CUDA out-of-memory error | GPU smaller than 40 GB, or another process on the same GPU |
| Job killed while loading data | Input file too large for host memory; trim it to the needed times |
