---
name: aliby-pipeline
description: >
  Run the aliby image analysis pipeline with nahual cellpose/trackastra GPU
  servers. TRIGGER when: user asks to run extraction, segmentation, profiling,
  or any of the timelapse/cellpainting/viability/cycle pipelines, mentions
  nahual servers, or wants to start/stop/monitor pipeline runs. Also trigger
  on "run nb01", "run pipelines", "start cellpose servers", "start trackastra".
---

# Aliby Pipeline Runner

Run the aliby image analysis pipeline (`notebooks/nb01_extract_profiles.py`)
with nahual cellpose and trackastra GPU servers for segmentation and tracking.

## Architecture overview

Three layers:

1. **Model servers** -- standalone GPU processes that load a single model
   (Cellpose-SAM, Trackastra, DINOv2, BABY, ViT, SubCell, ...) and expose
   it behind an nng IPC socket. Each model is its own repo/flake under
   `afermg/<model>` (e.g. `github:afermg/cellpose`,
   `github:afermg/trackastra`, `github:afermg/dinov2`). Every server
   exposes `python server.py ipc:///tmp/<name>.ipc`.
2. **Nahual transport/client** (`github:afermg/nahual`) -- thin numpy-only
   transport layer built on pynng. Client side provides
   `nahual.process.dispatch_setup_process('<model>')` which returns
   `(setup, process)` functions that serialize dict/numpy payloads to the
   matching server. No model code lives here; it's pure plumbing.
3. **ALIBY orchestrator** (`github:afermg/aliby`) -- end-to-end pipelining
   framework that reads images (TIFF/Zarr), tiles, dispatches to nahual
   servers for segmentation/tracking, and extracts features via
   [cp_measure](https://github.com/afermg/cp_measure). Canonical entry
   points: `aliby.pipe_builder.build_pipeline_steps` +
   `aliby.pipe.run_pipeline_and_post`. Install via `pip install aliby`,
   `uv sync --all-groups`, or `nix develop .` from a checkout.

In this project, the aliby invocation is wrapped in
`notebooks/nb01_extract_profiles.py` (marimo notebook); its
`run_pipelines()` function is importable for scripted runs and handles
per-batch/per-assay iteration on top of aliby.

## Notebook layout

| Notebook | Purpose |
|----------|---------|
| nb01_extract_profiles | Segmentation + feature extraction (this skill) |
| nb02_segmentation_audit | Review segmentation quality |
| nb03_tracking_quality | Review trackastra tracks |
| nb04_aggregate_profiles | Aggregate single-cell → well/site |
| nb05_phenotypic_scoring | Cross-batch phenotypic scoring |
| nb06_umap_explorer | UMAP visualisation |
| nb07_normalization | Per-batch normalization |
| nb09_cross_batch_normalization | Cross-batch normalization |
| nb10_feature_importance | Feature importance analysis |
| nb11_assay_concordance | Inter-assay concordance |

## Critical: CUDA stub library issue

Nix environments include a stub `libcuda.so` from `cuda_cudart-*-stubs` that
shadows the real NVIDIA driver at `/run/opengl-driver/lib/libcuda.so`. This
causes `Error 34: CUDA driver is a stub library` and forces CPU fallback.

**Fix**: prepend `/run/opengl-driver/lib` to `LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH
```

Already fixed in this project's `flake.nix`. Pass it explicitly when
launching nahual servers via `nix run`.

## Starting model servers

Each model repo is a standalone Nix flake exposing `server.py`. Per the
nahual README, the canonical launch pattern is:

```bash
nix develop --command bash -c "python server.py ipc:///tmp/<name>.ipc"
```

Against a GitHub flake (no local checkout) the equivalent is `nix run`
with the flake's default app, or `nix develop github:afermg/<model>
--command bash -c "python server.py ipc:///tmp/<name>.ipc"`.

Known server repos (pass the appropriate `<name>` to `dispatch_setup_process`):

| Repo | Socket name | `dispatch_setup_process` key |
|------|-------------|------------------------------|
| `github:afermg/cellpose` | `cellpose{N}` | `"cellpose"` |
| `github:afermg/trackastra` | `trackastra` | `"trackastra"` |
| `github:afermg/baby` | `baby` | `"baby"` |
| `github:afermg/dinov2` | `dinov2` | `"dinov2"` |
| `github:afermg/dinov3` | `dinov3` | `"dinov3"` |
| `github:afermg/nahual_vit` | `vit` | `"vit"` |
| `github:afermg/SubCellPortable` | `subcell` | `"subcell"` |

### Cellpose servers (segmentation)

Uses Cellpose-SAM, ~4 GB VRAM per instance. On 2× RTX A6000 (49 GB each),
run up to 6 servers per GPU (12 total). Launch each in a `screen`:

```bash
# 6 servers on GPU 0
for i in $(seq 0 5); do
  screen -dmS cellpose$i bash -c \
    "export LD_LIBRARY_PATH=/run/opengl-driver/lib:\$LD_LIBRARY_PATH && \
     export CUDA_VISIBLE_DEVICES=0 && \
     nix run github:afermg/cellpose -- ipc:///tmp/cellpose$i.ipc"
done

# 6 servers on GPU 1
for i in $(seq 6 11); do
  screen -dmS cellpose$i bash -c \
    "export LD_LIBRARY_PATH=/run/opengl-driver/lib:\$LD_LIBRARY_PATH && \
     export CUDA_VISIBLE_DEVICES=1 && \
     nix run github:afermg/cellpose -- ipc:///tmp/cellpose$i.ipc"
done
```

If the flake's `apps.default` does not forward args, fall back to:
`nix develop github:afermg/cellpose --command python server.py ipc:///tmp/cellpose$i.ipc`.

### Trackastra server (tracking)

A single instance is enough (tracking is fast, timelapse-only):

```bash
screen -dmS trackastra bash -c \
  "export LD_LIBRARY_PATH=/run/opengl-driver/lib:\$LD_LIBRARY_PATH && \
   export CUDA_VISIBLE_DEVICES=1 && \
   nix run github:afermg/trackastra -- ipc:///tmp/trackastra.ipc"
```

### Pinning a session (long runs)

These jobs outlive an SSH session only if they run under `systemd-run
--user --scope` or inside `tmux`/`screen` started with
`systemd-run --user --scope -- screen -dmS ...`. A plain `screen -dmS`
attached to a login `systemd --user` scope gets SIGTERM'd when the user
session ends (see journalctl for `user@$UID.service` stop events) --
this silently kills every cellpose/trackastra/pipeline screen at once.

### Verifying servers

Wait ~60-90 seconds for build + model loading, then test:

```python
from nahual.process import dispatch_setup_process
import numpy as np

setup, process = dispatch_setup_process('cellpose')
info = setup({}, address='ipc:///tmp/cellpose0.ipc')
print('Setup:', info)  # expect device: cuda:0

img = np.random.randint(0, 255, (1, 128, 128), dtype=np.uint16)
result = process(img, address='ipc:///tmp/cellpose0.ipc')
print('Result shape:', result.shape)  # (128, 128)
```

If setup returns `{}`, the model failed to load -- check the screen session
for errors (usually CUDA OOM from too many servers on one GPU).

## Running the pipeline

### This project (nb01 wrapper around aliby)

```python
from pathlib import Path
from notebooks.nb01_extract_profiles import run_pipelines

batches_and_assays = [
    ('<batch_id>', '<assay>'),
    ...
]
input_dir = '<path to batches root>'
profiles_path = Path('<path to profiles output root>')
nahual_addresses = [f'ipc:///tmp/cellpose{i}.ipc' for i in range(12)]

run_pipelines(
    batches_and_assays=batches_and_assays,
    input_dir=input_dir,
    profiles_path=profiles_path,
    extract_ncores=192,
    ntps=None,
    overwrite=False,
    ncores=18,           # parallel site workers
    max_sites=None,
    nahual_addresses=nahual_addresses,
)
```

Key parameters:
- **ncores**: parallel site workers (joblib loky). The *only* parallelism
  knob you should tune. See "Parallelism tuning" below.
- **extract_ncores**: cores for cp_measure feature extraction per site.
  **Always pin to the box's logical CPU count** (e.g., 192). Per-site
  extraction parallelizes well inside one process; capping it below the
  core count under-utilizes CPU during the extract stage, and raising it
  above the core count buys nothing. Do not vary this between runs.
- **nahual_addresses**: list of IPC addresses; assigned round-robin to sites.
  **Must match the number of running servers.** Listing addresses that have
  no server behind them causes workers to block forever on `ConnectionRefused`
  retries -- the pipeline hangs with no traceback until OOM or manual kill.
- **overwrite**: `False` skips already-processed sites.

#### Parallelism tuning

Rule: `extract_ncores` is fixed at the logical CPU count; `ncores` is the
only dial. To pick `ncores`:

1. **Probe first**. Run a side-output calibration on the heaviest assay
   (5-channel cellpainting with both nuclei + cell segmentation) with
   `profiles_path=.../standard_output_probe`, `max_sites=10`, and a
   conservative `ncores` (e.g., 10). Capture peak per-worker RSS and
   sustained CPU% from a resource monitor or repeated `ps -o rss,pcpu`.
2. **Compute the budget**:
   `ncores = min(target_cpu_cores / per_worker_cpu_cores,
   mem_budget / per_worker_peak_RSS)`.
   Aim for ~80% sustained CPU and leave RAM headroom (≤75% of total) so
   joblib's gather phase doesn't OOM-kill a worker.
3. **Failure mode hint**: SIGKILL-shaped
   `joblib.externals.loky.process_executor.TerminatedWorkerError: ...
   exit codes {SIGKILL(-9)}` with leaked semlock objects on shutdown
   reads as memory-driven even when system mem looks fine. Drop `ncores`
   first; do not touch `extract_ncores`.

Rule of thumb on this 192-core / 768 GiB box: `ncores=18-20` is a safe
default for most assays. Heavy 5-channel cellpainting may need `ncores=12-15`.

Available `(batch, assay)` pairs come from
`notebooks/nb01_extract_profiles.get_batches_assays()` -- call it to discover
the valid list for the current dataset instead of hard-coding.

Timelapse assays include trackastra tracking as a global step (uses the
trackastra server); non-timelapse assays don't need it.

### Canonical aliby usage (outside this project)

Per the aliby README, a standalone run looks like:

```python
from pathlib import Path
from aliby.io.dataset import DatasetDir
from aliby.pipe import run_pipeline_and_post
from aliby.pipe_builder import build_pipeline_steps

regex = r".*__([A-Z][0-9]{2})__([0-9])__([A-Za-z]+).tif"
capture_order = "WFC"

dataset = DatasetDir(Path("data/my_experiment"), regex=regex,
                     capture_order=capture_order)
positions = dataset.get_position_ids()
key, path = positions[0]["key"], positions[0]["path"]

pipeline = build_pipeline_steps(
    channels_to_segment={"nuclei": 1},
    channels_to_extract=[0, 1],
    features_to_extract=["intensity", "sizeshape"],
)
pipeline["steps"]["tile"]["image_kwargs"] = {
    "source": {"key": key, "path": path},
    "regex": regex,
    "capture_order": capture_order,
}

result = run_pipeline_and_post(
    pipeline=pipeline, pipeline_name=key, output_path="results",
)
```

Capture groups map filename fields (`W`ell, `F`ield, `C`hannel, `T`ime)
into aliby's internal TCZYX arrays. See `examples/` in the aliby repo
for nahual-backed segmentation and embedding variants.

### Direct nahual client (debugging a single server)

```python
import numpy as np
from nahual.process import dispatch_setup_process

setup, process = dispatch_setup_process("cellpose")
address = "ipc:///tmp/cellpose0.ipc"

setup({}, address=address)               # loads model server-side
img = np.random.randint(0, 255, (1, 128, 128), dtype=np.uint16)
masks = process(img, address=address)    # (128, 128)
```

Same pattern for other models -- swap `"cellpose"` for `"trackastra"`,
`"dinov2"`, `"vit"`, etc., and pass the model's setup parameters dict
(e.g. DINOv2 wants `{"repo_or_dir": "facebookresearch/dinov2",
"model": "dinov2_vits14_lc"}`).

### Without nahual (local cellpose)

Pass `nahual_addresses=None` to use local CellposeSAM. The model is loaded
in-process once per worker, so reduce `ncores` to avoid GPU OOM. Requires
`LD_LIBRARY_PATH=/run/opengl-driver/lib:$LD_LIBRARY_PATH` in the environment.

## Resource guidelines (192-core, 754 GB RAM, 2× RTX A6000)

| Config | CPU | RAM | Notes |
|--------|-----|-----|-------|
| ncores=18, 12 cellpose servers | ~35-60% | ~270-400 GB | Recommended |
| ncores=24, no nahual (local GPU) | ~50-90% | 400+ GB | OOM risk |
| ncores=12, no nahual (local GPU) | ~35-55% | ~270 GB | Conservative |

**RAM warning threshold**: >500 GB used (~66%). With no swap, the OOM killer
will terminate workers silently (no traceback -- just `resource_tracker`
leaked-object warnings at the end of the log).

## Monitoring

```bash
# Resource usage
top -bn1 | head -5 && free -h && \
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv

# Pipeline progress
grep -c "Saving"    <log>   # completed steps
grep -c "Skipping"  <log>   # already-done sites
grep -c "Exception" <log>   # failures

# OOM-killer signature (no traceback, just at end):
#   resource_tracker: There appear to be N leaked semlock/folder objects
```

## Stopping servers

```bash
# Kill all screen sessions for nahual servers
screen -ls | grep -E "cellpose|trackastra" | awk -F. '{print $1}' | \
  xargs -I{} screen -X -S {} quit

# Kill any orphaned server processes
pkill -f "python server.py ipc://"

# Clear leaked GPU memory (verify first with nvidia-smi)
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs kill
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `unpack requires a buffer of 2 bytes` | Server returned `{}` instead of numpy | Server crashed processing -- check screen for CUDA OOM |
| `ConnectionRefused` on IPC sockets | Server not up yet | Wait 60-90s after `nix run` for build + model load |
| `CUDA driver is a stub library` | Wrong `libcuda.so` found first | Prepend `/run/opengl-driver/lib` to `LD_LIBRARY_PATH` |
| Pipeline hangs silently | All workers blocked on dead server | Kill pipeline, restart servers, rerun with `overwrite=False` |
| Pipeline hangs silently, some servers idle | `nahual_addresses` lists more sockets than running servers; workers stuck retrying `ConnectionRefused` on the missing ones | Trim `nahual_addresses` to match the running server count (e.g. `range(12)` not `range(24)`) |
| Silent death, leaked-semlock warnings | OOM killer | Reduce `ncores` or number of cellpose servers |
| `ModuleNotFoundError: analysis` | Old path; notebooks moved to `notebooks/` | Import from `notebooks.nb01_extract_profiles` |
| `Could not pickle the task to send it to the workers` / `Cannot pickle files that are not opened for reading: a` | `configure_logging()` called in the parent of `joblib.Parallel(backend="loky")`; loguru's file handler is in module globals and cloudpickle traps the open `mode="a"` fd | Move `configure_logging(out_dir / "log.txt")` *inside* the worker function. Workers append to the same file; loguru serializes |
| `keys must be str, int, float, bool or None, not tuple` when dumping `pipelines.json` | aliby pipeline dicts have tuple-keyed nested mappings (`(0, 1)` in `passed_data` / `passed_methods` / multi-channel `tree`) | Coerce dict keys in your sanitizer, not just values |
| Parquet has `pearson/Correlation_*` etc. when you asked for "intensity only" | `build_pipeline_steps` silently appends an `extractmulti_<obj>` step whenever `len(channels_to_extract) > 1` and you're not on baby. `features_to_extract=("intensity",)` only governs the per-channel branch, not the multi-channel one | After building, walk `base["steps"]` and drop entries starting with `extractmulti_`, plus matching keys in `passed_data` / `passed_methods` / `save`. Note: prefix is `extractmulti_`, no underscore between `extract` and `multi` |
| "Random" CUDA failures on specific positions but not others | One of the two physical GPUs is hardware-faulty; nahual_addresses round-robin assigns positions to alternating GPUs, so failures cluster on every-other socket index | Tally CUDA errors per `cellpose{N}.log` and check if they cluster on even-vs-odd N. If so, cross-check with `journalctl -k \| grep -E "Xid\|NVRM"` — Xid 13 / 31 / 43 on a single PCI address confirms the bad card. Pin extraction to the healthy pool: `--nahual-addresses ipc:///tmp/cellpose{1,3,5,7,9}.ipc` (or whichever parity is clean) |
