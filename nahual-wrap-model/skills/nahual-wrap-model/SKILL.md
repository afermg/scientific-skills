---
name: nahual-wrap-model
description: >
  Wrap a third-party ML model (typically vision/microscopy or single-cell) as a
  Nahual server: fork the upstream repo to afermg/<repo>, add server.py +
  flake.nix + basic_test.py + .envrc + examples/<model>.py, validate it runs
  on GPU, push to a `nahual-wrap` branch on the fork. TRIGGER when: user asks
  to wrap, port, integrate, or "add a model to nahual"; mentions writing a
  server.py or flake.nix for an ML repo; says things like "make X work as a
  nahual server", "deploy X via nahual", "add X to nahual_models"; or asks how
  the existing wraps (dinov3, scdino, deepprofiler, channelsformer,
  cellwhisperer, dinov2, vit, subcell, cellpose, baby, trackastra, stardist,
  embedseg, instanseg, megaseg, microsam, cellsam, ultrack, nahual_bioimageio)
  are structured.
---

# Wrap a model as a Nahual server

Take an upstream ML repo and turn it into a self-contained Nix flake that
exposes the model behind an `nng` IPC socket via the standard Nahual
`setup`/`process` interface. The end state is something the user can run with:

```bash
nix run github:afermg/<model> -- ipc:///tmp/<model>.ipc
```

and call from anywhere with `nahual.process.dispatch_setup_process("<model>")`.

## When to use this

Use it whenever you're adding a new model to the family in
`/home/amunoz/projects/nahual_models/<model>/`. The pattern is the same every
time; the variation is in (a) which `nixpkgs` channel works for the model's
dependency stack and (b) how to wire the upstream model API into Nahual's
NCZYX numpy contract.

If you're being asked to *run* an existing wrap (e.g. start cellpose servers
for the aliby pipeline), use the `aliby-pipeline` skill instead — that one
covers operations, this one covers authoring.

## High-level workflow

1. **Fork upstream → `github:afermg/<repo>`.** Don't push to the original.
   Default branch on every afermg fork is `main`. Use `gh repo fork <owner>/<repo> --clone=false` with `GH_TOKEN=$(cat ~/.github_token)`.
2. **Locally clone** via SSH (HTTPS hangs) into `/home/amunoz/projects/nahual_models/<repo>/`. Create a `nahual-wrap` branch off the upstream default — never push to `main` of the fork unless the upstream itself uses `main`.
3. **Add five required files at the repo root** (templates in `references/`):
   - `server.py` — Nahual setup/process server
   - `flake.nix` — `apps.default` (`nix run`) + `devShells.default` (`nix develop`)
   - `basic_test.py` — standalone smoke test (no IPC, just model + forward), with the importlib source-shadow guard
   - `.envrc` — single line: `use flake .` (no `--impure`; `cudaSupport` is set inside the flake, not via env)
   - `examples/<model>.py` — client demo, in-repo (NOT in the central nahual repo)

   **Do NOT add `nix/nahual.nix`.** Nahual is pulled directly from its remote flake as a flake input — see "Wiring nahual via flake input" below. This was the standard as of 2026-05; older wraps that still ship `nix/nahual.nix` are being migrated.
4. **Pick a `nixpkgs` channel.** Default to `nixos-unstable`. Fall back to
   `nixos-24.11` only if the model's deps (typically `timm`, `tensorflow`,
   ancient `transformers`) are too painful on unstable. **Pin a specific commit** when a moving channel triggers a multi-hour torch rebuild — see ultrack's `nixpkgs.url = "github:NixOS/nixpkgs/<sha>";` for the pattern.
5. **Make GPU work.** `cudaSupport = true` is non-negotiable — even if the TF
   wheel takes 6 hours to build the first time. CPU-only wraps are not
   acceptable. **Conda-only deps must be packaged as proper Nix derivations** — no `micromamba` bootstrap fallback, no matter how heavy the C++ build is (the user has explicitly OK'd >1h cold-cache builds for vigra/nifty/affogato/torch_em).
6. **Validate twice:**
   - `nix develop --impure --command python basic_test.py` (no IPC)
   - `nix run --impure .#default -- ipc:///tmp/<name>.ipc` + a client roundtrip via the in-repo `examples/<model>.py`
   Both must show `device: cuda:0` (torch) or `device: GPU:0` (TF).
7. **Push to `git@github.com:afermg/<repo>.git`** (HTTPS hangs — use SSH).
   Push to a **`nahual-wrap` branch** (or `nahual-wrap-onnx`, `nahual-wrap-<variant>` for alternatives), not to the upstream default branch. Touch only the new files; never overwrite or delete anything from upstream (especially `.github/workflows/`).
8. **Update the `model-links` branch** of the central `afermg/nahual` repo (read-only catalogue) with the new fork URL — but **do NOT commit anything else to `afermg/nahual`**. Per-model client examples live in their own wrap repos at `examples/<model>.py`, not in `afermg/nahual/examples/`.

## File anatomy

### `server.py`

Three contracts to satisfy:

- `address = sys.argv[1]` at module top — the IPC address. (`basic_test.py`
  has to inject a placeholder when importing this file; see template.)
- `setup(...) -> tuple[Callable, dict]` — load weights, return
  `(processor_partial, info_dict)`. The dict is the JSON that the client gets
  back from its `setup()` call; it MUST include a `device` key.
- `process(pixels, ...) -> numpy.ndarray | torch.Tensor` — the per-call
  inference fn. Image models receive 5-D `NCZYX`; squeeze Z and pad/crop the
  channel dim with `nahual.preprocess.pad_channel_dim` /
  `validate_input_shape`. Non-image models (e.g. CellWhisperer) take 2-D
  `(N_cells, N_features)`.

The wire-up at the bottom is fixed:

```python
async def main():
    with pynng.Rep0(listen=address, recv_timeout=300) as sock:
        async with trio.open_nursery() as nursery:
            nursery.start_soon(partial(responder, setup=setup), sock)

if __name__ == "__main__":
    try:
        trio.run(main)
    except KeyboardInterrupt:
        pass
```

Full template: `references/server.py.template`.

### `flake.nix`

Two outputs that matter:

- `apps.default` — used by `nix run`. Builds a minimal Python env (just what
  `server.py` needs) and a `runserver.sh` wrapper that defaults the IPC
  address.
- `devShells.default` — used by `nix develop` and `direnv`. Should include
  the *runtime* deps plus typical dev extras (`tifffile`, `scikit-image`,
  `pyyaml`).

Both should pin nixpkgs the same way:

```nix
pkgs = import nixpkgs {
  system = system;
  config = {
    allowUnfree = true;
    cudaSupport = true;   # MUST be true — see GPU section
  };
};
```

Use `pynng-flake.url = "github:afermg/pynng"` with
`pynng-flake.inputs.nixpkgs.follows = "nixpkgs"` so we don't pull a second
nixpkgs into the closure. Full template: `references/flake.nix.template`.

### Wiring nahual via flake input (no `nix/nahual.nix`)

Nahual itself is pulled directly from its remote flake — do NOT vendor a
local `nix/nahual.nix`. Add to `inputs`:

```nix
nahual-flake.url = "github:afermg/nahual";
nahual-flake.inputs.nixpkgs.follows = "nixpkgs";
```

The `inputs.nixpkgs.follows = "nixpkgs"` is **load-bearing**. Without it,
nahual's flake builds against its own pinned nixpkgs (e.g. python 3.13.9),
and when your wrap uses a different interpreter (e.g. python 3.13.12 from
the same channel but a slightly different ABI), `python.withPackages`
*silently drops nahual from the env* — you'll discover it as a missing
import at runtime, not at build time. Always `follows = "nixpkgs"`.

Then reference the package. Two forms depending on interpreter alignment:

**Form 1 — same python interpreter as nahual builds for** (most common):
```nix
python.withPackages (pp: [ inputs.nahual-flake.packages.${system}.nahual pp.torch ... ])
```

**Form 2 — different python interpreter** (stardist on python3.11 for TF
2.13; deepprofiler on 3.11; ultrack/micro-sam/nahual_bioimageio with
`packageOverrides` that fork python's pkgset). Pull nahual's recipe and
re-`callPackage` against your interpreter's pkgset:

```nix
nahual = pythonForServer.pkgs.callPackage
  (inputs.nahual-flake + "/nix/nahual.nix") { pynng = packages.pynng; };
```

Either way: bumping nahual is a single `nix flake update nahual-flake` —
no per-wrap edit needed.

### `basic_test.py`

Smoke test that proves the model loads + forwards inside the dev shell,
without going through IPC. **Always use the importlib source-shadow guard**
even if the model package doesn't currently have C extensions — it's free
insurance against a future PyPI version that adds them, and against the
in-tree `<pkg>/` source dir shadowing the nix-built install:

```python
import importlib.util, os, sys

# Drop the repo root from sys.path BEFORE importing the model — otherwise
# an in-tree `<pkg>/` source dir can shadow the nix-built package and any
# compiled C-extension submodule (`<pkg>.lib.<ext>`) is reported missing.
# Stardist's `stardist.lib.stardist2d` is the canonical example.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p) not in {_HERE, ""}]

if len(sys.argv) < 2:
    sys.argv.append("ipc:///tmp/<model>_basic_test.ipc")

import numpy  # noqa: E402

# Load `server.py` via importlib so we never put _HERE back on sys.path.
_spec = importlib.util.spec_from_file_location("server", os.path.join(_HERE, "server.py"))
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)
setup = server.setup
```

Run with `nix develop --impure --command python basic_test.py`. Template:
`references/basic_test.py.template`.

### `.envrc`

Literally just:

```
use flake .
```

**No `--impure`.** `cudaSupport = true` and `allowUnfree = true` are set
inside `flake.nix`'s `import nixpkgs { config = { ... }; }`, so direnv
doesn't need impure mode. Wraps that historically had `use flake . --impure`
(trackastra, vit) are being normalized.

If the upstream `.gitignore` excludes `.envrc` (channelsformer did), force-add
it: `git add -f .envrc`.

### `examples/<model>.py`

The Nahual client demo lives **inside the wrap repo** at `examples/<model>.py`,
not in the central `afermg/nahual/examples/`. This makes wraps self-documenting
and keeps the central repo small. Mirror an existing wrap's example
(`dinov2/examples/dinov2.py` or `trackastra/examples/trackastra.py`) for the
imports + `dispatch_setup_process(...)` call. For models whose name isn't in
nahual's built-in registry (most new wraps), pass
`signature=("dict", "numpy")` explicitly.

**Multi-model wraps (vit, nahual_bioimageio):** when one repo serves several
models (e.g. ViT serves OpenPhenom and MorphEM, bioimageio serves any RDF),
ship one example *per model identifier* — `examples/openphenom.py`,
`examples/morphem.py`, etc. — not a single combined file. The flake's
`runserver.sh` typically points at a single dispatching `server.py` (or, for
ViT, separate per-model entry points under `src/vit/`); the examples cover the
client-side variations.

## Picking the nixpkgs channel

Default to `nixos-unstable`. If the dev-shell solve is taking >5 minutes or
recompiles `timm`/`transformers`/`tensorflow` from source for no good reason,
revert to `nixos-24.11` and move on — the user has explicitly OK'd this:

> "Some models may be too old for the newer nix versions, revert them if it
> is too much trouble"

Known fallbacks already on 24.11: `channelsformer` (timm 1.0.26 on unstable
hung pytest 35+ min; 1.0.22 on 24.11 is cached), `deepprofiler` (TF 2.13
isn't packaged on unstable), `cellpose` (legacy nahual branch),
`stardist` (TF backend, same constraints as deepprofiler).

**Pin a specific nixpkgs commit** (rather than a moving channel) when you
discover that a more-recent commit triggers a multi-hour `torch` rebuild.
Ultrack does this: `nixpkgs.url = "github:NixOS/nixpkgs/<sha>";` pinned to
the same commit cellpose/trackastra use, so the cached CUDA torch is
hit. Document the rationale in a comment above the URL.

**Override numba's pytestCheckPhase** when a custom Python override or
`cudaSupport = true` forces numba to rebuild — its test suite is 30+
minutes serial. The same trap exists for `numbagg`, `xarray`, and
`imagecodecs` once they're in the closure (each pytest takes 30 min to 2 h
on contended hosts). Pattern from ultrack / nahual_bioimageio:
```nix
python = pkgs.python313.override {
  packageOverrides = pyfinal: pyprev: {
    numba = pyprev.numba.overridePythonAttrs (_: {
      doCheck = false; doInstallCheck = false;
      pytestCheckPhase = "true"; installCheckPhase = "true";
    });
    # Apply the same to numbagg / xarray / imagecodecs / etc. as needed.
  };
};
```

When you stay on unstable, double-check: cudnn/CUDA versions match torch's
wheel, and `pp.timm` isn't in the dev shell unless it's actually imported.
Adding `timm` "just in case" cost an hour on scdino because it triggered the
full test suite rebuild on unstable.

## GPU support — non-negotiable details

The user has been explicit: "GPU MUST work, even if it takes a while to
build the tensorflow wheel necessary." Don't ship CPU-only.

### PyTorch models

Set `cudaSupport = true` in `nixpkgs` config. Add
`pkgs.cudaPackages.cudatoolkit` (and `pkgs.cudaPackages.cudnn` for some
models) to the dev shell's `packages` list. In `setup()`:

```python
if torch.cuda.is_available():
    torch_device = torch.device(int(device or 0))
else:
    torch_device = torch.device("cpu")
```

Smoke test should print `cuda:0`, not `cpu`.

### TensorFlow models (DeepProfiler is the only one so far)

`cudaSupport = true` selects nixpkgs' `python311-tensorflow-gpu-2.13.0` — a
*separate* derivation from the CPU `tensorflow-2.13.0`. Cold cache: pulls in
cuda-merged-11.8, cudnn-merged, UCC, NCCL, NVSHMEM, and compiles a chunk
(~6h on a fresh machine). This is expected. Don't try to dodge it.

The runtime-deps check on `tf-keras` will fail because tf-keras' wheel
METADATA lists `tensorflow` as a runtime dep but the store now has
`tensorflow-gpu` (different package name, same `tensorflow` import path).
Wrap it:

```nix
(pp.tf-keras.overridePythonAttrs (_: { dontCheckRuntimeDeps = true; }))
```

Set `TF_USE_LEGACY_KERAS=1` in the `runserver.sh` wrapper since TF 2.13
expects the standalone `keras` API and tf-keras 2.17 satisfies it under that
flag. Smoke test should report `device: GPU:0`.

## Common failure modes

| Symptom | Root cause | Fix |
|---|---|---|
| `Address in use` or client hangs at first connect | Stale pynng IPC socket from a prior run | `pkill -9 -f "server.py.*<name>"` then `rm -f /tmp/<name>.ipc` |
| `nix develop` recompiles timm test suite for 30+ min | `pp.timm` is in the dev shell on unstable | Drop `pp.timm` if it's only referenced by docstrings; or revert that flake to nixos-24.11 |
| TF build dies on `tf-keras` runtime check: "tensorflow not installed" | `cudaSupport=true` ships `tensorflow-gpu`, not `tensorflow` | `(pp.tf-keras.overridePythonAttrs (_: { dontCheckRuntimeDeps = true; }))` |
| Server boot stalls forever inside `torch.hub.load(..., source="local")` | hub spawns a subprocess for the local repo; flaky under nix | Import the factory directly: `from <model>.hub.backbones import <factory_name>` |
| `git push` to fork hangs at "Writing objects: 100%" | HTTPS remote | `git remote set-url origin git@github.com:afermg/<repo>.git` |
| `setup()` returns `{}` on the client | Server crashed processing the request | Read the server's stdout — usually a CUDA OOM or missing weight file |
| Transient `CUDA error: an illegal memory access was encountered` | GPU contention from another process; not a bug in the wrap | Re-check that `device: cuda:0` is in setup info; rerun |
| `.envrc` not picked up by direnv after git add | Upstream `.gitignore` ignores it (channelsformer) | `git add -f .envrc` |
| `ModuleNotFoundError: <pkg>.lib.<ext>` when running basic_test from the repo root | The in-tree source dir shadows the nix-built C-extension package (stardist, anything with a compiled `.so`) | Load `server.py` via `importlib.util.spec_from_file_location` instead of `from server import setup`, and remove the repo dir from `sys.path` first. See `references/basic_test-import-shadow.py.template`. |
| `ValueError: <something>_TOKEN not found` (cellSAM auth, hub-gated weights) | Upstream weight download is auth-gated (deepcell, HuggingFace private model) | Document the env var the user must set; don't try to bypass auth. Confirm server boots correctly otherwise. |
| `[Errno 30] Read-only file system` when downloading weights at `setup()` time | Upstream tries to write into the package dir (e.g. instanseg writes to `<pkg>/bioimageio_models/`), but the package lives in `/nix/store` | Set the upstream's cache-path env var (`INSTANSEG_BIOIMAGEIO_PATH`, `MICROSAM_CACHEDIR`, `XDG_CACHE_HOME`, etc.) in `runserver.sh` to a writable `~/.cache/<model>/` dir; `mkdir -p` it first. |
| `napari.utils...` ImportError pulled in via the model's package | Upstream's `__init__.py` eagerly imports napari for GUI hooks | Don't import the top-level package; import the specific inference submodule (`micro_sam.automatic_segmentation`, `cyto_dl.models.im2im` etc.). Set `try: from napari... except ImportError: ...` is upstream's pattern — match it. |
| `setuptools_scm` / `hatch-vcs` complains about no version | Source tarball in nix sandbox has no git tag visible | Set `SETUPTOOLS_SCM_PRETEND_VERSION=<x.y.z>` and `HATCH_BUILD_HOOKS_ENABLE=false` in the package's `nativeBuildInputs` env. |
| `pyproject.toml` `[tool.hatch.build.targets.wheel] sources = [...]` flattens namespace | Upstream's wheel layout strips the package prefix (EmbedSeg pattern) | `postPatch = "substituteInPlace pyproject.toml --replace-fail 'sources = [\"<Pkg>\"]' ''"` |
| Conda-only deps (`nifty`, `vigra`, `torch_em`, `python-elf`, `affogato`) | No PyPI distributions; nixpkgs Python packages don't exist | Package them as proper Nix derivations from upstream sources. **Do NOT fall back to micromamba** — the user has explicitly rejected conda bootstraps even for multi-hour cold-cache C++ builds. See micro-sam (`nix/{vigra,nifty,affogato,torch_em,python_elf}.nix`) for the concrete recipes. |
| ILP / Gurobi-style solver in upstream means "GPU mandatory" doesn't fit | Some tools (ultrack) have a CPU-bound combinatorial core | Run the *learning* parts on GPU and document honestly which steps are CPU. Don't ship CPU-only torch when the underlying detection nets *can* use GPU. |
| `cmake_minimum_required(VERSION <3.5)` rejected by cmake 4 | nixpkgs unstable's cmake bumped to 4; many older C++ libs (xtensor 0.25, xtl 0.7.7, affogato 0.3.x, nifty 1.2.x) target older cmake versions | Add `cmakeFlags = [ "-DCMAKE_POLICY_VERSION_MINIMUM=3.5" ];` to the derivation. |
| `xtensor 0.27` removes `svector(begin, end)` constructor that affogato/nifty rely on, AND introduces C++20 `concept` syntax | Upstream `xtensor` flag day | Pin `xtensor 0.25.0` + `xtl 0.7.7` + `xtensor-python 0.27.0` (the combo nixos-24.11 ships) via a custom `nix/xtensor_pinned.nix`; affogato pinned to `0.3.3` and nifty to `1.2.3` (later versions migrated to `xtensor/containers/...` includes that need 0.27). |
| `vigranumpy` Python bindings missing despite `pkgs.vigra` being installed | nixpkgs' `vigra` derivation builds boost without python support, silently disabling vigranumpy | Override boost-with-python and `toPythonModule`-wrap vigra. ~30-line `nix/vigra.nix`. |
| affogato 0.3.3 CMake uses removed `distutils.sysconfig.get_python_lib()` | Custom `FindNUMPY.cmake` + `FindPythonPyEnv.cmake` pre-date Python 3.12 | `postPatch` to replace both with modern `find_package(Python COMPONENTS NumPy Interpreter Development)` + `find_package(pybind11 CONFIG REQUIRED)`. Drop the `learning` subdir if it hits xtensor `operator/=` ambiguity (elf/micro_sam never call into it). |
| csbdeep raises "Starting with TF 2.6.0, a recent stand-alone keras package is required" | csbdeep does a raw `from keras import __version__` AND uses `_KERAS = 'tensorflow.keras'`, neither of which honor `TF_USE_LEGACY_KERAS` | `postPatch` in `nix/csbdeep.nix` to substitute `from keras` → `from tf_keras` AND `_KERAS = 'keras' if (IS_TF_1 or IS_KERAS_3_PLUS) else 'tensorflow.keras'` → `_KERAS = 'tf_keras'`. See afermg/stardist `nix/csbdeep.nix`. |
| Auth-gated weights (CellSAM `DEEPCELL_ACCESS_TOKEN`) | Upstream weight download requires a deepcell account | Document the env var; AND check HuggingFace for a community ONNX export (`keejkrej/cellsam-onnx`) that bypasses the auth — wrap it on a separate `nahual-wrap-onnx` branch using `onnxruntime` to orchestrate `image_encoder + cellfinder + mask_decoder + image_pe`. The original auth path stays on `nahual-wrap`. |
| One server can't load multiple models in sequence | Pre-0.0.9 `nahual.server.responder` only ran setup on the first dict message; subsequent dicts crashed `deserialize_numpy` | Bump the wrap's `nahual-flake` input to >= 0.0.9 (commit `dd58a250` or later). Dispatch is now by payload type — dict ⇒ setup, numpy ⇒ process. |

## Validation workflow

Run these in order. Don't claim "done" until both pass with GPU device.

```bash
cd /home/amunoz/projects/nahual_models/<model>

# 1. Cache the dev shell (forces nixpkgs solve + builds)
nix develop --impure --command true

# 2. Standalone smoke (no IPC)
nix develop --impure --command python basic_test.py
# expect: prints output shape, no errors, GPU device

# 3. End-to-end via IPC
rm -f /tmp/<name>.ipc                                    # paranoia
nix run --impure .#default -- ipc:///tmp/<name>.ipc &
SERVER_PID=$!
sleep 5
python /home/amunoz/projects/nahual/examples/<model>.py  # client roundtrip
kill $SERVER_PID
rm -f /tmp/<name>.ipc
```

The client example should print a real `(N, embed_dim)` (or NCYX masks)
shape, not `()`.

## Push + cataloguing

After local validation:

1. **Verify you didn't touch CI.** Run `git diff --stat` against the upstream
   default branch — your changes should be limited to: `server.py`,
   `flake.nix`, `flake.lock`, `nix/`, `basic_test.py`, `.envrc`,
   `examples/<model>.py`, and at most a tiny README note. If
   `.github/workflows/` shows up in the diff, revert it. (Cellpose, dinov2,
   dinov3, trackastra all have upstream workflows that must stay intact.)

2. **Push to a `nahual-wrap` branch** on the fork (SSH only):
   ```bash
   git remote set-url origin git@github.com:afermg/<repo>.git
   git checkout -b nahual-wrap   # or nahual-wrap-onnx for variants
   git push -u origin nahual-wrap
   ```
   Don't push to the upstream's default branch — keep the wrap on its own
   branch so users still see the upstream's `main` / `master` when they
   land on the repo.

3. **The client example is already in the wrap repo** at `examples/<model>.py`
   (per "File anatomy" above). No copy in the central nahual repo.

4. **Update the `model-links` branch** on `github:afermg/nahual` (the only
   place changes to nahual itself are allowed):
   ```bash
   cd /home/amunoz/projects/nahual
   git fetch origin model-links
   git switch model-links
   # add a row to README.md under the appropriate section
   git add README.md && git commit -m "docs: add <model> wrap link"
   git push origin model-links
   git switch -
   ```

   **Do NOT push to nahual's `master` / `main`** — the user has explicitly
   forbidden direct master pushes. Anything beyond the model-links README
   row goes on a separate branch (e.g. `clean-trailers`, `dedupe-examples`)
   for the user to fast-forward themselves.

   If a model was *considered but rejected*, document the why in the
   "Considered but not wrapped" section (e.g. cytoself produces a spatial
   VQ-VAE output that doesn't fit the embedding contract; CellDino is
   instance-seg with no released weights and needs MSDeformAttn).

## Output shape conventions

This caused enough back-and-forth that it's worth restating:

- Image embedding models should return **2-D** `(N, embed_dim)`. If the
  upstream returns a dict (DINO-style: `{"x_norm_clstoken": ..., ...}`),
  pluck the cls token in `process()` rather than asking the client to do
  it.
- Image segmentation models return masks; respect the upstream shape
  contract (cellpose: `(H, W)` instance label map).
- Single-cell encoders return `(N_cells, embed_dim)`.
- Trackastra returns tracks; non-image, non-numeric — handled separately.

If a model fundamentally returns spatial features (e.g. cytoself's VQ-VAE
codebook map), it doesn't fit the Nahual contract and shouldn't be wrapped
— note it in `model-links` instead.

## When upstream isn't easily nixified

Some packages aren't on PyPI or have C++ deps (vigra, nifty, gurobi) that
nobody has packaged for nixpkgs. Two escape hatches:

1. **Write your own `nix/<pkg>.nix`** that fetches the upstream source
   (PyPI for pure-Python, GitHub tag for C++ libraries) and declares deps
   manually — the EmbedSeg, InstanSeg, Ultrack, CellSAM, and micro-sam
   wraps all do this. Set `dontCheckRuntimeDeps = true` to skip wheel
   metadata mismatches and `pythonRuntimeDepsCheck = false` for older
   nixpkgs. List only the deps that the *server* import path actually pulls
   in (not the entire pyproject `dependencies` list — many are unused at
   inference time).

   For **C++ Python bindings** (vigra-numpy, nifty, affogato), expect to
   write 50-150 line derivations that handle: a `boost-with-python`
   override (vigra), `cmakeFlags = [ "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
   "-DBUILD_TESTS=OFF" ]`, a pinned older xtensor stack
   (`nix/xtensor_pinned.nix`), `postPatch` to substitute the
   distutils-using `Find*.cmake` modules with modern `find_package(Python
   COMPONENTS NumPy ...)`, and a substitute on
   `install(... DESTINATION ${Python_SITELIB})` to redirect to
   `${out}/${python.sitePackages}`. micro-sam's `nix/{vigra,nifty,
   affogato,torch_em,python_elf,xtensor_pinned}.nix` are the canonical
   example — copy them as a starting point.

   **The user has explicitly forbidden the micromamba/conda fallback**
   even when proper nixification takes >1h cold cache. If you find
   yourself reaching for `micromamba`, stop and write the Nix derivation
   instead.

2. **Generic loader wraps** for model-zoo packages (`bioimageio.core`).
   One server, accepts a model identifier (`"affable-shark"`, a DOI, a
   zenodo URL, a local rdf.yaml) at `setup()` time. Default to ONNX or
   TorchScript weight formats (self-contained — no `architecture.source`
   import is needed). Tier-2 flake outputs (`apps.with-stardist`,
   `apps.with-careamics`, `apps.with-monai`) for the popular
   custom-architecture families share one `server.py`. See the
   `nahual_bioimageio` wrap. Each tier-2 variant must build cleanly (no
   `pp.<missing-pkg>` references); package the Python deps yourself if
   nixpkgs doesn't ship them.

## Multi-model on one server (nahual >= 0.0.9)

Since nahual 0.0.9 the responder dispatches by payload type — a dict-shaped
incoming message re-runs `setup()` (replacing the processor in place);
numpy-shaped messages go to `process()`. This means a single long-lived
server can host many models in sequence: just call setup again with a new
identifier dict.

The wire format is unchanged, so older clients keep working. To take
advantage, bump the wrap's `nahual-flake` input:

```bash
nix flake update nahual-flake
```

If your wrap still pins `nix/nahual.nix` directly, replace it with the
flake-input pattern (see "Wiring nahual via flake input" above) so future
nahual bumps are a single `nix flake update`.

## Reference templates

The actual file skeletons live next to this skill:

- `references/server.py.template` — full server with NCZYX preprocess
- `references/server-non-image.py.template` — for 2-D `(N, features)` inputs
- `references/server-bioimageio.py.template` — generic loader for any BIMZ
  model identifier (one server, model picked at `setup()` time)
- `references/flake.nix.template` — `apps.default` + `devShells.default`
- `references/flake-tensorflow.nix.template` — TF/Keras variant with the
  `dontCheckRuntimeDeps` override
- `references/flake-conda.nix.template` — micromamba bootstrap variant for
  packages with conda-only deps (vigra, nifty, torch_em, python-elf)
- `references/nahual.nix.template` — Nahual python package pin
- `references/basic_test.py.template` — standalone smoke harness with
  importlib-based server load (avoids sys.path shadow)
- `references/example-client.py.template` — nahual/examples/<model>.py skeleton

Read whichever template matches the model class before editing in place;
don't reconstruct these from memory each time.
