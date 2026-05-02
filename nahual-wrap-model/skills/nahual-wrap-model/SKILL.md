---
name: nahual-wrap-model
description: >
  Wrap a third-party ML model (typically vision/microscopy or single-cell) as a
  Nahual server: fork the upstream repo to afermg/<repo>, add server.py +
  flake.nix + nix/nahual.nix + basic_test.py + .envrc, validate it runs on GPU,
  push, and add a client example to the nahual repo. TRIGGER when: user asks
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
2. **Locally clone** under `/home/amunoz/projects/nahual_models/<repo>/`.
3. **Add the five required files** (templates in `references/`):
   - `server.py` — Nahual setup/process server
   - `flake.nix` — `apps.default` (`nix run`) + `devShells.default` (`nix develop`)
   - `nix/nahual.nix` — pin nahual itself as a Python package
   - `basic_test.py` — standalone smoke test (no IPC, just model + forward)
   - `.envrc` — single line: `use flake .`
4. **Pick a `nixpkgs` channel.** Default to `nixos-unstable`. Fall back to
   `nixos-24.11` only if the model's deps (typically `timm`, `tensorflow`,
   ancient `transformers`) are too painful on unstable.
5. **Make GPU work.** `cudaSupport = true` is non-negotiable — even if the TF
   wheel takes 6 hours to build the first time. CPU-only wraps are not
   acceptable.
6. **Validate twice:**
   - `nix develop --impure --command python basic_test.py` (no IPC)
   - `nix run --impure .#default -- ipc:///tmp/<name>.ipc` + a client roundtrip
   Both must show `device: cuda:0` (torch) or `device: GPU:0` (TF).
7. **Push to `git@github.com:afermg/<repo>.git`** (HTTPS hangs — use SSH).
   Touch only the new files; never overwrite or delete anything from upstream
   (especially `.github/workflows/`).
8. **Add a client example** at `/home/amunoz/projects/nahual/examples/<model>.py`,
   mirroring `dinov2.py`'s style.
9. **Update the `model-links` branch** of the nahual repo with the new fork URL.

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

### `nix/nahual.nix`

Pins the Nahual transport layer as a Python package. Update `rev` and
`sha256` only when nahual itself moves. Copy verbatim from
`references/nahual.nix.template` for new wraps — the only thing that ever
changes is `version`, `rev`, and `sha256`.

### `basic_test.py`

Smoke test that proves the model loads + forwards inside the dev shell,
without going through IPC. Pattern:

```python
import sys
if len(sys.argv) < 2:
    sys.argv.append("ipc:///tmp/<model>_basic_test.ipc")  # placate server.py's argv[1]

from server import setup
processor, info = setup()
print(info)
out = processor(<small synthetic numpy array>)
print(type(out), out.shape)
```

Run with `nix develop --impure --command python basic_test.py`. Template:
`references/basic_test.py.template`.

### `.envrc`

Literally just:

```
use flake .
```

If the upstream `.gitignore` excludes `.envrc` (channelsformer did), force-add
it: `git add -f .envrc`.

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

**Override numba's pytestCheckPhase** when a custom Python override forces
numba to rebuild — its test suite is 30+ minutes serial. Pattern from ultrack:
```nix
python = pkgs.python313.override {
  packageOverrides = pyfinal: pyprev: {
    numba = pyprev.numba.overridePythonAttrs (_: {
      doCheck = false; doInstallCheck = false;
      pytestCheckPhase = "true"; installCheckPhase = "true";
    });
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
| Conda-only deps (`nifty`, `vigra`, `torch_em`, `python-elf`) | No PyPI distributions; nixpkgs Python packages don't exist | Fall back to a `micromamba`-bootstrapped conda env *inside* the flake; nix shells `micromamba` + `cudaPackages.cudatoolkit` + `cudnn` only. See micro-sam wrap. Document the cold-cache 3-4 min env solve. Always prepend `/run/opengl-driver/lib` to `LD_LIBRARY_PATH` in the wrapper for CUDA visibility. |
| ILP / Gurobi-style solver in upstream means "GPU mandatory" doesn't fit | Some tools (ultrack) have a CPU-bound combinatorial core | Run the *learning* parts on GPU and document honestly which steps are CPU. Don't ship CPU-only torch when the underlying detection nets *can* use GPU. |

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
   `flake.nix`, `flake.lock`, `nix/`, `basic_test.py`, `.envrc`, and at most
   a tiny README note. If `.github/workflows/` shows up in the diff, revert
   it. (Cellpose, dinov2, dinov3, trackastra all have upstream workflows
   that must stay intact.)

2. **Push to the fork** (SSH only):
   ```bash
   git remote set-url origin git@github.com:afermg/<repo>.git
   git push -u origin <branch>
   ```

3. **Add the client example.** Mirror `dinov2.py`'s structure, located at
   `/home/amunoz/projects/nahual/examples/<model>.py`. Image models use 5-D
   `NCZYX` random data; non-image models pass `signature=("dict", "numpy")`
   explicitly to `dispatch_setup_process` because their name isn't in the
   built-in registry.

4. **Update the `model-links` branch** on `github:afermg/nahual`:
   ```bash
   cd /home/amunoz/projects/nahual
   git fetch origin model-links
   git switch model-links
   # add a row to README.md under the appropriate section
   git add README.md && git commit -m "docs: add <model> wrap link"
   git push origin model-links
   git switch -                                 # back to your previous branch
   ```

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
nobody has packaged for nixpkgs. Three escape hatches in order of preference:

1. **Write your own `nix/<pkg>.nix`** that fetches a PyPI tarball and
   declares deps manually — the EmbedSeg, InstanSeg, Ultrack, and CellSAM
   wraps all do this. Set `dontCheckRuntimeDeps = true` to skip wheel
   metadata mismatches and `pythonRuntimeDepsCheck = false` for older
   nixpkgs. List only the deps that the *server* import path actually pulls
   in (not the entire pyproject `dependencies` list — many are unused at
   inference time).
2. **micromamba bootstrap inside the flake** when the dep is conda-only
   (vigra-py, nifty, torch_em, python-elf — all conda-forge exclusives).
   Pattern from micro-sam wrap: nix shells `micromamba` + cudaPackages, the
   `runserver.sh` does `micromamba create -p ~/.cache/<model>/env -y -f
   nix/env-server.yaml` on first run, then `micromamba run -p <env>
   python server.py ...`. Cold cache: ~3-4 min env solve, ~4 GB download;
   subsequent runs reuse the env. **Critical**: prepend
   `/run/opengl-driver/lib` to `LD_LIBRARY_PATH` in `runserver.sh` so the
   conda-shipped torch sees NixOS's `libcuda.so` driver stub.
3. **Generic loader wraps** for model-zoo packages (`bioimageio.core`).
   One server, accepts a model identifier (`"affable-shark"`, a DOI, a
   zenodo URL, a local rdf.yaml) at `setup()` time. Default to ONNX or
   TorchScript weight formats (self-contained — no `architecture.source`
   import is needed). Tier-2 flake outputs (`apps.with-stardist`,
   `apps.with-careamics`, `apps.with-monai`) for the popular
   custom-architecture families share one `server.py`. See the
   `nahual_bioimageio` wrap.

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
