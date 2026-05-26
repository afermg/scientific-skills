---
name: compose-notebook
description: Compose a new marimo notebook in the aliby repo by reusing @app.function helpers from the existing catalog (notebooks/nb01 through nb03) to build end-to-end image analysis pipelines — segmentation, feature extraction, deep learning embeddings, or data exploration. Trigger whenever the user asks for a notebook, pipeline, analysis, or visualization that touches microscopy images, tiling, cellpose segmentation, CellProfiler-style feature extraction, Nahual deep learning embeddings, or dataset loading — even if they don't explicitly say "marimo" or "reuse the catalog". Also trigger when the user says "write me a notebook that..." inside aliby, asks to build on top of nb01-nb03, or wants to combine segmentation with embedding in a single workflow.
---

# Compose a new marimo notebook from the aliby catalog

## What this skill is for

The aliby repo holds a catalog of marimo notebooks (`notebooks/nb01_*.py`
through `notebooks/nb03_*.py`) whose `@app.function` helpers handle the
expensive plumbing for microscopy image analysis: locating datasets (local
or Zenodo), discovering image files via regex, loading images, running
Cellpose segmentation, extracting CellProfiler-style features, and computing
deep learning embeddings via Nahual. When a user wants to answer a question —
"segment these images and extract features", "embed this dataset with
DINOv2", "load my images and visualize them" — the right move is to
**compose a notebook from these helpers**, not to write a new pipeline from
scratch. This skill tells you how.

## The catalog at a glance

Every catalog file defines `@app.function` helpers (top-level pure functions,
safe to import from other notebooks) and UI cells that exercise them. The
functions are the contract; the UI cells are illustrative.

| Module | Reusable functions | What they do |
|---|---|---|
| `nb01_data_loading` | `get_data_path(Path)`, `dataset_catalog()`, `load_dataset_dir(data_path, ds_info)`, `load_image_from_position(position, ds_info)` | Locate test data (local or Zenodo download), discover datasets via `DatasetDir` with regex + capture order, load images via `ImageList`. |
| `nb02_cellpose_pipeline` | `parse_seg_channels(text)`, `parse_ext_channels(text)` | Parse user-supplied segmentation channel specs (`"nuclei:1, cell:0"`) and extraction channel lists (`"0, 1, 2"`). The notebook also demonstrates `build_pipeline_steps` and `run_pipeline_and_post` for full segmentation + extraction. |
| `nb03_deep_learning` | `build_embed_pipeline(input_path, address, tile_size, ds_info, model_config)` | Build Nahual embedding pipelines for any supported model (OpenPhenom, MorphEM, DINOv2, SubCell). Returns a pipeline dict ready for `run_pipeline_and_post`. |

When the question isn't obviously one of the above, read the catalog file
itself (not just this table) before deciding something is missing.

## The composition pattern

A composed notebook is a new file in `notebooks/` (e.g.,
`nb04_combined_analysis.py`) that imports catalog helpers as plain Python
and glues them together. Three things matter: **imports**, **interactive
UI**, and **guarded execution**.

### 1. Imports — plain Python, no hacks

Marimo's `@app.function` decorator exposes functions at module top level.
Since the catalog files use `nbNN_` prefixes (valid Python module names)
and marimo automatically adds the notebook's directory to `sys.path`,
a normal import is all you need:

```python
@app.cell
def _():
    from pathlib import Path
    import marimo as mo
    import nb01_data_loading as nb01
    return Path, mo, nb01
```

**Do not** use `sys.path.insert`, `importlib`, or any other hack. If `aliby`
is listed as a dependency in the `# /// script` header, it's importable
directly. If sibling notebooks are in the same directory, marimo handles it.

### 2. Interactive UI — widgets for exploration, not static scripts

The point of a composed notebook is to let the user *explore* — change the
dataset, pick different channels, try a different model — not to produce a
single static output.

**Data source pattern** — every notebook should start with flexible inputs:

```python
@app.cell
def _(mo, nb01, Path):
    test_data_path = nb01.get_data_path(Path)
    catalog = nb01.dataset_catalog()

    dataset_dropdown = mo.ui.dropdown(
        options={"(custom path)": None, **{d["name"]: d for d in catalog}},
        value=catalog[0]["name"],
        label="Test dataset (from Zenodo)",
    )
    dataset_dropdown
    return catalog, dataset_dropdown, test_data_path

@app.cell
def _(dataset_dropdown, mo, test_data_path):
    _selected = dataset_dropdown.value
    if _selected is not None:
        _default_folder = str(test_data_path / _selected["name"])
        _default_regex = _selected["regex"]
        _default_capture = _selected["capture_order"]
    else:
        _default_folder = ""
        _default_regex = ".*__([A-Z][0-9]{2})__([0-9])__([A-Za-z]+).tif"
        _default_capture = "WFC"

    folder_input = mo.ui.text(value=_default_folder, label="Image folder", full_width=True)
    regex_input = mo.ui.text(value=_default_regex, label="Filename regex", full_width=True)
    capture_order_input = mo.ui.text(value=_default_capture, label="Capture order")
    mo.vstack([folder_input, regex_input, capture_order_input])
    return folder_input, regex_input, capture_order_input
```

This pattern gives the user:
- A dropdown to pull test datasets from Zenodo (auto-downloaded on first use)
- Text inputs pre-filled from the selection, overridable for custom data
- Reactive updates downstream when any input changes

**Result tables** — use `mo.ui.table(df, selection="single")` for any result
set the user might want to click through. Wire downstream cells to `.value`.

**Run buttons** — guard expensive operations (pipeline execution, embedding)
with `mo.ui.run_button()` + `mo.stop(not run_button.value)` so they only
execute on explicit user action, not on every upstream change.

### 3. Pipeline construction — compose, don't duplicate

Build pipelines by combining catalog helpers rather than copying their
internals:

```python
@app.cell
def _(Path, folder_input, regex_input, capture_order_input):
    from aliby.io.dataset import DatasetDir

    ds_info = {
        "name": Path(folder_input.value).name,
        "regex": regex_input.value,
        "capture_order": capture_order_input.value,
    }
    dset = DatasetDir(Path(folder_input.value), regex=ds_info["regex"],
                      capture_order=ds_info["capture_order"])
    positions = dset.get_position_ids()
    return ds_info, positions
```

For segmentation + extraction, use `aliby.pipe_builder.build_pipeline_steps`
(as demonstrated in nb02). For embeddings, use
`nb03.build_embed_pipeline` (as demonstrated in nb03). For combined
workflows, chain them — tile once, then branch into segmentation and
embedding steps.

## Running the notebook

Each notebook has a PEP 723 `# /// script` header so it works standalone:

```bash
uv run --script notebooks/nb04_combined_analysis.py
```

For interactive development, use the **marimo-pair** skill to edit cells
in a live kernel. Once the file exists on disk:

```bash
.venv/bin/marimo edit notebooks/nb04_combined_analysis.py --port 2719 --no-token
```

When working against a running server via marimo-pair, use `code_mode`
to create and edit cells. Do not write to the `.py` file directly while
a session is running — the kernel owns it.

### Saving from code_mode

Marimo's `code_mode` edits are live in memory but not auto-saved to disk.
To persist changes programmatically:

```python
import marimo._code_mode as cm
async with cm.get_context() as ctx:
    doc = ctx._document
    from marimo._ast.codegen import generate_filecontents
    from marimo._runtime.context.utils import get_context
    from pathlib import Path

    names = [c.name if c.name else "_" for c in doc.cells]
    contents = generate_filecontents(
        codes=[c.code for c in doc.cells],
        names=names,
        cell_configs=[c.config for c in doc.cells],
    )
    Path(get_context().filename).write_text(contents)
```

After saving, restore the `# /// script` header and `App(width="medium")`
— `generate_filecontents` strips both. Also ensure no cell has an empty
name (`""`) — replace with `"_"` to avoid `_unparsable_cell` errors.

## Cell output rendering — last expression only

marimo captures **the last top-level expression** of a cell as output.
Statements (`if/else`, `for`, `while`, `with`, `try`) are not expressions,
so `mo.vstack([...])` placed inside an `if` branch is evaluated and
discarded — the cell renders nothing.

```python
# WRONG — nothing renders
if ok:
    mo.vstack([mo.md("### Result"), table])
else:
    mo.md("*no data*")

# RIGHT — assign to a variable, then evaluate it as the last expression
if ok:
    _out = mo.vstack([mo.md("### Result"), table])
else:
    _out = mo.md("*no data*")
_out
```

For plotly widgets specifically, `mo.output.replace(chart)` also works
and avoids stale output issues.

## Known gotchas

- **Bare widget expressions trigger ruff B018.** Marimo renders the last
  expression in a cell, so `dataset_dropdown` on a bare line is
  intentional. Add `# noqa: B018` to suppress the lint warning.
- **`if/else` as a cell's last top-level construct renders nothing.** See
  the rendering section above — assign to `_out` and end with `_out`.
- **Bool toggles default to `True` when the downstream UI depends on them.**
  A `mo.ui.switch(value=False)` gating an `mo.stop()` hides the cell's
  content until the user clicks — preview defaults to empty, which reads as
  "broken". Default to `True` unless the gated step is genuinely expensive.
- **`mo.image(src=...)` accepts bytes directly.** Raw PNG bytes from
  `io.BytesIO` render inline via marimo's `./@file/...` URLs — no data URLs
  or temp files needed.
- **`ctx.cells` yields `NotebookCell` objects, not IDs.** Iterate as
  `for cell in ctx.cells: cell.id, cell.code, cell.status, cell.errors`.
  Index access `ctx.cells["TqIu"]` returns a view by id.
- **`ctx.set_ui_value(element, value)` for reactive UI updates.** Works on
  `mo.ui.switch`, slider, text — triggers downstream cells to re-run.
  Does NOT work on `mo.ui.dropdown`; construct with `value=...` instead.
- **`code_mode` context lacks `get_graph()`.** Use `ctx.cells` to walk the
  notebook and filter by `cell.code` substring to locate a specific cell,
  then act via `cell.id`.
- **`create_cell` produces empty names.** Cells created via
  `ctx.create_cell()` get `name=""` instead of `"_"`. Fix names before
  saving with `generate_filecontents` or codegen will produce
  `_unparsable_cell` wrappers.
- **`generate_filecontents` strips script headers.** The `# /// script`
  PEP 723 block and `App(width=...)` config are lost on codegen save.
  Re-add them after writing the file.
- **Don't mix `sys.path` hacks with installed packages.** If `aliby` is
  in the `# /// script` dependencies, it's importable directly. Adding
  `../src` to `sys.path` creates version conflicts and import confusion.
- **`mo.ui.dropdown(options=dict, value=…)` wants the *label*, not the
  underlying value.** When `options` is a dict, the keys are the dropdown
  labels and the values are what `dropdown.value` resolves to. The
  `value=` parameter must match one of the **keys**. Passing the
  underlying mapping value (e.g. an int) raises `ValueError: The option
  name 'N' is not a valid option.` The trap: most other `mo.ui.*` widgets
  accept the underlying value, so muscle memory misleads you.
- **`alt.Chart(big_df.to_pandas())` serialises every column.** A UMAP
  scatter over a 240-col profiles frame produces a ~25 MB vega spec,
  which marimo refuses to render (`Your output is too large …`). Always
  `df.select([cols_used_by_the_chart]).to_pandas()` before passing into
  Altair — typically 2-3 orders of magnitude smaller. Same trap with
  `unpivot()` for facet panels: pre-select the columns you actually face.
- **`alt.Chart(...).facet(facet=alt.Facet("x:N"), columns=N)` — `columns`
  belongs on `.facet()`, not on `alt.Facet()`.** vega-lite accepts
  `columns` only at the facet-composition layer; nested inside
  `alt.Facet(...)` it raises `Facet has no parameter named 'columns'`.
- **Polars `.select(list_of_cols)` errors on duplicates.** When you
  build a column list from a UI dropdown + tooltip list + color encoding
  and the same name appears more than once,
  `DuplicateError: the name 'X' is duplicate` fires. Pre-deduplicate with
  `list(dict.fromkeys(cols))` to preserve order *and* drop repeats.
- **Don't run formatters (`ruff format`, `black`, `treefmt`, `nix fmt`)
  while a marimo kernel is editing the same file.** The kernel owns the
  on-disk `.py` while running; a parallel formatter overwrites your
  live `code_mode` edits silently. Stop the kernel first, format, then
  restart. The reverse trap also applies: `code_mode` edits a cell in
  the kernel's memory but the disk file may briefly disagree until the
  kernel autosaves — don't read the file expecting kernel state.
- **Marimo only accepts a parquet column name as a key, not an index.**
  If your aggregation step renames feature columns to positional aliases
  (e.g. duckdb `AVG(COLUMNS('.*/.*')) AS "avg"` produces `avg`, `avg_1`,
  `avg_2`, …), downstream notebooks lose channel identity. Either keep
  the original names through the aggregation, or re-load the per-cell
  parquets when channel-keyed columns matter.
- **jupyter-scatter inside marimo: six traps stack on top of each
  other, and skipping any one leaves the plot looking broken.**
  1. `scatter.show()` / `scatter.widget` returns a `JupyterScatter`
     whose default MIME bundle is the traditional Jupyter-widget type
     (`application/vnd.jupyter.widget-view+json`). marimo silently
     refuses to render that — the cell renders blank. Wrap with
     `mo.ui.anywidget(scatter.widget)` so marimo treats it as the
     anywidget it actually is.
  2. `Scatter.__init__` defaults `_opacity_by='density'`. A subsequent
     `.opacity(default=0.85)` flips `_opacity_by` back to `None` but
     does **not** clear the 256-stop `_opacity_map` that was seeded
     during the density init. Widget construction then prefers the
     truthy map over the scalar default, and points render with a
     density ramp (mostly near-zero alpha — "invisible until I select
     them"). After `.opacity(...)` and before reading `.widget`, force
     the scalar path with `scatter._opacity_map = None`.
  3. `opacity_unselected` defaults to `0.5` and multiplies the alpha of
     every point *not* in the selection set the moment any point is
     selected. Combined with the density-ramp bug above this looks
     identical, but it bites independently — e.g. a stray lasso. Pass
     `.opacity(default=…, unselected=1.0)` to disable selection-state
     dimming entirely. Reach for `opacity_unselected < 1` only when you
     actively want selection-driven contrast.
  4. `.color(by="categorical_str_col")` calls `sum()` on the colour
     column to detect NaNs. For object/string columns with any `None`,
     `sum(['x', None, ...])` raises
     `TypeError: can only concatenate str (not "NoneType") to str` and
     the chart cell errors out, leaving everything downstream
     undefined. Filter null rows from the colour column *before*
     handing the DataFrame to jscatter:
     `df.filter(pl.col(color_col).is_not_null())`.
  5. **Mutating `widget.selection` / `widget.filter` from a downstream
     cell races with WebGL init and silently kills the renderer.** If
     a separate cell assigns to these traits while jscatter is still
     standing up its WebGL canvas (or even on a freshly recreated
     widget, before the frontend has bound the new model_id), the
     instance ends up in a broken state for its entire lifetime — the
     Python traits look fine but the canvas is blank, the lasso
     doesn't work, and clearing the trait back to `None` does not
     recover. The fix: **bake selection/filter into the widget at
     construction time** via `scatter.selection(idx)` /
     `scatter.filter(idx)` chained *before* reading `.widget`. Every
     change to the picker/filter input rebuilds the chart cell, which
     constructs a fresh widget with the right state from the start —
     no post-mutation, no race. If you must mutate after the fact (eg.
     "clear on empty after a pick"), only do it when the trait
     actually has content, so a freshly-built widget is never touched.
  6. **`mo.ui.anywidget(widget).value` reacts to every synced trait,
     including `hovered_point`.** A downstream cell that reads
     `chart.value['selection']` (or any other key) re-runs whenever
     `chart.value` changes — and jscatter updates `hovered_point` on
     every mouse move, so an image grid downstream flickers at ~60 fps
     as the user moves the cursor. Use the
     [`mo.state` + `.observe(names=[...])` pattern from the marimo
     docs](https://docs.marimo.io/api/inputs/anywidget/) to bridge
     **only the specific traits you care about** into marimo's
     reactive graph:

     ```python
     chart = mo.ui.anywidget(scatter.widget)  # still wrap so it renders
     get_sel, set_sel = mo.state(scatter.widget.selection)
     scatter.widget.observe(
         lambda _c: set_sel(scatter.widget.selection),
         names=["selection"],
     )
     # downstream cell reads get_sel(), not chart.value
     ```

     The chart cell returns both `chart` (for rendering) and `get_sel`
     (for downstream). Cells that read `get_sel()` re-run only on
     selection changes — no flicker.

## Process for a new composition

When the user gives you a question like "segment these images and extract
features, then embed them with DINOv2":

1. **Map the English to catalog calls.** Which nbN functions give you
   each step? If you need something not in the table, read the catalog
   file itself before inventing new code.
2. **Start with the data source pattern.** Dropdown + text inputs for
   folder/regex/capture_order, wired to `DatasetDir`.
3. **Draft the notebook with `@app.function` helpers first**, then
   `@app.cell` UI on top. Keep imports clean — no `sys.path` hacks.
4. **Guard expensive steps** (pipeline execution, model inference) behind
   `mo.ui.run_button()` + `mo.stop()`.
5. **Use `mo.ui.table(..., selection="single")`** for any result set the
   user might want to click through, and wire downstream cells to `.value`.
6. **Run it in a live kernel (marimo-pair) and iterate on cells in place.**
7. **After the first successful run**, look for anything expensive you're
   repeating on every edit and consider caching it.

## When *not* to use this skill

- If the user wants to modify an existing catalog notebook (e.g., fix a
  bug in `nb02`), edit that file directly.
- If the task is pure infrastructure (venv setup, CI, CLI tooling), this
  skill doesn't apply.
- If the question is about the old `examples/` scripts (non-marimo), those
  are legacy — suggest converting to a marimo notebook instead.
