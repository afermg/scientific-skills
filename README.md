# Scientific Skills

A collection of Claude Code skills for reproducible scientific data analysis and notebook composition.

## Skills

| Skill | Description |
|---|---|
| [aliby-pipeline](aliby-pipeline/) | Run the [aliby](https://github.com/afermg/aliby) image analysis pipeline with [nahual](https://github.com/afermg/nahual) cellpose/trackastra GPU servers for segmentation, tracking, and feature extraction |
| [compose-notebook](compose-notebook/) | Compose marimo notebooks from the aliby catalog to build microscopy image analysis pipelines (segmentation, feature extraction, deep learning embeddings) |
| [org-duckdb-analysis](org-duckdb-analysis/) | Create Org Mode literate notebooks that use DuckDB SQL queries via org-babel to explore and analyze tabular data (CSV, Parquet) inside Emacs |
| [nahual-wrap-model](nahual-wrap-model/) | Wrap a third-party deep learning model into a Nix flake exposing a [nahual](https://github.com/afermg/nahual) server (server.py + flake.nix + basic_test.py), GPU-validated, pushed to a fork under `afermg` |

## Installation

Add the marketplace from GitHub, then install plugins:

```text
/plugin marketplace add afermg/scientific_skills
/plugin install aliby-pipeline@scientific-skills
/plugin install compose-notebook@scientific-skills
/plugin install org-duckdb-analysis@scientific-skills
/plugin install nahual-wrap-model@scientific-skills
```

Pin a ref with `afermg/scientific_skills@<branch|tag|sha>`, or pass a local path for development. Refresh with `/plugin marketplace update scientific-skills`.

To pick up new versions automatically, enable auto-update: `/plugin` → **Marketplaces** → select **scientific-skills** → **Enable auto-update** (checked on startup, applies to all plugins from this marketplace).

## Repo layout

```
.claude-plugin/marketplace.json      # marketplace manifest
<plugin>/.claude-plugin/plugin.json  # per-plugin manifest
<plugin>/skills/<plugin>/SKILL.md    # skill body + YAML frontmatter
```

## Adding a new skill

1. Create `<plugin>/.claude-plugin/plugin.json` with `name`, `description`, `version`.
2. Put the skill at `<plugin>/skills/<plugin>/SKILL.md` (frontmatter: `name`, `description`).
3. Register the plugin in `.claude-plugin/marketplace.json` under `plugins`.
