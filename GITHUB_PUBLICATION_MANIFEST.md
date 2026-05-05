# GitHub Publication Manifest

This folder is the clean GitHub-ready version of ContextNegBench-Lite.

## Included

- `README.md`: concise research-style project overview.
- `docs/`: methodology, metrics, limitations, results summary, and appendices.
- `src/`: reusable evaluation/model/data utilities.
- `scripts/`: reproduction and analysis entry points.
- `tests/`: lightweight tests.
- `configs/models/`: model preset configuration files.
- `results/model_matrix_summary/`: compact precomputed cross-model summaries.
- `results/selected_figures/`: selected plots referenced by the README/docs.
- `results/full_research_report.md`: comprehensive report.
- `results/human_sanity_table.csv`: dataset count and review-protocol summary.

## Excluded

- Downloaded web images.
- Reviewed/raw dataset folders.
- Metadata logs and review galleries.
- Per-model bulk result folders.
- Archive/smoke/cache outputs.

## Why Datasets Are Excluded

The image data is web-collected and may have licensing constraints. The public repo should provide enough code, prompts, methodology, and result summaries for review without redistributing image assets.

Users who want to reproduce the full analysis should collect candidate images with the scripts, perform human review, build annotations, and then run the analysis commands in `README.md` or `docs/runbook.md`.
