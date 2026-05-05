# Results Guide

Start here:

| path | purpose |
| --- | --- |
| `model_matrix_summary/summary.md` | Compact public result summary. |
| `model_matrix_summary/dog_grass_by_model.csv` | Full dog/grass model-matrix metrics. |
| `model_matrix_summary/final_contextneg_by_model.csv` | Full with/without scenario metrics. |
| `model_matrix_summary/model_manifest.csv` | Model, pretrained weights, device note, and run date. |
| `selected_figures/` | README-ready plots and diagnostic figures. |
| `full_research_report.md` | Full narrative report. |
| `reproducibility/result_fingerprints.json` | Hashes and environment metadata for frozen public results. |
| `sample_synthetic_demo/` | Outputs from the license-safe quick demo. |
| `lexical_bias_baselines/` | Caption length and object-word-overlap baseline outputs. |

The reviewed web-image datasets are not included. Public result summaries are frozen in `model_matrix_summary/` and can be validated with:

```powershell
python scripts/reproduce_paper_tables.py --full
python scripts/validate_result_schemas.py
```
