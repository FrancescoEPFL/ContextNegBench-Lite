# Scripts Guide

The scripts are kept in one folder for stable command paths. Use this guide to identify the main entrypoints.

## Demo

| script | purpose |
| --- | --- |
| `smoke_test.py` | Fast temporary synthetic run; no model download or web data. |
| `generate_sample_synthetic_dataset.py` | Regenerate the committed CC0 synthetic sample dataset. |
| `reproduce_paper_tables.py` | `--small` runs the synthetic demo; `--full` validates frozen public result tables. |
| `benchmark_runtime_memory.py` | Small runtime benchmark over the synthetic sample. |

## Analysis

| script | purpose |
| --- | --- |
| `run_dog_grass_false_negation_analysis.py` | Main dog/grass diagnostic. |
| `run_final_contextneg_analysis.py` | With/without scenario diagnostics. |
| `run_model_matrix.py` | Multi-model run over selected analyses. |
| `aggregate_model_matrix.py` | Aggregates model-matrix outputs into public summary tables. |
| `analyze_negation_delta_consistency.py` | Text-space negation delta analysis. |
| `analyze_logical_connector_embeddings.py` | Text-space logical connector analysis. |
| `analyze_lexical_bias_baselines.py` | Caption length and object-word-overlap baseline. |

## Data

| script | purpose |
| --- | --- |
| `search_download_context_neg_images.py` | Search/download helper for candidate web images. |
| `enrich_context_neg_datasets.py` | Rate-limit-friendly dataset enrichment. |
| `make_context_neg_review_gallery.py` | Builds local review galleries. |
| `check_context_neg_dataset.py` | Checks local reviewed dataset structure/counts. |
| `build_context_neg_annotations.py` | Builds annotations from reviewed folders. |
| `build_human_sanity_table.py` | Summarizes reviewed local scenario counts. |

## Validation

| script | purpose |
| --- | --- |
| `validate_result_schemas.py` | Validates public CSV schemas. |

## Recommended First Commands

```bash
python scripts/smoke_test.py
python scripts/reproduce_paper_tables.py --small
python scripts/reproduce_paper_tables.py --full
python scripts/validate_result_schemas.py
```
