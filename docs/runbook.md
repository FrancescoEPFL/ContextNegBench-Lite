# Runbook

Use these commands from the repository root.

## 1. Install

```powershell
python -m pip install -r requirements.txt
python -m pip install ddgs
```

`openclip_vit_b16_siglip` uses a Hugging Face tokenizer through OpenCLIP, so `transformers` must be installed. It is included in `requirements.txt`.

## 2. Rate-Limit-Friendly Dataset Enrichment

For a no-download demo path, use the bundled generated sample first:

```powershell
python scripts/reproduce_paper_tables.py --small
python scripts/benchmark_runtime_memory.py
python scripts/validate_result_schemas.py
```

The sample dataset can be regenerated with:

```powershell
python scripts/generate_sample_synthetic_dataset.py --output data/sample_synthetic --samples-per-task 3 --seed 101
```

Dry run first:

```powershell
python scripts/enrich_context_neg_datasets.py `
  --target-per-folder 80 `
  --batch-size-per-job 8 `
  --max-results-per-query 8 `
  --sleep-seconds 5 `
  --jitter-seconds 4 `
  --max-downloads-per-minute 6 `
  --dry-run
```

Download in small batches:

```powershell
python scripts/enrich_context_neg_datasets.py `
  --target-per-folder 80 `
  --batch-size-per-job 8 `
  --max-results-per-query 8 `
  --sleep-seconds 5 `
  --jitter-seconds 4 `
  --max-downloads-per-minute 6 `
  --resume `
  --stop-on-rate-limit
```

If rate-limited, wait and re-run the same command with `--resume`.

## 3. Manual Review

Open each gallery and move ambiguous images to `reviewed/rejected`.

```powershell
Start-Process data/context_neg/cat_sofa/review_gallery.html
Start-Process data/context_neg/person_beach/review_gallery.html
Start-Process data/context_neg/bicycle_street/review_gallery.html
```

Then check counts:

```powershell
python scripts/check_context_neg_dataset.py --root data/context_neg/cat_sofa --scene "living room" --object cat
python scripts/check_context_neg_dataset.py --root data/context_neg/person_beach --scene beach --object person
python scripts/check_context_neg_dataset.py --root data/context_neg/bicycle_street --scene street --object bicycle
```

## 4. Build Annotations

```powershell
python scripts/build_context_neg_annotations.py --root data/context_neg/cat_sofa --scene "living room" --object cat --languages en
python scripts/build_context_neg_annotations.py --root data/context_neg/person_beach --scene beach --object person --languages en
python scripts/build_context_neg_annotations.py --root data/context_neg/bicycle_street --scene street --object bicycle --languages en
```

## 5. Human Sanity Table

```powershell
python scripts/build_human_sanity_table.py `
  --scenarios kitchen_table street_car cat_sofa person_beach bicycle_street `
  --output results/human_sanity_table.csv
```

## 6. Main Research Runs

Existing headline scenarios:

```powershell
python scripts/run_final_contextneg_analysis.py `
  --scenarios kitchen_table street_car `
  --model openclip_vit_b32 `
  --output results/final_contextneg_analysis `
  --bootstrap-samples 1000 `
  --batch-size 8
```

Extended scenarios after review:

```powershell
python scripts/run_final_contextneg_analysis.py `
  --scenarios kitchen_table street_car cat_sofa person_beach bicycle_street `
  --model openclip_vit_b32 `
  --output results/final_contextneg_analysis `
  --bootstrap-samples 1000 `
  --batch-size 8
```

Dog/grass diagnostic with detailed-generic control:

```powershell
python scripts/run_dog_grass_false_negation_analysis.py `
  --root data/context_neg/dog_grass_false_negation `
  --model openclip_vit_b32 `
  --output results/dog_grass_false_negation `
  --bootstrap-samples 1000 `
  --batch-size 8
```

## 7. Model Matrix

Use this after the single-model run is stable:

```powershell
python scripts/run_model_matrix.py `
  --models openclip_vit_b32 openclip_vit_b32_openai openclip_vit_b32_datacomp openclip_rn50 openclip_vit_b16_openai openclip_vit_b16_siglip `
  --analyses final dog_grass `
  --scenarios kitchen_table street_car cat_sofa person_beach bicycle_street `
  --output-root results/model_matrix `
  --bootstrap-samples 1000 `
  --batch-size 8
```

## 8. Tests

```powershell
python -m compileall src scripts
python -m ruff check src scripts tests
python -m mypy src/negcompbench/eval/schema_validation.py scripts/reproduce_paper_tables.py scripts/validate_result_schemas.py
python -m pytest -q
python scripts/validate_result_schemas.py
```

## 9. Aggregate Model Matrix

```powershell
python scripts/aggregate_model_matrix.py `
  --root results/model_matrix `
  --output results/model_matrix_summary
```

## 10. Reproduce Public Tables

Validate and fingerprint the frozen public result tables:

```powershell
python scripts/reproduce_paper_tables.py --full
```

Run the lightweight sample pipeline:

```powershell
python scripts/reproduce_paper_tables.py --small
```
