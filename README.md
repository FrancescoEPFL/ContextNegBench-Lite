# When False Negation Beats Partial Truth: Object Specificity in CLIP Image-Text Matching

**A low-compute semantic embedding analysis of negation, object specificity, and logical connectors in CLIP-style models.**

## Abstract

CLIP-style models embed images and text into a shared vector space, then compare image-text similarity. Prior work has shown that these models can struggle with negation and compositional meaning, especially when small linguistic changes reverse the truth of a caption.

This project asks a narrower question: can a false negated caption that names a visible object score above a true but underspecified caption? For example, if an image contains a dog on grass, `an image with no dog` is false but contains the salient object word `dog`, while `an image of a grassy field` is true but incomplete.

The main result is yes: across several CLIP-style models in a dog-on-grass diagnostic, false object-specific negation often beats true generic partial descriptions. However, the fully correct positive caption usually remains top-ranked. The result is therefore not "false beats truth"; it is a more specific failure mode where object specificity can overpower weakly grounded absence semantics.

## Motivating Example

Image: a dog on grass

Captions:

A. `an image with no dog`  
B. `an image of a grassy field`  
C. `an image of a dog on a grassy field`

A is false but object-specific. B is true but incomplete. C is fully correct. This project tests whether A can beat B, and whether C still beats both.

## Research Questions

- Are affirmative and negated captions close in CLIP text embedding space?
- Can false object-specific negation beat true generic partial descriptions?
- Does the effect persist across multiple CLIP-style models?
- Does the effect weaken when the true caption becomes more detailed?
- Are negation and logical connectors represented as structured text-space directions?
- Do image-side object separability and scenario choice affect behavior?

## Methodology

No model is trained. All experiments use frozen embeddings and image-text scores from CLIP-style models. This is a diagnostic study, not a benchmark leaderboard.

Models tested:

- `openclip_vit_b32`
- `openclip_vit_b32_openai`
- `openclip_vit_b32_datacomp`
- `openclip_vit_b16_openai`
- `openclip_vit_b16_siglip`
- `openclip_rn50`

Main diagnostic:

- `dog_grass_false_negation`: 74 reviewed images of dogs visible on grass.

Supporting with/without scenarios:

- `kitchen_table`
- `street_car`
- `cat_sofa`
- `person_beach`
- `bicycle_street`

Text-only analyses:

- negation delta consistency
- logical connector embeddings

See [docs/methodology.md](docs/methodology.md) for dataset creation, human review, scoring, and confidence interval details.

## Key Metrics

`false_negated_win_rate_over_generic`: rate at which a false negated caption beats a true but generic caption.

`false_negated_win_rate_over_detailed_generic`: same comparison, but the true caption is strengthened, for example with `with an animal`.

`false_negated_win_rate_over_positive`: rate at which the false negated caption beats the fully correct positive caption.

`top_positive_rate`: rate at which the fully correct positive caption is top-ranked.

`mean_false_absence_tolerance`: for with-object images, `score(scene without object) - score(scene)`.

`image_condition_separation`: distance between with-object and without-object image centroids in image embedding space.

`delta_direction_similarity`: how consistently an operator transformation like `embedding("no X") - embedding("X")` points in a similar direction across objects.

Detailed metric definitions are in [docs/metrics.md](docs/metrics.md).

## Main Result: Dog/Grass False Negation

This is the cleanest stress test in the project. The false negated caption is logically wrong, but it names the visible object. The generic caption is true but incomplete.

Prompt-pair labels:

- `core`: `an image with no dog` vs `an image of a grassy field`
- `field`: `a grassy field with no dog` vs `a grassy field`

![False negated win rate by pair](results/selected_figures/false_negated_win_rate_by_pair.png)

| model | core_false_negated_over_generic | field_false_negated_over_generic | core_false_negated_over_positive | field_false_negated_over_positive | core_top_positive_rate | field_top_positive_rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| openclip_rn50 | 0.851 | 0.986 | 0.041 | 0.054 | 0.919 | 0.919 |
| openclip_vit_b16_openai | 0.757 | 0.986 | 0.081 | 0.054 | 0.892 | 0.946 |
| openclip_vit_b16_siglip | 0.973 | 0.986 | 0.081 | 0.014 | 0.905 | 0.959 |
| openclip_vit_b32 | 0.959 | 1.000 | 0.027 | 0.041 | 0.946 | 0.932 |
| openclip_vit_b32_datacomp | 0.770 | 0.973 | 0.068 | 0.135 | 0.851 | 0.784 |
| openclip_vit_b32_openai | 0.676 | 0.986 | 0.041 | 0.041 | 0.919 | 0.946 |

The false negated caption often beats the true generic caption. It rarely beats the fully correct positive caption. Therefore the result is not "false beats truth." The correct claim is:

```text
false object-specific negation can beat true underspecified descriptions
```

## Why The Control Matters

When the true caption is made more detailed, for example by mentioning `animal`, the false-negation effect weakens in several models and prompt forms. This control prevents overclaiming: the failure is strongest when the true caption is generic or underspecified.

The table averages the `core` and `field` dog/grass prompt pairs.

| model | false_negated_over_detailed_generic | false_negated_over_generic | top_positive_rate |
| --- | ---: | ---: | ---: |
| openclip_rn50 | 0.662 | 0.919 | 0.919 |
| openclip_vit_b16_openai | 0.676 | 0.872 | 0.919 |
| openclip_vit_b16_siglip | 0.547 | 0.980 | 0.932 |
| openclip_vit_b32 | 0.655 | 0.980 | 0.939 |
| openclip_vit_b32_datacomp | 0.588 | 0.872 | 0.818 |
| openclip_vit_b32_openai | 0.547 | 0.831 | 0.932 |

The generic comparison is consistently high. The detailed-generic comparison is lower, but still non-trivial for several models. This suggests that object-specific false negation is most competitive when the true alternative caption omits the salient object or a close visual category.

## Supporting Diagnostics: Scenario Dependence

The with/without scenarios show that behavior depends on object, scene, model, and prompt wording. This section is supporting evidence, not the central result.

| scenario | false_negative_top_rate | mean_false_absence_tolerance | positive_top_rate |
| --- | ---: | ---: | ---: |
| bicycle_street | 0.500 | 0.055 | 0.491 |
| cat_sofa | 0.275 | 0.070 | 0.721 |
| kitchen_table | 0.260 | 0.024 | 0.721 |
| person_beach | 0.089 | 0.011 | 0.867 |
| street_car | 0.210 | 0.016 | 0.749 |

`person_beach` is the cleanest scenario on average. `bicycle_street` is harder. `kitchen_table` is strongly model-dependent. This supports the idea that the failure mode is not universal or fixed; it depends on model, scenario, and prompt wording.

## Image-Side Separability

Image-side separability varies strongly by scenario and model. It helps interpret behavior, but it is not sufficient by itself. For example, `cat_sofa` can have high with/without image separation while still showing false absence tolerance.

Full image condition separation tables are in [docs/results_summary.md](docs/results_summary.md).

## Text-Only Embedding Analysis

We also tested whether negation is pure noise in text space by checking whether:

```text
embedding("no dog") - embedding("dog")
~= embedding("no cat") - embedding("cat")
~= embedding("no car") - embedding("car")
```

| operator | mean_delta_direction_similarity |
| --- | ---: |
| no | 0.7392 |
| without | 0.7692 |
| with no | 0.6833 |
| without any | 0.7967 |
| no visible | 0.8330 |
| absent | 0.7932 |
| not a | 0.8149 |

Negation is not pure noise: operator directions are partially structured. But these directions do not behave as hard logical constraints in image-text scoring. `no visible X` is the most consistent tested absence-like template.

Detailed negation-delta results are in [docs/appendix_negation_delta.md](docs/appendix_negation_delta.md). Logical connector results are in [docs/appendix_logical_connectors.md](docs/appendix_logical_connectors.md).

## Main Conclusion

These diagnostics identify a repeatable scoring failure mode where salient object mentions can overpower weakly grounded absence semantics, especially when the true alternative caption is generic or underspecified.

## What This Does Not Claim

This project does not claim that CLIP never understands negation. It does not claim that CLIP is purely bag-of-words. It does not claim that false captions generally beat true captions.

The claim is narrower: false negated captions that mention a visible object can outrank true but underspecified captions in diagnostic settings.

## Data Policy

Downloaded images are not committed by default. Users can recreate candidate datasets with the provided scripts, but human review is required before running the analyses. Web data may contain noise, ambiguous labels, duplicates, watermarks, or licensing constraints.

The repository keeps scripts, metadata conventions, selected result CSVs, and selected figures. Local `raw/`, `reviewed/`, `metadata/`, and review-gallery files are ignored by default.

Precomputed summary reports are available in [results/model_matrix_summary/](results/model_matrix_summary/).

## Quick Start

Run the main dog/grass diagnostic with the default OpenCLIP ViT-B/32 model:

```powershell
python scripts/run_dog_grass_false_negation_analysis.py `
  --root data/context_neg/dog_grass_false_negation `
  --model openclip_vit_b32 `
  --output results/dog_grass_false_negation `
  --bootstrap-samples 1000 `
  --batch-size 8
```

## Reproduction

Install dependencies:

```powershell
python -m pip install -r requirements.txt
python -m pip install ddgs
```

Build or refresh the human sanity table:

```powershell
python scripts/build_human_sanity_table.py `
  --scenarios kitchen_table street_car cat_sofa person_beach bicycle_street `
  --output results/human_sanity_table.csv
```

Run the dog/grass diagnostic:

```powershell
python scripts/run_dog_grass_false_negation_analysis.py `
  --root data/context_neg/dog_grass_false_negation `
  --model openclip_vit_b32 `
  --output results/dog_grass_false_negation `
  --bootstrap-samples 1000 `
  --batch-size 8
```

Run the with/without scenario analysis:

```powershell
python scripts/run_final_contextneg_analysis.py `
  --scenarios kitchen_table street_car cat_sofa person_beach bicycle_street `
  --model openclip_vit_b32 `
  --output results/final_contextneg_analysis `
  --bootstrap-samples 1000 `
  --batch-size 8
```

Run the model matrix:

```powershell
python scripts/run_model_matrix.py `
  --models openclip_vit_b32 openclip_vit_b32_openai openclip_vit_b32_datacomp openclip_rn50 openclip_vit_b16_openai openclip_vit_b16_siglip `
  --analyses final dog_grass `
  --scenarios kitchen_table street_car cat_sofa person_beach bicycle_street `
  --output-root results/model_matrix `
  --bootstrap-samples 1000 `
  --batch-size 8
```

Aggregate the model matrix:

```powershell
python scripts/aggregate_model_matrix.py `
  --root results/model_matrix `
  --output results/model_matrix_summary
```

Run tests:

```powershell
python -m compileall src scripts
python -m pytest -q
```

More collection and run commands are in [docs/runbook.md](docs/runbook.md).

## Repository Map

```text
README.md
docs/
  methodology.md
  metrics.md
  results_summary.md
  limitations.md
  appendix_logical_connectors.md
  appendix_negation_delta.md
scripts/
src/
tests/
results/
  model_matrix_summary/
  dog_grass_false_negation/
  final_contextneg_analysis/
  negation_delta_consistency/
  logical_connector_embeddings/
  selected_figures/
```
