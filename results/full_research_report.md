# ContextNegBench-Lite: Full Research Report

Generated after the extended dataset runs and the 6-model matrix.

## Executive Summary

This project does not show that CLIP-style models simply "do not understand negation." The stronger and more defensible finding is narrower:

```text
false object-specific negation can outrank true generic partial descriptions
```

Across six OpenCLIP/SigLIP-style presets, dog-on-grass images show a consistent pattern: false negated captions such as `an image with no dog` often beat true but generic captions such as `an image of a grassy field`. However, the fully correct positive caption usually remains top-ranked. This means the failure is not "falsehood beats truth" in general; it is a specificity failure where a false caption that names a visible object can beat a true caption that omits it.

The detailed-generic control is important. When the true caption is strengthened from `a grassy field` to `a grassy field with an animal`, the effect weakens sharply for some prompt forms and models. This narrows the final claim: CLIP-like matching is especially vulnerable when the true caption is underspecified.

The extended with/without scenarios show that the failure is not uniform across objects, scenes, or models. Visual separability, object salience, prompt wording, and model pretraining all matter. The new datasets are useful diagnostics, but smaller than the original kitchen/table and street/car sets; they should be treated as extensions, not final benchmark evidence.

## Models

The model matrix includes:

- `openclip_vit_b32`
- `openclip_vit_b32_openai`
- `openclip_vit_b32_datacomp`
- `openclip_vit_b16_openai`
- `openclip_vit_b16_siglip`
- `openclip_rn50`

All runs use frozen image-text embeddings and cosine-style image-text scores from the model runner. No model is trained or fine-tuned.

## Dataset Sanity

All five with/without scenarios have annotations. The three extension datasets are much smaller, so their results should be read as low-compute diagnostics.

| scenario | n_annotations_with_object | n_annotations_without_object | has_annotations |
| --- | --- | --- | --- |
| kitchen_table | 193 | 152 | True |
| street_car | 170 | 142 | True |
| cat_sofa | 37 | 35 | True |
| person_beach | 30 | 35 | True |
| bicycle_street | 36 | 33 | True |

The dog/grass diagnostic uses 74 reviewed images with a visible dog on grass. It is a one-condition stress test, not a full with/without benchmark.

## Headline Metrics

The final analysis focuses on a small set of metrics:

- `false_negated_win_rate_over_generic`: rate at which the false negated caption beats a true generic caption.
- `false_negated_win_rate_over_detailed_generic`: same comparison against a stronger true generic caption such as `with an animal`.
- `false_negated_win_rate_over_positive`: rate at which the false negated caption beats the full positive caption.
- `top_positive_rate`: rate at which the full positive caption is top-ranked.
- `mean_false_absence_tolerance`: for with-object images, `score(scene without object) - score(scene)`.
- `false_negative_top_rate`: for with-object images, rate at which the false negated caption is top-ranked.
- `image_condition_separation`: `1 - cosine(with_object_centroid, without_object_centroid)` in image embedding space.

Secondary text-only analyses such as delta consistency and logical connector embeddings are useful appendices, but they are not the main evidence for the final claim.

## Dog/Grass Diagnostic

### Core And Field Prompt Results

The dog/grass result is the cleanest and most memorable evidence in the project. Across all models, the false negated caption often beats the partial generic caption. The fully correct positive caption remains top in most cases.

| model | pair_id | false_negated_win_rate_over_detailed_generic | false_negated_win_rate_over_generic | false_negated_win_rate_over_positive | top_positive_rate |
| --- | --- | --- | --- | --- | --- |
| openclip_rn50 | core | 0.405 | 0.851 | 0.041 | 0.919 |
| openclip_rn50 | field | 0.919 | 0.986 | 0.054 | 0.919 |
| openclip_vit_b16_openai | core | 0.419 | 0.757 | 0.081 | 0.892 |
| openclip_vit_b16_openai | field | 0.932 | 0.986 | 0.054 | 0.946 |
| openclip_vit_b16_siglip | core | 0.676 | 0.973 | 0.081 | 0.905 |
| openclip_vit_b16_siglip | field | 0.419 | 0.986 | 0.014 | 0.959 |
| openclip_vit_b32 | core | 0.486 | 0.959 | 0.027 | 0.946 |
| openclip_vit_b32 | field | 0.824 | 1.000 | 0.041 | 0.932 |
| openclip_vit_b32_datacomp | core | 0.365 | 0.770 | 0.068 | 0.851 |
| openclip_vit_b32_datacomp | field | 0.811 | 0.973 | 0.135 | 0.784 |
| openclip_vit_b32_openai | core | 0.230 | 0.676 | 0.041 | 0.919 |
| openclip_vit_b32_openai | field | 0.865 | 0.986 | 0.041 | 0.946 |

### Model-Level Dog/Grass Average

Averaging the `core` and `field` prompt pairs:

| model | false_negated_win_rate_over_detailed_generic | false_negated_win_rate_over_generic | top_positive_rate |
| --- | --- | --- | --- |
| openclip_rn50 | 0.662 | 0.919 | 0.919 |
| openclip_vit_b16_openai | 0.676 | 0.872 | 0.919 |
| openclip_vit_b16_siglip | 0.547 | 0.980 | 0.932 |
| openclip_vit_b32 | 0.655 | 0.980 | 0.939 |
| openclip_vit_b32_datacomp | 0.588 | 0.872 | 0.818 |
| openclip_vit_b32_openai | 0.547 | 0.831 | 0.932 |

### Dog/Grass Interpretation

The dog/grass effect is robust against true but underspecified generic captions. It is weaker against true detailed-generic captions, and rarely beats the full positive caption. This supports the final claim:

```text
false object-specific negation can beat partial truth, but usually not full truth
```

The detailed-generic control prevents an overclaim. The model is not merely preferring false negation; it is often preferring the caption that contains the visible object concept or a close visual category.

## Extended With/Without Scenario Analysis

The with/without analysis asks whether a false absence caption remains compatible with with-object images. The table below shows base-prompt behavior.

| model | scenario | false_negative_top_rate | mean_false_absence_tolerance | positive_top_rate |
| --- | --- | --- | --- | --- |
| openclip_rn50 | bicycle_street | 0.750 | 0.049 | 0.250 |
| openclip_rn50 | cat_sofa | 0.054 | 0.035 | 0.946 |
| openclip_rn50 | kitchen_table | 0.005 | 0.008 | 0.984 |
| openclip_rn50 | person_beach | 0.100 | 0.005 | 0.900 |
| openclip_rn50 | street_car | 0.482 | 0.021 | 0.494 |
| openclip_vit_b16_openai | bicycle_street | 0.667 | 0.046 | 0.333 |
| openclip_vit_b16_openai | cat_sofa | 0.108 | 0.036 | 0.892 |
| openclip_vit_b16_openai | kitchen_table | 0.047 | 0.014 | 0.948 |
| openclip_vit_b16_openai | person_beach | 0.133 | 0.006 | 0.867 |
| openclip_vit_b16_openai | street_car | 0.241 | 0.017 | 0.741 |
| openclip_vit_b16_siglip | bicycle_street | 0.333 | 0.027 | 0.639 |
| openclip_vit_b16_siglip | cat_sofa | 0.514 | 0.064 | 0.486 |
| openclip_vit_b16_siglip | kitchen_table | 0.244 | 0.006 | 0.679 |
| openclip_vit_b16_siglip | person_beach | 0.100 | 0.012 | 0.800 |
| openclip_vit_b16_siglip | street_car | 0.047 | -0.008 | 0.876 |
| openclip_vit_b32 | bicycle_street | 0.250 | 0.067 | 0.750 |
| openclip_vit_b32 | cat_sofa | 0.243 | 0.093 | 0.757 |
| openclip_vit_b32 | kitchen_table | 0.767 | 0.035 | 0.223 |
| openclip_vit_b32 | person_beach | 0.000 | 0.009 | 0.900 |
| openclip_vit_b32 | street_car | 0.018 | 0.014 | 0.959 |
| openclip_vit_b32_datacomp | bicycle_street | 0.278 | 0.095 | 0.694 |
| openclip_vit_b32_datacomp | cat_sofa | 0.324 | 0.163 | 0.649 |
| openclip_vit_b32_datacomp | kitchen_table | 0.497 | 0.069 | 0.492 |
| openclip_vit_b32_datacomp | person_beach | 0.167 | 0.029 | 0.767 |
| openclip_vit_b32_datacomp | street_car | 0.224 | 0.044 | 0.706 |
| openclip_vit_b32_openai | bicycle_street | 0.722 | 0.046 | 0.278 |
| openclip_vit_b32_openai | cat_sofa | 0.405 | 0.032 | 0.595 |
| openclip_vit_b32_openai | kitchen_table | 0.000 | 0.010 | 1.000 |
| openclip_vit_b32_openai | person_beach | 0.033 | 0.005 | 0.967 |
| openclip_vit_b32_openai | street_car | 0.247 | 0.010 | 0.718 |

### Scenario-Level Average Across Models

| scenario | false_negative_top_rate | mean_false_absence_tolerance | positive_top_rate |
| --- | --- | --- | --- |
| bicycle_street | 0.500 | 0.055 | 0.491 |
| cat_sofa | 0.275 | 0.070 | 0.721 |
| kitchen_table | 0.260 | 0.024 | 0.721 |
| person_beach | 0.089 | 0.011 | 0.867 |
| street_car | 0.210 | 0.016 | 0.749 |

### Scenario Interpretation

The with/without results are heterogeneous:

- `person_beach` is the cleanest scenario on average: low false-negative top rate and high positive top rate.
- `bicycle_street` is the hardest extension: the false negated caption wins too often on with-bicycle images for several models.
- `cat_sofa` shows high false absence tolerance, especially for `openclip_vit_b32_datacomp` and `openclip_vit_b16_siglip`.
- `kitchen_table` remains highly model-dependent: `openclip_vit_b32` performs poorly on base wording, while `openclip_vit_b32_openai` and `openclip_rn50` perform very well.
- `street_car` is no longer uniformly clean across all models; the original OpenCLIP ViT-B/32 result was strong, but other presets vary.

This is actually useful: it shows that the failure mode is not a fixed property of a single architecture. Prompt wording, visual separability, and pretraining recipe all affect behavior.

## Image Condition Separation

Image condition separation measures how far the with-object and without-object image centroids are in each model's image embedding space.

| scenario | openclip_rn50 | openclip_vit_b16_openai | openclip_vit_b16_siglip | openclip_vit_b32 | openclip_vit_b32_datacomp | openclip_vit_b32_openai |
| --- | --- | --- | --- | --- | --- | --- |
| bicycle_street | 0.138 | 0.096 | 0.121 | 0.233 | 0.248 | 0.118 |
| cat_sofa | 0.259 | 0.189 | 0.151 | 0.320 | 0.417 | 0.183 |
| kitchen_table | 0.028 | 0.019 | 0.041 | 0.071 | 0.130 | 0.024 |
| person_beach | 0.083 | 0.065 | 0.141 | 0.171 | 0.233 | 0.067 |
| street_car | 0.094 | 0.053 | 0.088 | 0.155 | 0.201 | 0.063 |

The separation values support a cautious interpretation: image-side separability varies substantially by scenario and model. However, higher separation does not automatically imply better logical behavior. For example, `cat_sofa` often has high image separation but can still show high false absence tolerance. This suggests that separability is relevant but not sufficient.

## Text-Side Negation And Connector Geometry

The text-only analyses support a secondary point: negation is not pure noise in CLIP text space. Operator deltas such as:

```text
embedding("no dog") - embedding("dog")
```

show partial alignment across objects. Logical connector templates also show structured but fuzzy behavior. This supports a nuanced reading:

```text
CLIP-style text encoders encode some operator-like directions, but image-text matching does not enforce them as strong grounded constraints.
```

These analyses should remain appendix material. They are useful for interpretation, but the strongest empirical evidence comes from the image-text diagnostics above.

## Final Claims To Use

Recommended claims:

1. False object-specific negation can outrank true generic partial descriptions in CLIP-style image-text scoring.
2. The effect is robust across several OpenCLIP/SigLIP-style presets for dog/grass diagnostics.
3. The effect weakens when the true caption is made more detailed, showing that caption specificity is central.
4. Fully correct positive captions usually remain top-ranked, so this is not a general "false beats true" result.
5. With/without scenario behavior varies strongly by object, scene, model, and prompt wording.
6. Text-space negation directions are partially structured, but they do not reliably behave like hard logical constraints during image-text matching.

## Claims To Avoid

Avoid:

- "CLIP does not understand negation."
- "CLIP is bag-of-words."
- "Logical connectors are not grounded" as a universal statement.
- "This benchmark proves VLMs fail at logic."
- "The new extension datasets are conclusive."

Better:

```text
These diagnostics identify a repeatable scoring failure mode where salient object mentions can overpower weakly grounded absence semantics, especially when the true alternative caption is generic or underspecified.
```

## Limitations

- The image sets are small, especially the three extension datasets.
- Web-collected images may contain hidden biases, duplicates, stock-photo artifacts, or subtle label ambiguity.
- The dog/grass diagnostic has only with-dog images; it is not a paired absence benchmark.
- The prompts are English only.
- Some prompts are diagnostic rather than natural.
- Model comparisons are robustness checks, not a leaderboard.
- No training-data inspection is performed.
- The analysis measures embeddings and scores, not causal mechanisms inside the model.

## Publication Recommendation

For GitHub, use the dog/grass result as the lead finding and present the with/without model matrix as supporting evidence. For a workshop-style paper, the cleanest title would be:

```text
When False Negation Beats Partial Truth: Object Specificity in CLIP Image-Text Matching
```

The central abstract claim should be:

```text
We show that false negated captions containing a salient visible object can score above true but underspecified captions across multiple CLIP-style models, while full positive captions usually remain preferred.
```

## Files For Audit

- Full model matrix summary: `results/model_matrix_summary/summary.md`
- Final scenario CSV: `results/model_matrix_summary/final_contextneg_by_model.csv`
- Dog/grass model CSV: `results/model_matrix_summary/dog_grass_by_model.csv`
- Extended single-model report: `results/final_contextneg_analysis/final_report.md`
- Dog/grass report: `results/dog_grass_false_negation/report.md`
- Human sanity table: `results/human_sanity_table.csv`
- Metrics definitions: `docs/metrics.md`
- Methodology: `docs/methodology.md`
- Limitations: `docs/limitations.md`
