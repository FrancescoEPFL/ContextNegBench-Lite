# Results Summary

This file contains the fuller tables behind the README. The compact publication narrative is in `README.md`.

## Dataset Sanity

| scenario | n_annotations_with_object | n_annotations_without_object | has_annotations |
| --- | ---: | ---: | --- |
| kitchen_table | 193 | 152 | True |
| street_car | 170 | 142 | True |
| cat_sofa | 37 | 35 | True |
| person_beach | 30 | 35 | True |
| bicycle_street | 36 | 33 | True |

## Dog/Grass: Core And Field Prompt Results

| model | pair_id | false_negated_win_rate_over_detailed_generic | false_negated_win_rate_over_generic | false_negated_win_rate_over_positive | top_positive_rate |
| --- | --- | ---: | ---: | ---: | ---: |
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

## Dog/Grass: Model-Level Average

Average over the `core` and `field` dog/grass prompt pairs.

| model | false_negated_win_rate_over_detailed_generic | false_negated_win_rate_over_generic | top_positive_rate |
| --- | ---: | ---: | ---: |
| openclip_rn50 | 0.662 | 0.919 | 0.919 |
| openclip_vit_b16_openai | 0.676 | 0.872 | 0.919 |
| openclip_vit_b16_siglip | 0.547 | 0.980 | 0.932 |
| openclip_vit_b32 | 0.655 | 0.980 | 0.939 |
| openclip_vit_b32_datacomp | 0.588 | 0.872 | 0.818 |
| openclip_vit_b32_openai | 0.547 | 0.831 | 0.932 |

## With/Without Scenarios: Scenario-Level Average

Average over models for base prompt results.

| scenario | false_negative_top_rate | mean_false_absence_tolerance | positive_top_rate |
| --- | ---: | ---: | ---: |
| bicycle_street | 0.500 | 0.055 | 0.491 |
| cat_sofa | 0.275 | 0.070 | 0.721 |
| kitchen_table | 0.260 | 0.024 | 0.721 |
| person_beach | 0.089 | 0.011 | 0.867 |
| street_car | 0.210 | 0.016 | 0.749 |

## Image Condition Separation

Image condition separation is `1 - cosine(with_object_centroid, without_object_centroid)`.

| scenario | openclip_rn50 | openclip_vit_b16_openai | openclip_vit_b16_siglip | openclip_vit_b32 | openclip_vit_b32_datacomp | openclip_vit_b32_openai |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| bicycle_street | 0.138 | 0.096 | 0.121 | 0.233 | 0.248 | 0.118 |
| cat_sofa | 0.259 | 0.189 | 0.151 | 0.320 | 0.417 | 0.183 |
| kitchen_table | 0.028 | 0.019 | 0.041 | 0.071 | 0.130 | 0.024 |
| person_beach | 0.083 | 0.065 | 0.141 | 0.171 | 0.233 | 0.067 |
| street_car | 0.094 | 0.053 | 0.088 | 0.155 | 0.201 | 0.063 |

## Interpretation

The dog/grass diagnostic is the most robust result: false object-specific negation often beats true generic partial captions across models. The full positive caption usually remains top-ranked.

The detailed-generic control weakens the effect, which shows that underspecification is central. The with/without scenarios are more heterogeneous and should be interpreted as supporting diagnostics rather than the main claim.

Full CSV outputs:

- `results/model_matrix_summary/dog_grass_by_model.csv`
- `results/model_matrix_summary/final_contextneg_by_model.csv`
- `results/human_sanity_table.csv`
