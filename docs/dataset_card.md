# Dataset Card

ContextNegBench-Lite uses small local image sets collected from the web and manually reviewed into condition folders. The repository should not commit downloaded images unless their rights are explicitly verified.

## Current Datasets

| scenario | scene | object | conditions |
| --- | --- | --- | --- |
| `kitchen_table` | kitchen | table | with table / without table |
| `street_car` | street | car | with cars / without cars |
| `dog_grass_false_negation` | grassy field | dog | with dog on grass only |

## Planned Extension Datasets

| scenario | scene | object | purpose |
| --- | --- | --- | --- |
| `cat_sofa` | living room | cat | object on common indoor support |
| `person_beach` | beach | person | salient human object vs empty scene |
| `bicycle_street` | street | bicycle | vehicle-like object smaller than cars |

## Human Review Rules

Labels are assigned by folder placement, not by CLIP, YOLO, or any other model.

Accept `with_object` only when:

- the scene is clear;
- the target object is clearly visible;
- the object is not merely implied;
- the image is not too blurry, dark, cropped, watermarked, or dominated by text.

Accept `without_object` only when:

- the scene is clear;
- the target object is not visible;
- absence is plausible and not just caused by an extreme crop;
- no similar object is likely to be confused for the target.

Reject ambiguous or duplicate images.

## Public Release Note

The public GitHub repo should include metadata, scripts, selected result CSVs, and selected figures. Downloaded images should remain ignored by default.
