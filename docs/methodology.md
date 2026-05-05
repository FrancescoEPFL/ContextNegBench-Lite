# Methodology

ContextNegBench-Lite is a diagnostic analysis of CLIP-style image-text embeddings. It does not train or fine-tune models.

## Dataset Creation

The project uses small, manually reviewed image folders under `data/context_neg/`.

Candidate images can be collected with:

```powershell
python scripts/search_download_context_neg_images.py
```

or with the batch enrichment helper:

```powershell
python scripts/enrich_context_neg_datasets.py
```

The search/download scripts are only collection helpers. They do not assign final labels.

## Human Review

Human review is required. Images are manually placed into:

- `reviewed/with_<object>/`
- `reviewed/without_<object>/`
- `reviewed/rejected/`

Accept `with_object` only when the scene and object are both clearly visible. Accept `without_object` only when the scene is clear and the target object is not visible. Reject ambiguous, duplicate, heavily watermarked, cropped, or low-quality images.

No CLIP, YOLO, or other model is used to assign dataset labels.

## Datasets

Main dog/grass diagnostic:

- `dog_grass_false_negation`: 74 reviewed images with a visible dog on grass.

With/without scenarios:

- `kitchen_table`: 193 with table, 152 without table.
- `street_car`: 170 with cars, 142 without cars.
- `cat_sofa`: 37 with cat, 35 without cat.
- `person_beach`: 30 with person, 35 without person.
- `bicycle_street`: 36 with bicycle, 33 without bicycle.

## Models

The model matrix includes:

- `openclip_vit_b32`
- `openclip_vit_b32_openai`
- `openclip_vit_b32_datacomp`
- `openclip_vit_b16_openai`
- `openclip_vit_b16_siglip`
- `openclip_rn50`

All models are evaluated with frozen embeddings. No training or adaptation is performed.

## Scoring

For each image, the model encodes the image and a set of candidate captions. Captions are compared with the image embedding using normalized embedding similarity.

For the dog/grass diagnostic, each image is scored against:

- a false negated caption, such as `an image with no dog`;
- a true partial generic caption, such as `an image of a grassy field`;
- a true detailed-generic caption, such as `an image of a grassy field with an animal`;
- a fully correct positive caption, such as `an image of a dog on a grassy field`.

For with/without scenarios, each image is scored against:

- a generic scene caption;
- a positive object-specific caption;
- a negated absence caption.

## Bootstrap Confidence Intervals

Reported confidence intervals are bootstrap percentile intervals over images. The default run uses `--bootstrap-samples 1000` and seed `42`.

Confidence intervals are used for diagnostic uncertainty, not for claiming formal benchmark significance.

## Text-Only Analyses

Two text-only analyses support the interpretation:

- negation delta consistency: tests whether `embedding("no X") - embedding("X")` is aligned across objects;
- logical connector embeddings: compares templates such as `and`, `or`, `but not`, `without`, `only`, and `neither nor`.

These analyses are appendices. The main evidence comes from image-text scoring.
