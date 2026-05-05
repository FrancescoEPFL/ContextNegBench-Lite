# Limitations

ContextNegBench-Lite is a diagnostic project, not a universal benchmark for VLM negation.

## Dataset Size

The dog/grass diagnostic has 74 reviewed images. The original with/without datasets are larger, but the three extension datasets are small:

- `cat_sofa`: 37 with object, 35 without object.
- `person_beach`: 30 with object, 35 without object.
- `bicycle_street`: 36 with object, 33 without object.

These extension datasets are useful low-compute diagnostics, not conclusive benchmark splits.

## Web Image Noise

Images are web-collected and manually reviewed. Manual review reduces obvious errors, but cannot remove all ambiguity, duplicates, stock-photo artifacts, watermarks, unusual crops, or hidden dataset biases.

## Data Rights

Downloaded images are not committed by default. Users should verify image rights before redistributing any image assets.

## Prompt Sensitivity

The results depend on wording. For example, `an image with no dog`, `a grassy field with no dog`, `no visible dog`, and `without a dog` can behave differently.

## Model Scope

The model matrix includes several OpenCLIP/SigLIP-style presets, but not every CLIP-like model or modern VLM. Results should not be generalized to all multimodal systems.

## Language Scope

The analysis uses English prompts. Negation and connector behavior may differ across languages and tokenizers.

## Diagnostic, Not Causal

The project measures embedding geometry and image-text scores. It does not inspect training data, attention mechanisms, internal circuits, or causal model behavior.

## Logical Claims

The results are consistent with weakly grounded absence semantics in some settings, but they do not prove that CLIP lacks logical structure. Text-only analyses show partially structured operator directions.

## Dog/Grass Scope

The dog/grass experiment is a one-condition stress test with dog-visible images. It is not a paired with/without dog benchmark.
