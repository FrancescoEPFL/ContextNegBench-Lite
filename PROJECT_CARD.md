# Project Card

## One-Line Summary

ContextNegBench-Lite is a small diagnostic evaluation artifact for CLIP-style image-text scoring failures involving negation and object specificity.

## What I Built

- A Python evaluation pipeline for frozen CLIP-style image/text embeddings.
- A dog/grass diagnostic for false object-specific negation versus true underspecified captions.
- Supporting with/without scenario analyses.
- A multi-model result matrix across OpenCLIP-style presets.
- Bootstrap summaries, schema validation, result fingerprints, selected figures, and CI.
- A license-safe synthetic demo dataset for quick pipeline checks.

## What The Project Demonstrates

- Multimodal evaluation design.
- Careful claim framing and limitation handling.
- Research-style result aggregation.
- Python package organization with scripts, docs, tests, and CI.
- Practical reproducibility decisions when full data cannot be redistributed.

## Main Claim

```text
false object-specific negation can beat true underspecified descriptions
```

This does not mean false captions generally beat true captions. The fully correct positive caption usually remains top-ranked.

##  Bullet

Built `ContextNegBench-Lite`, a lightweight diagnostic framework for CLIP-style vision-language models, testing when false object-specific negation captions outrank true but underspecified descriptions across multiple OpenCLIP/SigLIP-style presets.

## More Explanation

I avoided a broad claim like "CLIP does not understand negation" and designed a narrower diagnostic. For a dog-on-grass image, I tested whether a false caption like `an image with no dog` can score above a true but generic caption like `an image of a grassy field`. The result was often yes, but the fully correct caption usually remained top-ranked. So the failure mode is about object specificity overpowering weak absence semantics, not false captions generally beating true captions.
