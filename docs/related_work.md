# Related Work

ContextNegBench-Lite sits between CLIP compositionality diagnostics and recent negation-focused VLM benchmarks.

## CLIP and OpenCLIP

- Radford et al., 2021, *Learning Transferable Visual Models From Natural Language Supervision*.
- Cherti et al., 2023, *Reproducible Scaling Laws for Contrastive Language-Image Learning*.

## Compositionality Benchmarks

- Winoground probes visio-linguistic compositionality with paired image/caption examples.
- VL-CheckList evaluates objects, attributes, and relations.
- ARO studies attribute, relation, and order sensitivity and is especially relevant to bag-of-words-like behavior.
- SugarCrepe shows that many compositionality benchmarks are hackable and need careful hard-negative design.

## Negation and Text-Encoder Bottlenecks

- Negation-focused VLM work, including NegBench-style evaluations, directly motivates the absence/negation part of this project.
- Work on text encoders as compositional bottlenecks is relevant because this project separately analyzes text-space negation geometry and image-text scoring.

## Positioning

This project should not claim to replace these benchmarks. Its contribution is narrower:

```text
false object-specific negation vs true generic partial description
```

The goal is to provide a small reproducible diagnostic that exposes a concrete scoring behavior in CLIP-like embeddings.
