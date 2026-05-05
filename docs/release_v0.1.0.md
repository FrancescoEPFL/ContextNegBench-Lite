# ContextNegBench-Lite v0.1.0

Initial public release with frozen result summaries.

Included:

- research-style README and methodology docs
- public model-matrix summary CSVs
- selected figures and full research report
- CC0 synthetic demo dataset
- result schema validation
- one-command small/full reproduction helper
- smoke test, end-to-end test, ruff, mypy, pytest, and GitHub Actions CI

Not included:

- downloaded web images
- reviewed dog/grass and with/without scenario image folders
- local review galleries
- temporary experiment logs

The central frozen claim is:

```text
false object-specific negation can beat true underspecified descriptions
```

This release does not claim that CLIP never understands negation, that CLIP is purely bag-of-words, or that false captions generally beat true captions.
