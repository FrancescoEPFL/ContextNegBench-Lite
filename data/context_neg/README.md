# ContextNeg Data Layout

Each real-image diagnostic scenario uses a local folder under `data/context_neg/`.

Two-condition scenarios:

```text
data/context_neg/kitchen_table/
  reviewed/
    with_table/
    without_table/
    rejected/
  annotations.jsonl

data/context_neg/street_car/
  reviewed/
    with_car/
    without_car/
    rejected/
  annotations.jsonl
```

One-condition dog/grass diagnostic:

```text
data/context_neg/dog_grass_false_negation/
  reviewed/
    with_dog/
    rejected/
```

Images are intentionally not committed. Recreate local datasets with the downloader/prepare scripts, then manually review images before building annotations or running analysis.

