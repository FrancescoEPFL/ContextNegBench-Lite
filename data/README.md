# Data

This repository does not include downloaded web images by default.

ContextNegBench-Lite uses small, manually reviewed local image folders. The scripts can download or organize candidate images, but the user must review them before running analysis. The project intentionally avoids using CLIP, YOLO, or other AI models to assign dataset labels.

Expected local scenarios:

```text
data/context_neg/kitchen_table/
data/context_neg/street_car/
data/context_neg/dog_grass_false_negation/
data/context_neg/cat_sofa/
data/context_neg/person_beach/
data/context_neg/bicycle_street/
```

Reviewed images, raw downloaded images, metadata logs, annotations, and review galleries are ignored by git. This keeps the public repository lightweight and avoids redistributing images without verified rights.
