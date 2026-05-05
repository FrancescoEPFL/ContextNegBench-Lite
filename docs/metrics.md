# Metrics

This file defines the metrics used in the project. The README intentionally includes only the most important ones.

## Dog/Grass Metrics

### `false_negated_win_rate_over_generic`

Rate at which the false negated caption scores above the true generic caption.

```text
mean(score(false_negated) > score(partial_generic))
```

Example:

```text
score("an image with no dog") > score("an image of a grassy field")
```

### `false_negated_win_rate_over_detailed_generic`

Rate at which the false negated caption scores above a stronger true generic caption.

```text
mean(score(false_negated) > score(true_detailed_generic))
```

Example:

```text
score("an image with no dog") > score("an image of a grassy field with an animal")
```

### `false_negated_win_rate_over_positive`

Rate at which the false negated caption scores above the fully correct positive caption.

```text
mean(score(false_negated) > score(positive_reference))
```

### `top_positive_rate`

Rate at which the fully correct positive caption is top-ranked among the tested captions.

```text
mean(top_caption == positive_reference)
```

### `mean_margin_false_vs_generic`

Average score margin between the false negated caption and the true generic caption.

```text
mean(score(false_negated) - score(partial_generic))
```

### `mean_margin_false_vs_detailed_generic`

Average score margin between the false negated caption and the true detailed-generic caption.

```text
mean(score(false_negated) - score(true_detailed_generic))
```

## With/Without Scenario Metrics

### `positive_specificity_gain`

For with-object images:

```text
score(scene with object) - score(scene)
```

This measures how much the model rewards naming the visible object.

### `mean_false_absence_tolerance`

For with-object images:

```text
score(scene without object) - score(scene)
```

Higher values mean the false absence caption remains compatible with an image that contains the object.

### `false_negative_top_rate`

For with-object images:

```text
mean(top_caption == false_negated_caption)
```

This is stricter than false absence tolerance because it asks whether the false caption is top-ranked.

### `image_condition_separation`

Distance between with-object and without-object image centroids in image embedding space.

```text
1 - cosine(centroid_with_object, centroid_without_object)
```

This helps diagnose whether the image-side distinction is visually separable to the model.

## Text-Only Metrics

### `text_negation_distance`

Distance between an affirmative caption and its negated counterpart in text embedding space.

```text
1 - cosine(positive_caption, negative_caption)
```

### `delta_direction_similarity`

Pairwise cosine similarity between normalized operator deltas across objects.

```text
delta_X = normalize(embedding(operator X) - embedding(X))
delta_direction_similarity = mean cosine(delta_X, delta_Y)
```

Example operator delta:

```text
embedding("no dog") - embedding("dog")
```

High values suggest the operator transformation is partially structured across objects.

## Secondary Diagnostics

The code may also emit:

- median margins;
- low-margin rates;
- PCA explained variance;
- nearest-neighbor categories;
- object dominance indices;
- template-specific prompt variants.

These are useful for appendix analysis but should not drive the main README claim.
