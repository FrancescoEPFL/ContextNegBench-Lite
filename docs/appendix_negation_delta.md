# Appendix: Negation Delta Consistency

This appendix tests whether negation behaves like a partially stable text-space transformation.

The basic diagnostic is:

```text
embedding("no dog") - embedding("dog")
~= embedding("no cat") - embedding("cat")
~= embedding("no car") - embedding("car")
```

## Main Results

| pair_group | mean_delta_direction_similarity | median_delta_direction_similarity | pc1_explained_variance |
| --- | ---: | ---: | ---: |
| bare_no | 0.6074 | 0.6084 | 0.2770 |
| article_no | 0.7392 | 0.7429 | 0.1775 |
| image_with_no | 0.6178 | 0.6228 | 0.1435 |
| photo_with_no | 0.6156 | 0.6123 | 0.1534 |
| visible_no_visible | 0.8330 | 0.8351 | 0.2116 |

## Baseline Comparison

Real negation deltas are much more aligned than object-object deltas and mismatched negation deltas.

| pair_group | real_negation_delta | object_object_delta | mismatched_no_delta |
| --- | ---: | ---: | ---: |
| bare_no | 0.6074 | -0.0409 | 0.2999 |
| article_no | 0.7392 | -0.0430 | 0.3065 |
| image_with_no | 0.6178 | -0.0511 | 0.2075 |
| photo_with_no | 0.6156 | -0.0306 | 0.2310 |
| visible_no_visible | 0.8330 | -0.0111 | 0.2728 |

## Interpretation

Negation is not pure noise in text embedding space. Several negation templates produce partially aligned directions across objects. `no visible X` is the most consistent template tested here.

However, this text-space structure does not guarantee logical grounding in image-text matching. The dog/grass diagnostic shows that a false negated caption can still score above a true underspecified caption.

Full outputs are in `results/negation_delta_consistency/`.
