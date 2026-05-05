# Appendix: Logical Connector Embeddings

This appendix summarizes the text-only logical connector analysis. It should be read as interpretive support, not as the main evidence for the project.

## Unary Operators

| operator | mean_delta_direction_similarity | median_delta_direction_similarity | pc1_explained_variance |
| --- | ---: | ---: | ---: |
| no | 0.7392 | 0.7429 | 0.1775 |
| without | 0.7692 | 0.7663 | 0.1520 |
| with_no | 0.6833 | 0.6821 | 0.1379 |
| without_any | 0.7967 | 0.8003 | 0.1878 |
| no_visible | 0.8330 | 0.8351 | 0.2116 |
| absent | 0.7932 | 0.8003 | 0.1794 |
| not_a | 0.8149 | 0.8187 | 0.1560 |

## Binary Connectors

The text-only connector analysis found that many logical connectors remain close to simpler object combinations. For example, `or` is close to `and`, and exclusion-like templates such as `but not` and `without` can remain close to conjunction-like phrases.

Selected cosine similarities:

| connector | reference | cosine_similarity |
| --- | --- | ---: |
| or | and | 0.9198 |
| but_not | and | 0.8705 |
| without | and | 0.8658 |
| neither_nor | and | 0.7757 |
| only | only_object1 | 1.0000 |

## Object Dominance

| operator | object_dominance_index |
| --- | ---: |
| no | 0.0653 |
| without | 0.0515 |
| with_no | 0.0079 |
| without_any | 0.0088 |
| no_visible | -0.0038 |
| absent | 0.0563 |
| not_a | 0.1013 |

Positive object dominance means object identity remains closer to its own positive object phrase than to other phrases with the same operator. Near-zero or negative values suggest stronger operator clustering.

## Interpretation

These results do not imply formal logic. They suggest that connector templates have partially structured but fuzzy text-space behavior. This helps explain why image-text scores can be dominated by object identity and scene priors even when text embeddings contain some operator-like structure.

Full outputs are in `results/logical_connector_embeddings/`.
