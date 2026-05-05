# Lexical Bias Baselines

The main diagnostic is motivated by a lexical asymmetry:

- false negated caption: mentions the visible object, for example `dog`;
- true generic caption: describes the scene, but omits the visible object;
- positive caption: mentions both the scene and the visible object.

Two lightweight lexical baselines are useful when interpreting failures:

| baseline | definition | interpretation |
| --- | --- | --- |
| caption length | number of word tokens in each candidate caption | checks whether longer captions are systematically favored |
| object-word overlap | whether the candidate caption contains the target object word | checks whether naming the visible object predicts higher score |

For the dog/grass setup, object-word overlap is the important baseline: the false negated caption and the positive caption contain `dog`; the generic caption usually does not. This does not prove CLIP is bag-of-words. It quantifies the lexical advantage that the diagnostic is designed around.

Use:

```powershell
python scripts/analyze_lexical_bias_baselines.py
```
