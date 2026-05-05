# Prompt Sensitivity

The dog/grass diagnostic intentionally uses multiple prompt forms instead of relying on one wording.

| pair_id | false negated | true generic | detailed generic | positive |
| --- | --- | --- | --- | --- |
| core | `an image with no dog` | `an image of a grassy field` | `an image of a grassy field with an animal` | `an image of a dog on a grassy field` |
| photo | `a photo with no dog` | `a photo of a grassy field` | `a photo of a grassy field with an animal` | `an image of a dog on a grassy field` |
| visible | `no dog is visible in this image` | `a grassy field` | `a grassy field with a visible animal` | `a grassy field with a visible dog` |
| without | `an image without a dog` | `an image of grass` | `an image of grass with an animal` | `a grassy field with a dog` |
| field | `a grassy field with no dog` | `a grassy field` | `a grassy field with an animal` | `a grassy field with a dog` |

The README reports the two clearest pairs, `core` and `field`, because they are easiest to explain. The full model-matrix CSV keeps all prompt-pair rows, so prompt wording can be inspected without rerunning the models:

```powershell
python scripts/reproduce_paper_tables.py --full
```

This project should not be read as a single-prompt benchmark. Prompt wording is part of the behavior being diagnosed.
