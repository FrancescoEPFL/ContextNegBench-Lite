from __future__ import annotations

import re

from PIL import Image

from negcompbench.models.base import CaptionRanker

TOKEN_RE = re.compile(r"[a-z]+")


class TextOnlyBowBaseline(CaptionRanker):
    """A diagnostic baseline that rewards captions containing known dataset words.

    It does not inspect the image. If it performs well, the caption set has a
    lexical artifact rather than requiring visual reasoning.
    """

    def __init__(self) -> None:
        self.name = "text_bow_baseline"
        self.vocab_weights = {
            "red": 1.0,
            "blue": 1.0,
            "green": 1.0,
            "yellow": 1.0,
            "purple": 1.0,
            "orange": 1.0,
            "square": 1.0,
            "circle": 1.0,
            "triangle": 1.0,
            "star": 1.0,
            "left": 0.4,
            "right": 0.4,
            "above": 0.4,
            "below": 0.4,
            "without": 0.4,
            "with": 0.2,
            "border": 0.3,
            "two": 0.2,
            "three": 0.2,
            "four": 0.2,
        }

    def score_batch(self, images: list[Image.Image], caption_sets: list[list[str]]) -> list[list[float]]:
        del images
        return [[self.score_caption(caption) for caption in captions] for captions in caption_sets]

    def score_caption(self, caption: str) -> float:
        tokens = TOKEN_RE.findall(caption.lower())
        return sum(self.vocab_weights.get(token, 0.0) for token in tokens) / max(len(tokens), 1)
