from __future__ import annotations

import random

from PIL import Image

from negcompbench.models.base import CaptionRanker


class RandomBaseline(CaptionRanker):
    def __init__(self, seed: int = 0) -> None:
        self.name = "random_baseline"
        self.rng = random.Random(seed)

    def score_batch(self, images: list[Image.Image], caption_sets: list[list[str]]) -> list[list[float]]:
        return [[self.rng.random() for _ in captions] for captions in caption_sets]
