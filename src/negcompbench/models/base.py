from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image


class CaptionRanker(ABC):
    name: str

    @abstractmethod
    def score_batch(self, images: list[Image.Image], caption_sets: list[list[str]]) -> list[list[float]]:
        """Return one score per caption for each image."""
