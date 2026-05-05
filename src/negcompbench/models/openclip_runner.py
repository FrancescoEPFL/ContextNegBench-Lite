from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from negcompbench.models.base import CaptionRanker
from negcompbench.utils.device import get_device


@dataclass(frozen=True)
class OpenCLIPConfig:
    model_name: str = "ViT-B-32"
    pretrained: str = "laion2b_s34b_b79k"
    device: str = "auto"


class OpenCLIPRanker(CaptionRanker):
    def __init__(self, config: OpenCLIPConfig) -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise ImportError("OpenCLIP support requires open_clip_torch. Install with `pip install -r requirements.txt`.") from exc

        self.open_clip = open_clip
        self.torch = torch
        self.device = get_device(config.device)
        self.name = f"openclip_{config.model_name}_{config.pretrained}".replace("/", "_").replace("-", "_").lower()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.model_name,
            pretrained=config.pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(config.model_name)
        self.model.eval()

    def score_batch(self, images: list[Image.Image], caption_sets: list[list[str]]) -> list[list[float]]:
        torch = self.torch
        flat_captions = [caption for captions in caption_sets for caption in captions]
        lengths = [len(captions) for captions in caption_sets]

        with torch.no_grad():
            image_tensor = torch.stack([self.preprocess(image.convert("RGB")) for image in images]).to(self.device)
            text_tensor = self.tokenizer(flat_captions).to(self.device)
            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            scores: list[list[float]] = []
            offset = 0
            for image_idx, length in enumerate(lengths):
                caption_features = text_features[offset : offset + length]
                sims = image_features[image_idx : image_idx + 1] @ caption_features.T
                scores.append(sims.squeeze(0).detach().cpu().numpy().astype(float).tolist())
                offset += length
        return scores

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            image_tensor = torch.stack([self.preprocess(image.convert("RGB")) for image in images]).to(self.device)
            image_features = self.model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features.detach().cpu().numpy().astype(float)

    def encode_texts(self, captions: list[str]) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            text_tensor = self.tokenizer(captions).to(self.device)
            text_features = self.model.encode_text(text_tensor)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features.detach().cpu().numpy().astype(float)


def get_peak_memory_mb(device: str) -> float | None:
    try:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024**2))
    except Exception:
        return None
    return None
