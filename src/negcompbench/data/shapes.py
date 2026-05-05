from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from negcompbench.data.schemas import ObjectSpec

COLOR_RGB = {
    "red": (220, 52, 52),
    "blue": (55, 105, 220),
    "green": (55, 165, 80),
    "yellow": (238, 205, 55),
    "black": (20, 20, 20),
    "white": (245, 245, 245),
    "purple": (145, 80, 200),
    "orange": (235, 140, 45),
}


def draw_scene(
    objects: list[ObjectSpec],
    output_path: str | Path,
    image_size: int = 224,
    background: str = "white",
    noise: str = "clean",
    seed: int = 0,
) -> None:
    rng = random.Random(seed)
    bg = COLOR_RGB.get(background, (245, 245, 245))
    image = Image.new("RGB", (image_size, image_size), bg)
    draw = ImageDraw.Draw(image)
    for obj in objects:
        draw_object(draw, obj)

    if noise == "mild_blur":
        image = image.filter(ImageFilter.GaussianBlur(radius=0.8))
    elif noise == "gaussian_noise":
        image = add_gaussian_noise(image, sigma=7.0, seed=seed)
    elif noise == "positional_jitter":
        jittered = []
        for obj in objects:
            jittered.append(
                ObjectSpec(
                    shape=obj.shape,
                    color=obj.color,
                    x=obj.x + rng.randint(-5, 5),
                    y=obj.y + rng.randint(-5, 5),
                    size=obj.size,
                    has_border=obj.has_border,
                    border_color=obj.border_color,
                )
            )
        image = Image.new("RGB", (image_size, image_size), bg)
        draw = ImageDraw.Draw(image)
        for obj in jittered:
            draw_object(draw, obj)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def draw_object(draw: ImageDraw.ImageDraw, obj: ObjectSpec) -> None:
    fill = COLOR_RGB[obj.color]
    outline = COLOR_RGB.get(obj.border_color or "black") if obj.has_border else None
    width = 4 if obj.has_border else 1
    half = obj.size // 2
    bbox = [obj.x - half, obj.y - half, obj.x + half, obj.y + half]

    if obj.shape == "square":
        draw.rectangle(bbox, fill=fill, outline=outline, width=width)
    elif obj.shape == "circle":
        draw.ellipse(bbox, fill=fill, outline=outline, width=width)
    elif obj.shape == "triangle":
        points = [(obj.x, obj.y - half), (obj.x - half, obj.y + half), (obj.x + half, obj.y + half)]
        draw.polygon(points, fill=fill, outline=outline)
        if obj.has_border:
            draw.line(points + [points[0]], fill=outline, width=width, joint="curve")
    elif obj.shape == "star":
        points = star_points(obj.x, obj.y, half, max(half // 2, 1))
        draw.polygon(points, fill=fill, outline=outline)
        if obj.has_border:
            draw.line(points + [points[0]], fill=outline, width=width, joint="curve")
    else:
        raise ValueError(f"Unknown shape: {obj.shape}")


def star_points(cx: int, cy: int, outer: int, inner: int) -> list[tuple[int, int]]:
    points = []
    for idx in range(10):
        angle = -math.pi / 2 + idx * math.pi / 5
        radius = outer if idx % 2 == 0 else inner
        points.append((int(cx + radius * math.cos(angle)), int(cy + radius * math.sin(angle))))
    return points


def add_gaussian_noise(image: Image.Image, sigma: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.asarray(image).astype(np.float32)
    arr = np.clip(arr + rng.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")
