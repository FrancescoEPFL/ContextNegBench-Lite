from __future__ import annotations

import random
from pathlib import Path

from tqdm import tqdm

from negcompbench.data.captions import (
    attribute_binding_caption,
    object_phrase,
    prompt_variants,
    spatial_caption,
)
from negcompbench.data.schemas import Annotation, ObjectSpec
from negcompbench.data.shapes import draw_scene
from negcompbench.utils.io import ensure_dir, write_jsonl
from negcompbench.utils.seed import seed_everything

DEFAULT_TASKS = [
    "attribute_binding",
    "spatial_left_right",
    "spatial_above_below",
    "counting",
    "negation",
]


def generate_dataset(config: dict, output_dir: str | Path) -> list[Annotation]:
    seed = int(config.get("seed", 13))
    seed_everything(seed)
    rng = random.Random(seed)

    output = ensure_dir(output_dir)
    image_dir = ensure_dir(output / "images")
    annotations_path = output / "annotations.jsonl"

    image_size = int(config.get("image_size", 224))
    samples_per_task = int(config.get("samples_per_task", 200))
    tasks = list(config.get("tasks", DEFAULT_TASKS))
    colors = list(config.get("colors", ["red", "blue", "green", "yellow", "purple", "orange"]))
    shapes = list(config.get("shapes", ["square", "circle", "triangle", "star"]))
    background = str(config.get("background", "white"))
    noise = str(config.get("noise", "clean"))
    object_size = int(config.get("object_size", 56))

    annotations: list[Annotation] = []
    sample_idx = 0
    for task in tasks:
        for _ in tqdm(range(samples_per_task), desc=f"generate:{task}"):
            sample_seed = seed + sample_idx * 9973
            local_rng = random.Random(sample_seed)
            image_id = f"{task}_{sample_idx:06d}"
            image_path = image_dir / f"{image_id}.png"

            if task == "attribute_binding":
                ann = make_attribute_binding(image_id, image_path, sample_seed, local_rng, colors, shapes, image_size, object_size)
            elif task == "spatial_left_right":
                ann = make_spatial_left_right(image_id, image_path, sample_seed, local_rng, colors, shapes, image_size, object_size)
            elif task == "spatial_above_below":
                ann = make_spatial_above_below(image_id, image_path, sample_seed, local_rng, colors, shapes, image_size, object_size)
            elif task == "counting":
                ann = make_counting(image_id, image_path, sample_seed, local_rng, colors, shapes, image_size, object_size)
            elif task == "negation":
                ann = make_negation(image_id, image_path, sample_seed, local_rng, colors, shapes, image_size, object_size)
            else:
                raise ValueError(f"Unknown task: {task}")

            draw_scene(
                ann.objects,
                image_path,
                image_size=image_size,
                background=background,
                noise=noise,
                seed=sample_seed,
            )
            annotations.append(
                Annotation(
                    image_id=ann.image_id,
                    image_path=(Path("images") / image_path.name).as_posix(),
                    task_type=ann.task_type,
                    objects=ann.objects,
                    attributes=ann.attributes,
                    relation=ann.relation,
                    correct_caption=ann.correct_caption,
                    hard_negative_captions=ann.hard_negative_captions,
                    metadata={**ann.metadata, "image_size": image_size, "noise": noise},
                    seed=ann.seed,
                )
            )
            sample_idx += 1
            rng.random()

    write_jsonl(annotations_path, [ann.to_dict() for ann in annotations])
    return annotations


def make_attribute_binding(
    image_id: str,
    image_path: Path,
    seed: int,
    rng: random.Random,
    colors: list[str],
    shapes: list[str],
    image_size: int,
    object_size: int,
) -> Annotation:
    color_a, color_b = rng.sample(colors, 2)
    shape_a, shape_b = rng.sample(shapes, 2)
    y = image_size // 2
    objects = [
        ObjectSpec(shape_a, color_a, image_size // 3, y, object_size),
        ObjectSpec(shape_b, color_b, 2 * image_size // 3, y, object_size),
    ]
    correct = attribute_binding_caption(color_a, shape_a, color_b, shape_b)
    negative = attribute_binding_caption(color_b, shape_a, color_a, shape_b)
    return Annotation(
        image_id=image_id,
        image_path=str(image_path),
        task_type="attribute_binding",
        objects=objects,
        attributes={"colors": [color_a, color_b], "shapes": [shape_a, shape_b]},
        relation=None,
        correct_caption=correct,
        hard_negative_captions=[negative],
        metadata={"failure_type": "attribute_binding_failure"},
        seed=seed,
    )


def make_spatial_left_right(
    image_id: str,
    image_path: Path,
    seed: int,
    rng: random.Random,
    colors: list[str],
    shapes: list[str],
    image_size: int,
    object_size: int,
) -> Annotation:
    color_a, color_b = rng.sample(colors, 2)
    shape_a, shape_b = rng.sample(shapes, 2)
    left_x = image_size // 3
    right_x = 2 * image_size // 3
    y = image_size // 2
    objects = [ObjectSpec(shape_a, color_a, left_x, y, object_size), ObjectSpec(shape_b, color_b, right_x, y, object_size)]
    correct = spatial_caption(color_a, shape_a, "left_of", color_b, shape_b)
    negatives = [
        spatial_caption(color_b, shape_b, "left_of", color_a, shape_a),
        spatial_caption(color_a, shape_a, "right_of", color_b, shape_b),
    ]
    variants = prompt_variants(color_a, shape_a, "left_of", color_b, shape_b)
    return Annotation(
        image_id=image_id,
        image_path=str(image_path),
        task_type="spatial_left_right",
        objects=objects,
        attributes={"colors": [color_a, color_b], "shapes": [shape_a, shape_b]},
        relation="left_of",
        correct_caption=correct,
        hard_negative_captions=negatives,
        metadata={"failure_type": "spatial_relation_inversion", "prompt_variants": variants},
        seed=seed,
    )


def make_spatial_above_below(
    image_id: str,
    image_path: Path,
    seed: int,
    rng: random.Random,
    colors: list[str],
    shapes: list[str],
    image_size: int,
    object_size: int,
) -> Annotation:
    color_a, color_b = rng.sample(colors, 2)
    shape_a, shape_b = rng.sample(shapes, 2)
    x = image_size // 2
    objects = [
        ObjectSpec(shape_a, color_a, x, image_size // 3, object_size),
        ObjectSpec(shape_b, color_b, x, 2 * image_size // 3, object_size),
    ]
    correct = spatial_caption(color_a, shape_a, "above", color_b, shape_b)
    negatives = [
        spatial_caption(color_a, shape_a, "below", color_b, shape_b),
        spatial_caption(color_b, shape_b, "above", color_a, shape_a),
    ]
    variants = prompt_variants(color_a, shape_a, "above", color_b, shape_b)
    return Annotation(
        image_id=image_id,
        image_path=str(image_path),
        task_type="spatial_above_below",
        objects=objects,
        attributes={"colors": [color_a, color_b], "shapes": [shape_a, shape_b]},
        relation="above",
        correct_caption=correct,
        hard_negative_captions=negatives,
        metadata={"failure_type": "spatial_relation_inversion", "prompt_variants": variants},
        seed=seed,
    )


def make_counting(
    image_id: str,
    image_path: Path,
    seed: int,
    rng: random.Random,
    colors: list[str],
    shapes: list[str],
    image_size: int,
    object_size: int,
) -> Annotation:
    color = rng.choice(colors)
    shape = rng.choice(shapes)
    count = rng.choice([2, 3, 4])
    positions = grid_positions(image_size, count)
    objects = [ObjectSpec(shape, color, x, y, max(34, object_size - 14)) for x, y in positions]
    wrong_count = count - 1 if count > 2 else count + 1
    correct = object_phrase(color, shape, count)
    negative = object_phrase(color, shape, wrong_count)
    return Annotation(
        image_id=image_id,
        image_path=str(image_path),
        task_type="counting",
        objects=objects,
        attributes={"color": color, "shape": shape, "count": count},
        relation=None,
        correct_caption=correct,
        hard_negative_captions=[negative],
        metadata={"failure_type": "counting_failure"},
        seed=seed,
    )


def make_negation(
    image_id: str,
    image_path: Path,
    seed: int,
    rng: random.Random,
    colors: list[str],
    shapes: list[str],
    image_size: int,
    object_size: int,
) -> Annotation:
    color = rng.choice([c for c in colors if c != "black"])
    shape = rng.choice(shapes)
    has_border = rng.choice([False, False, True])
    objects = [ObjectSpec(shape, color, image_size // 2, image_size // 2, object_size + 8, has_border, "black" if has_border else None)]
    if has_border:
        correct = f"{object_phrase(color, shape)} with a black border"
        negative = f"{object_phrase(color, shape)} without a black border"
    else:
        correct = f"{object_phrase(color, shape)} without a black border"
        negative = f"{object_phrase(color, shape)} with a black border"
    return Annotation(
        image_id=image_id,
        image_path=str(image_path),
        task_type="negation",
        objects=objects,
        attributes={"color": color, "shape": shape, "has_border": has_border},
        relation="border_absence" if not has_border else "border_presence",
        correct_caption=correct,
        hard_negative_captions=[negative],
        metadata={"failure_type": "negation_failure"},
        seed=seed,
    )


def grid_positions(image_size: int, count: int) -> list[tuple[int, int]]:
    if count == 2:
        return [(image_size // 3, image_size // 2), (2 * image_size // 3, image_size // 2)]
    if count == 3:
        return [(image_size // 2, image_size // 3), (image_size // 3, 2 * image_size // 3), (2 * image_size // 3, 2 * image_size // 3)]
    return [
        (image_size // 3, image_size // 3),
        (2 * image_size // 3, image_size // 3),
        (image_size // 3, 2 * image_size // 3),
        (2 * image_size // 3, 2 * image_size // 3),
    ]
