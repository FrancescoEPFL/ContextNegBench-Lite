from __future__ import annotations

import csv
from pathlib import Path

from negcompbench.data.context_neg_dataset import IMAGE_EXTENSIONS, load_source_lookup
from negcompbench.data.context_neg_text import english_object_form
from negcompbench.utils.io import ensure_dir, write_jsonl

SUPPORTED_LANGUAGES = {"en", "it"}

CAPTION_TEMPLATES = {
    "en": {
        "generic": "a photo of a {scene}",
        "positive": "a photo of a {scene} with {object_np}",
        "negative": "a photo of a {scene} without {object_np}",
    },
    "it": {
        "generic": "foto di una {scene_it}",
        "positive": "foto di una {scene_it} con {object_it}",
        "negative": "foto di una {scene_it} senza {object_it}",
    },
}

ITALIAN_TERMS = {
    "kitchen": "cucina",
    "table": "tavolo",
    "bedroom": "camera da letto",
    "bed": "letto",
    "street": "strada",
    "car": "auto",
}


def normalize_languages(languages: list[str] | tuple[str, ...]) -> list[str]:
    normalized = []
    for language in languages:
        for part in language.split(","):
            lang = part.strip().lower()
            if not lang:
                continue
            if lang not in SUPPORTED_LANGUAGES:
                raise ValueError(f"Unsupported language: {language}. Supported: {sorted(SUPPORTED_LANGUAGES)}")
            if lang not in normalized:
                normalized.append(lang)
    if not normalized:
        raise ValueError("At least one language must be provided.")
    return normalized


def build_context_neg_annotations(
    root: str | Path,
    scene: str,
    object_name: str,
    languages: list[str] | tuple[str, ...] = ("en", "it"),
    scene_it: str | None = None,
    object_it: str | None = None,
    cwd: str | Path | None = None,
) -> list[dict]:
    root_path = Path(root)
    languages = normalize_languages(languages)
    captions = make_captions(scene, object_name, languages, scene_it=scene_it, object_it=object_it)
    sources = load_source_lookup(root_path)

    rows: list[dict] = []
    index_rows: list[dict[str, str]] = []
    for condition_dir, condition in condition_directories(object_name):
        image_dir = root_path / "reviewed" / condition_dir
        for index, image_path in enumerate(sorted(scan_images(image_dir), key=lambda path: path.as_posix().lower()), start=1):
            relative_path = to_relative_path(image_path, cwd=cwd)
            metadata = sources.get(relative_path, {})
            image_id = context_neg_image_id(scene, object_name, condition, index)
            original_filename = image_path.name
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": relative_path,
                    "scene": scene,
                    "object": object_name,
                    "condition": condition,
                    "captions": captions,
                    "metadata": {
                        "source_url": metadata.get("source_url") or None,
                        "license": metadata.get("license") or None,
                        "reviewed": True,
                        "original_filename": original_filename,
                    },
                }
            )
            index_rows.append(
                {
                    "image_id": image_id,
                    "condition": condition,
                    "image_path": relative_path,
                    "original_filename": original_filename,
                }
            )

    output_path = root_path / "annotations.jsonl"
    ensure_dir(output_path.parent)
    write_jsonl(output_path, rows)
    write_image_index(root_path / "image_index.csv", index_rows)
    return rows


def make_captions(
    scene: str,
    object_name: str,
    languages: list[str] | tuple[str, ...],
    scene_it: str | None = None,
    object_it: str | None = None,
) -> dict[str, str]:
    languages = normalize_languages(languages)
    scene_it = scene_it or ITALIAN_TERMS.get(scene, scene)
    object_it = object_it or ITALIAN_TERMS.get(object_name, object_name)
    object_form = english_object_form(scene, object_name)
    values = {
        "scene": scene,
        "object": object_form.text,
        "object_np": object_form.noun_phrase,
        "scene_it": scene_it,
        "object_it": object_it,
    }
    captions: dict[str, str] = {}
    for language in languages:
        for role, template in CAPTION_TEMPLATES[language].items():
            captions[f"{role}_{language}"] = template.format(**values)
    return captions


def scan_images(image_dir: str | Path) -> list[Path]:
    directory = Path(image_dir)
    if not directory.exists():
        return []
    return [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def condition_directories(object_name: str) -> list[tuple[str, str]]:
    object_slug = object_name.strip().lower().replace(" ", "_")
    return [(f"with_{object_slug}", "with_object"), (f"without_{object_slug}", "without_object")]


def context_neg_image_id(scene: str, object_name: str, condition: str, index: int) -> str:
    scene_slug = scene.strip().lower().replace(" ", "_")
    object_slug = object_name.strip().lower().replace(" ", "_")
    condition_slug = "with" if condition == "with_object" else "without"
    return f"{scene_slug}_{object_slug}_{condition_slug}_{index:04d}"


def to_relative_path(path: Path, cwd: str | Path | None = None) -> str:
    base = Path(cwd) if cwd is not None else Path.cwd()
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def write_image_index(path: str | Path, rows: list[dict[str, str]]) -> None:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "condition", "image_path", "original_filename"])
        writer.writeheader()
        writer.writerows(rows)
