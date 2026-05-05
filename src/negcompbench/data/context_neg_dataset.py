from __future__ import annotations

import csv
import hashlib
import html
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from negcompbench.utils.io import ensure_dir

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_OBJECT = "table"
SOURCE_COLUMNS = ["relative_path", "original_path", "source_url", "license", "condition_hint", "sha256", "width", "height"]
REVIEW_COLUMNS = ["relative_path", "area", "condition", "status", "sha256", "width", "height"]


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    relative_path: str
    area: str
    condition: str
    sha256: str
    width: int
    height: int
    source_url: str | None = None
    license: str | None = None


def object_slug(object_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", object_name.strip().lower()).strip("_")


def positive_folder(object_name: str) -> str:
    return f"with_{object_slug(object_name)}"


def negative_folder(object_name: str) -> str:
    return f"without_{object_slug(object_name)}"


def raw_conditions(object_name: str) -> tuple[str, str, str]:
    return (positive_folder(object_name), negative_folder(object_name), "unsure")


def reviewed_conditions(object_name: str) -> tuple[str, str, str]:
    return (positive_folder(object_name), negative_folder(object_name), "rejected")


def normalize_condition(condition: str, object_name: str) -> str:
    value = condition.strip().lower()
    positive = positive_folder(object_name)
    negative = negative_folder(object_name)
    aliases = {
        "with_object": positive,
        "positive": positive,
        positive: positive,
        "without_object": negative,
        "negative": negative,
        negative: negative,
        "unsure": "unsure",
        "rejected": "rejected",
    }
    if value not in aliases:
        raise ValueError(f"Condition must be one of: with_object, without_object, {positive}, {negative}, unsure, rejected")
    return aliases[value]


def semantic_condition(folder: str, object_name: str) -> str:
    if folder == positive_folder(object_name):
        return "with_object"
    if folder == negative_folder(object_name):
        return "without_object"
    return folder


def ensure_context_neg_structure(root: str | Path, object_name: str = DEFAULT_OBJECT) -> None:
    root_path = Path(root)
    for condition in raw_conditions(object_name):
        ensure_dir(root_path / "raw" / condition)
    for condition in reviewed_conditions(object_name):
        ensure_dir(root_path / "reviewed" / condition)
    ensure_dir(root_path / "metadata")
    ensure_dir(root_path / "metadata" / "thumbnails")
    ensure_csv(root_path / "metadata" / "sources.csv", SOURCE_COLUMNS)
    ensure_csv(root_path / "metadata" / "review_status.csv", REVIEW_COLUMNS)


def ensure_csv(path: Path, columns: list[str]) -> None:
    if path.exists():
        return
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()


def prepare_context_neg_dataset(
    root: str | Path,
    ingest_folder: str | Path | None = None,
    condition: str = "unsure",
    auto_stage: bool = False,
    object_name: str = DEFAULT_OBJECT,
) -> list[str]:
    root_path = Path(root)
    ensure_context_neg_structure(root_path, object_name)
    warnings: list[str] = []
    seen_hashes = known_hashes(root_path)

    if ingest_folder is not None:
        warnings.extend(ingest_images(root_path, Path(ingest_folder), condition, seen_hashes, object_name))

    warnings.extend(normalize_tree_filenames(root_path))
    seen_hashes = known_hashes(root_path)
    warnings.extend(remove_duplicates(root_path, seen_hashes))

    if auto_stage:
        warnings.extend(auto_stage_raw_images(root_path, object_name))

    write_review_status(root_path, object_name)
    return warnings


def ingest_images(root: Path, ingest_folder: Path, condition: str, seen_hashes: set[str], object_name: str = DEFAULT_OBJECT) -> list[str]:
    condition = normalize_condition(condition, object_name)
    if condition not in raw_conditions(object_name):
        raise ValueError(f"Condition must be one of {raw_conditions(object_name)}")
    if not ingest_folder.exists():
        raise FileNotFoundError(f"Ingest folder does not exist: {ingest_folder}")
    warnings: list[str] = []
    destination = root / "raw" / condition
    rows = read_csv_rows(root / "metadata" / "sources.csv")

    for source in sorted([path for path in ingest_folder.rglob("*") if path.is_file()], key=lambda path: path.as_posix().lower()):
        try:
            source_hash = file_sha256(source)
        except OSError as exc:
            warnings.append(f"Skipping unreadable file {source}: {exc}")
            continue
        if source_hash in seen_hashes:
            warnings.append(f"Skipping duplicate file {source}")
            continue

        try:
            width, height = image_dimensions(source)
        except Exception:
            warnings.append(f"Skipping unsupported or invalid image {source}")
            continue

        target = unique_path(destination, normalize_filename(source.name, default_stem=f"context_neg_{condition}"))
        if source.suffix.lower() in IMAGE_EXTENSIONS:
            shutil.copy2(source, target)
        else:
            target = target.with_suffix(".jpg")
            convert_to_jpg(source, target)

        target_hash = file_sha256(target)
        seen_hashes.add(target_hash)
        rows.append(
            {
                "relative_path": to_relative_path(target),
                "original_path": str(source),
                "source_url": "",
                "license": "",
                "condition_hint": condition,
                "sha256": target_hash,
                "width": str(width),
                "height": str(height),
            }
        )

    write_csv_rows(root / "metadata" / "sources.csv", SOURCE_COLUMNS, rows)
    return warnings


def normalize_tree_filenames(root: Path) -> list[str]:
    warnings: list[str] = []
    for base in [root / "raw", root / "reviewed"]:
        for path in sorted(scan_supported_images(base), key=lambda item: item.as_posix().lower()):
            normalized = normalize_filename(path.name, default_stem="context_neg_image")
            if normalized == path.name:
                continue
            target = unique_path(path.parent, normalized)
            path.rename(target)
            warnings.append(f"Renamed {path} -> {target}")
    return warnings


def remove_duplicates(root: Path, known: set[str] | None = None) -> list[str]:
    del known
    warnings: list[str] = []
    seen: dict[str, Path] = {}
    for path in sorted(scan_dataset_images(root), key=lambda item: item.as_posix().lower()):
        digest = file_sha256(path)
        if digest in seen:
            target = unique_path(root / "reviewed" / "rejected", path.name)
            shutil.move(str(path), target)
            warnings.append(f"Moved duplicate {path} to {target}; first copy is {seen[digest]}")
        else:
            seen[digest] = path
    return warnings


def auto_stage_raw_images(root: Path, object_name: str = DEFAULT_OBJECT) -> list[str]:
    warnings: list[str] = []
    for condition in (positive_folder(object_name), negative_folder(object_name)):
        for path in sorted(scan_supported_images(root / "raw" / condition), key=lambda item: item.as_posix().lower()):
            target = unique_path(root / "reviewed" / condition, path.name)
            shutil.copy2(path, target)
            warnings.append(f"Auto-staged {path} -> {target}")
    return warnings


def write_review_status(root: Path, object_name: str = DEFAULT_OBJECT) -> list[ImageRecord]:
    records = collect_image_records(root, object_name)
    rows = [
        {
            "relative_path": record.relative_path,
            "area": record.area,
            "condition": record.condition,
            "status": infer_status(record.area, record.condition),
            "sha256": record.sha256,
            "width": str(record.width),
            "height": str(record.height),
        }
        for record in records
    ]
    write_csv_rows(root / "metadata" / "review_status.csv", REVIEW_COLUMNS, rows)
    return records


def collect_image_records(root: str | Path, object_name: str = DEFAULT_OBJECT) -> list[ImageRecord]:
    root_path = Path(root)
    sources = load_source_lookup(root_path)
    records: list[ImageRecord] = []
    for area, conditions in [("raw", raw_conditions(object_name)), ("reviewed", reviewed_conditions(object_name))]:
        for condition in conditions:
            directory = root_path / area / condition
            for path in sorted(scan_supported_images(directory), key=lambda item: item.as_posix().lower()):
                width, height = image_dimensions(path)
                relative_path = to_relative_path(path)
                source = sources.get(relative_path, {})
                records.append(
                    ImageRecord(
                        path=path,
                        relative_path=relative_path,
                        area=area,
                        condition=condition,
                        sha256=file_sha256(path),
                        width=width,
                        height=height,
                        source_url=source.get("source_url") or None,
                        license=source.get("license") or None,
                    )
                )
    return records


def make_review_gallery(
    root: str | Path, output_path: str | Path | None = None, scene: str = "kitchen", object_name: str = DEFAULT_OBJECT
) -> Path:
    root_path = Path(root)
    ensure_context_neg_structure(root_path, object_name)
    records = write_review_status(root_path, object_name)
    thumbs_dir = ensure_dir(root_path / "metadata" / "thumbnails")
    sections = []
    for area, conditions in [("raw", raw_conditions(object_name)), ("reviewed", reviewed_conditions(object_name))]:
        for condition in conditions:
            section_records = [record for record in records if record.area == area and record.condition == condition]
            sections.append(render_gallery_section(root_path, thumbs_dir, f"{area}/{condition}", section_records, object_name))
    output = Path(output_path) if output_path is not None else root_path / "review_gallery.html"
    output.write_text(render_gallery_html(sections, scene, object_name), encoding="utf-8")
    return output


def render_gallery_section(root: Path, thumbs_dir: Path, title: str, records: list[ImageRecord], object_name: str = DEFAULT_OBJECT) -> str:
    cards = []
    for record in records:
        thumb = create_thumbnail(record.path, thumbs_dir)
        cards.append(
            "<article>"
            f'<img src="{html.escape(relative_between(thumb, root))}" alt="{html.escape(record.path.name)}">'
            f"<h3>{html.escape(record.path.name)}</h3>"
            f"<p><strong>Path:</strong> {html.escape(record.relative_path)}</p>"
            f"<p><strong>Inferred:</strong> {html.escape(infer_condition(record.area, record.condition, object_name))}</p>"
            f"<p><strong>Size:</strong> {record.width} x {record.height}</p>"
            f"<p><strong>Source:</strong> {html.escape(record.source_url or 'unknown')}</p>"
            "</article>"
        )
    body = "\n".join(cards) if cards else "<p>No images found.</p>"
    return f'<section><h2>{html.escape(title)}</h2><div class="grid">{body}</div></section>'


def render_gallery_html(sections: list[str], scene: str = "kitchen", object_name: str = DEFAULT_OBJECT) -> str:
    obj = html.escape(object_name)
    scene_text = html.escape(scene)
    checklist = f"""
<aside>
  <h2>Human Review Checklist</h2>
  <h3>with_{obj}</h3>
  <ul>
    <li>Is this clearly a {scene_text}?</li>
    <li>Is a {obj} clearly visible?</li>
    <li>Is the object not ambiguous or only implied?</li>
    <li>Is the image not too dark, blurry, or cropped?</li>
  </ul>
  <h3>without_{obj}</h3>
  <ul>
    <li>Is this clearly a {scene_text}?</li>
    <li>Is there no visible {obj}?</li>
    <li>Is there no object likely to be interpreted as a {obj}?</li>
    <li>Is the absence visually plausible and not due to extreme crop?</li>
  </ul>
  <h3>reject</h3>
  <ul>
    <li>ambiguous scene</li>
    <li>{obj} ambiguous</li>
    <li>image too low quality</li>
    <li>people/objects dominate</li>
    <li>watermark or heavy text overlay</li>
    <li>duplicate</li>
  </ul>
</aside>
"""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ContextNeg-Test Review Gallery</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #202020; }}
    aside {{ border: 1px solid #ccc; border-radius: 8px; padding: 16px; margin-bottom: 24px; background: #fafafa; }}
    section {{ margin-bottom: 28px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 14px; }}
    article {{ border: 1px solid #ddd; border-radius: 8px; padding: 10px; background: white; }}
    img {{ width: 100%; max-width: 200px; height: auto; display: block; margin-bottom: 8px; }}
    h1 {{ font-size: 26px; }}
    h2 {{ font-size: 20px; }}
    h3 {{ font-size: 15px; margin: 8px 0; }}
    p, li {{ font-size: 13px; }}
  </style>
</head>
<body>
  <h1>ContextNeg-Test Review Gallery</h1>
  {checklist}
  {"".join(sections)}
</body>
</html>
"""


def check_context_neg_dataset(root: str | Path, min_per_condition: int = 5, object_name: str = DEFAULT_OBJECT) -> tuple[list[str], Path]:
    root_path = Path(root)
    ensure_context_neg_structure(root_path, object_name)
    records = collect_image_records(root_path, object_name)
    warnings: list[str] = []
    positive = positive_folder(object_name)
    negative = negative_folder(object_name)
    reviewed_with = [record for record in records if record.area == "reviewed" and record.condition == positive]
    reviewed_without = [record for record in records if record.area == "reviewed" and record.condition == negative]
    if len(reviewed_with) < min_per_condition:
        warnings.append(f"reviewed/{positive} has {len(reviewed_with)} images; recommended minimum is {min_per_condition}.")
    if len(reviewed_without) < min_per_condition:
        warnings.append(f"reviewed/{negative} has {len(reviewed_without)} images; recommended minimum is {min_per_condition}.")
    for record in records:
        if record.width < 128 or record.height < 128:
            warnings.append(f"Small image: {record.relative_path} is {record.width}x{record.height}.")
    duplicates = duplicate_hashes(records)
    for digest, duplicate_records in duplicates.items():
        paths = ", ".join(record.relative_path for record in duplicate_records)
        warnings.append(f"Duplicate hash {digest[:12]} across: {paths}")
    report_path = root_path / "dataset_report.md"
    report_path.write_text(render_dataset_report(records, warnings, object_name), encoding="utf-8")
    return warnings, report_path


def render_dataset_report(records: list[ImageRecord], warnings: list[str], object_name: str = DEFAULT_OBJECT) -> str:
    lines = ["# ContextNeg-Test Dataset Report", ""]
    for area, conditions in [("raw", raw_conditions(object_name)), ("reviewed", reviewed_conditions(object_name))]:
        lines.append(f"## {area}")
        lines.append("")
        for condition in conditions:
            count = len([record for record in records if record.area == area and record.condition == condition])
            lines.append(f"- `{condition}`: {count}")
        lines.append("")
    lines.append("## Warnings")
    lines.append("")
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("- No warnings.")
    lines.append("")
    return "\n".join(lines)


def duplicate_hashes(records: list[ImageRecord]) -> dict[str, list[ImageRecord]]:
    by_hash: dict[str, list[ImageRecord]] = {}
    for record in records:
        by_hash.setdefault(record.sha256, []).append(record)
    return {digest: items for digest, items in by_hash.items() if len(items) > 1}


def load_source_lookup(root: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_rows(root / "metadata" / "sources.csv")
    return {row["relative_path"]: row for row in rows if row.get("relative_path")}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def scan_supported_images(root: str | Path) -> list[Path]:
    root_path = Path(root)
    if not root_path.exists():
        return []
    return [path for path in root_path.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]


def normalize_filename(filename: str, default_stem: str = "image") -> str:
    path = Path(filename)
    stem = re.sub(r"[^a-z0-9]+", "_", path.stem.lower()).strip("_")
    stem = stem or default_stem
    suffix = path.suffix.lower()
    if suffix == ".jpeg":
        suffix = ".jpg"
    if suffix not in IMAGE_EXTENSIONS:
        suffix = ".jpg"
    return f"{stem}{suffix}"


def unique_path(directory: Path, filename: str) -> Path:
    ensure_dir(directory)
    candidate = directory / filename
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    index = 2
    while True:
        candidate = directory / f"{stem}_{index:03d}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def known_hashes(root: Path) -> set[str]:
    return {file_sha256(path) for path in scan_dataset_images(root)}


def scan_dataset_images(root: Path) -> list[Path]:
    paths: list[Path] = []
    for folder in [root / "raw", root / "reviewed"]:
        paths.extend(scan_supported_images(folder))
    return paths


def image_dimensions(path: str | Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def convert_to_jpg(source: Path, target: Path) -> None:
    with Image.open(source) as image:
        image.convert("RGB").save(target, quality=92)


def create_thumbnail(path: Path, thumbs_dir: Path, size: int = 220) -> Path:
    thumb_name = f"{file_sha256(path)[:16]}.jpg"
    target = thumbs_dir / thumb_name
    if target.exists():
        return target
    with Image.open(path) as image:
        image.thumbnail((size, size))
        image.convert("RGB").save(target, quality=88)
    return target


def infer_status(area: str, condition: str) -> str:
    if area == "reviewed" and condition == "rejected":
        return "rejected"
    if area == "reviewed":
        return "accepted"
    return "needs_review"


def infer_condition(area: str, condition: str, object_name: str = DEFAULT_OBJECT) -> str:
    if area == "reviewed" and condition == "rejected":
        return "rejected"
    if condition == positive_folder(object_name):
        return "with_object"
    if condition == negative_folder(object_name):
        return "without_object"
    return "unsure"


def to_relative_path(path: Path, cwd: str | Path | None = None) -> str:
    base = Path(cwd) if cwd is not None else Path.cwd()
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def relative_between(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()
