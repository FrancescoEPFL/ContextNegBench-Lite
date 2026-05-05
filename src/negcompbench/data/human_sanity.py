from __future__ import annotations

from pathlib import Path

import pandas as pd

from negcompbench.data.context_neg_dataset import read_csv_rows, scan_supported_images
from negcompbench.utils.io import ensure_dir, read_jsonl


def build_human_sanity_table(scenarios: list[dict], output_path: str | Path) -> pd.DataFrame:
    rows = [scenario_sanity_row(scenario) for scenario in scenarios]
    frame = pd.DataFrame(rows)
    output = Path(output_path)
    ensure_dir(output.parent)
    frame.to_csv(output, index=False)
    return frame


def scenario_sanity_row(scenario: dict) -> dict:
    root = Path(scenario["root"])
    object_name = scenario["object"]
    annotations_path = root / "annotations.jsonl"
    review_status_path = root / "metadata" / "review_status.csv"
    source_path = root / "metadata" / "sources.csv"
    annotations = read_jsonl(annotations_path) if annotations_path.exists() else []
    review_rows = read_csv_rows(review_status_path)
    source_rows = read_csv_rows(source_path)
    accepted_rows = [row for row in review_rows if row.get("status") == "accepted"]
    rejected_rows = [row for row in review_rows if row.get("status") == "rejected"]
    with_folder = root / "reviewed" / f"with_{object_name.replace(' ', '_')}"
    without_folder = root / "reviewed" / f"without_{object_name.replace(' ', '_')}"
    with_images = scan_supported_images(with_folder)
    without_images = scan_supported_images(without_folder)
    with_annotations = [row for row in annotations if row.get("condition") == "with_object"]
    without_annotations = [row for row in annotations if row.get("condition") == "without_object"]
    return {
        "scenario": scenario["scenario"],
        "scene": scenario["scene"],
        "object": object_name,
        "root": root.as_posix(),
        "review_protocol": "manual folder review",
        "labeling_models_used": "none",
        "n_reviewed_with_object": len(with_images),
        "n_reviewed_without_object": len(without_images),
        "n_annotations_with_object": len(with_annotations),
        "n_annotations_without_object": len(without_annotations),
        "n_review_status_accepted": len(accepted_rows),
        "n_review_status_rejected": len(rejected_rows),
        "n_source_rows": len(source_rows),
        "has_annotations": annotations_path.exists(),
        "has_review_status": review_status_path.exists(),
    }
