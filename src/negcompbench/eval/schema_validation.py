from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CsvSchema:
    name: str
    required_columns: frozenset[str]
    numeric_columns: frozenset[str] = frozenset()
    non_empty: bool = True


SCHEMAS: dict[str, CsvSchema] = {
    "dog_grass_by_model": CsvSchema(
        name="dog_grass_by_model",
        required_columns=frozenset({"model", "pair_id", "metric", "value", "ci_low", "ci_high", "n"}),
        numeric_columns=frozenset({"value", "ci_low", "ci_high", "n"}),
    ),
    "final_contextneg_by_model": CsvSchema(
        name="final_contextneg_by_model",
        required_columns=frozenset({"model", "scenario", "prompt_group", "metric", "value", "ci_low", "ci_high", "n"}),
        numeric_columns=frozenset({"value", "ci_low", "ci_high", "n"}),
    ),
    "human_sanity_table": CsvSchema(
        name="human_sanity_table",
        required_columns=frozenset({"scenario", "scene", "object", "root", "n_reviewed_with_object", "n_reviewed_without_object"}),
        numeric_columns=frozenset({"n_reviewed_with_object", "n_reviewed_without_object"}),
    ),
}


def validate_csv(path: str | Path, schema: CsvSchema) -> list[str]:
    csv_path = Path(path)
    errors: list[str] = []
    if not csv_path.exists():
        return [f"{csv_path}: file does not exist"]

    try:
        frame = pd.read_csv(csv_path)
    except Exception as exc:  # pragma: no cover - pandas error text varies by version
        return [f"{csv_path}: could not read CSV: {exc}"]

    if schema.non_empty and frame.empty:
        errors.append(f"{csv_path}: CSV is empty")

    missing = sorted(schema.required_columns.difference(frame.columns))
    if missing:
        errors.append(f"{csv_path}: missing required columns: {', '.join(missing)}")

    for column in sorted(schema.numeric_columns.intersection(frame.columns)):
        parsed = pd.to_numeric(frame[column], errors="coerce")
        if parsed.isna().any():
            errors.append(f"{csv_path}: column {column} contains non-numeric values")

    return errors


def validate_known_result_files(root: str | Path) -> list[str]:
    base = Path(root)
    checks = [
        (base / "results" / "model_matrix_summary" / "dog_grass_by_model.csv", SCHEMAS["dog_grass_by_model"]),
        (base / "results" / "model_matrix_summary" / "final_contextneg_by_model.csv", SCHEMAS["final_contextneg_by_model"]),
        (base / "results" / "human_sanity_table.csv", SCHEMAS["human_sanity_table"]),
    ]
    errors: list[str] = []
    for path, schema in checks:
        errors.extend(validate_csv(path, schema))
    return errors
