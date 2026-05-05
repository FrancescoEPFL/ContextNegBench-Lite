from pathlib import Path

from negcompbench.eval.schema_validation import SCHEMAS, validate_csv, validate_known_result_files


def test_validate_public_result_schemas():
    assert validate_known_result_files(Path.cwd()) == []


def test_sample_synthetic_dataset_is_committed():
    root = Path.cwd() / "data" / "sample_synthetic"
    assert (root / "annotations.jsonl").exists()
    assert (root / "manifest.json").exists()
    assert len(list((root / "images").glob("*.png"))) == 15


def test_validate_csv_reports_missing_columns(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text("model,value\nx,1\n", encoding="utf-8")
    errors = validate_csv(path, SCHEMAS["dog_grass_by_model"])
    assert errors
    assert "missing required columns" in errors[0]
