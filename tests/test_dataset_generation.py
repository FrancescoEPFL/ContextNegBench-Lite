from pathlib import Path

from negcompbench.data.generate_dataset import generate_dataset
from negcompbench.utils.io import read_jsonl


def test_generate_dataset_writes_expected_rows(tmp_path: Path):
    config = {
        "seed": 7,
        "samples_per_task": 2,
        "image_size": 128,
        "object_size": 36,
        "tasks": ["attribute_binding", "spatial_left_right", "spatial_above_below", "counting", "negation"],
        "colors": ["red", "blue", "green", "yellow"],
        "shapes": ["square", "circle", "triangle"],
    }
    annotations = generate_dataset(config, tmp_path)
    rows = read_jsonl(tmp_path / "annotations.jsonl")
    assert len(annotations) == 10
    assert len(rows) == 10
    assert (tmp_path / rows[0]["image_path"]).exists()
    assert rows[0]["correct_caption"]
    assert rows[0]["hard_negative_captions"]
