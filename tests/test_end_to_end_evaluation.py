from pathlib import Path

from PIL import Image

from negcompbench.data.schemas import Annotation
from negcompbench.eval.run_eval import run_evaluation
from negcompbench.utils.io import write_jsonl


def test_random_baseline_end_to_end_is_seeded(tmp_path: Path):
    image_path = tmp_path / "image.png"
    annotations_path = tmp_path / "annotations.jsonl"
    Image.new("RGB", (24, 24), color=(40, 120, 80)).save(image_path)
    annotation = Annotation(
        image_id="seeded",
        image_path=image_path.name,
        task_type="unit",
        objects=[],
        attributes={},
        relation=None,
        correct_caption="a green square",
        hard_negative_captions=["a red square"],
    )
    write_jsonl(annotations_path, [annotation.to_dict()])

    rows_a = run_evaluation("random_baseline", annotations_path, tmp_path / "out_a", batch_size=1, seed=123)
    rows_b = run_evaluation("random_baseline", annotations_path, tmp_path / "out_b", batch_size=1, seed=123)

    assert rows_a[0]["scores"] == rows_b[0]["scores"]
    assert (tmp_path / "out_a" / "summary.csv").exists()
