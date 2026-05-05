from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negcompbench.data.schemas import Annotation  # noqa: E402
from negcompbench.eval.run_eval import run_evaluation  # noqa: E402
from negcompbench.utils.io import write_jsonl  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="contextneg_smoke_") as tmp:
        workdir = Path(tmp)
        image_path = workdir / "smoke.png"
        annotations_path = workdir / "annotations.jsonl"
        output_dir = workdir / "out"

        Image.new("RGB", (32, 32), color=(70, 160, 90)).save(image_path)
        annotation = Annotation(
            image_id="smoke_0001",
            image_path=image_path.name,
            task_type="smoke",
            objects=[],
            attributes={"scene": "synthetic"},
            relation=None,
            correct_caption="a green square",
            hard_negative_captions=["a red square", "a blue circle"],
            metadata={"source": "generated_smoke_test"},
            seed=0,
        )
        write_jsonl(annotations_path, [annotation.to_dict()])

        rows = run_evaluation(
            model_name="random_baseline",
            annotations_path=annotations_path,
            output_dir=output_dir,
            batch_size=1,
            seed=0,
        )

        expected_outputs = [output_dir / "results.jsonl", output_dir / "summary.csv"]
        if len(rows) != 1 or not all(path.exists() for path in expected_outputs):
            raise RuntimeError("Smoke test did not produce the expected evaluation outputs.")

    print("Smoke test passed: generated one synthetic sample and evaluation outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
