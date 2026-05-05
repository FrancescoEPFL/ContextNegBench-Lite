from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negcompbench.data.generate_dataset import generate_dataset  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a small redistributable synthetic demo dataset.")
    parser.add_argument("--output", default="data/sample_synthetic")
    parser.add_argument("--samples-per-task", type=int, default=3)
    parser.add_argument("--seed", type=int, default=101)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = ROOT / args.output
    config = {
        "seed": args.seed,
        "samples_per_task": args.samples_per_task,
        "image_size": 224,
        "object_size": 54,
        "tasks": ["attribute_binding", "spatial_left_right", "spatial_above_below", "counting", "negation"],
        "colors": ["red", "blue", "green", "yellow", "purple", "orange"],
        "shapes": ["square", "circle", "triangle", "star"],
        "background": "white",
        "noise": "clean",
    }
    annotations = generate_dataset(config, output)
    write_manifest(output, config, len(annotations))
    print(f"Wrote {len(annotations)} synthetic samples to {output}")
    return 0


def write_manifest(root: Path, config: dict, n_samples: int) -> None:
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    manifest = {
        "name": "sample_synthetic",
        "description": "Small generated geometric dataset for license-safe pipeline demos.",
        "license": "CC0-1.0",
        "source": "Generated locally by scripts/generate_sample_synthetic_dataset.py.",
        "n_samples": n_samples,
        "config": config,
        "files": files,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
