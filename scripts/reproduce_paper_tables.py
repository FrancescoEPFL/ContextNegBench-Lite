from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negcompbench.eval.run_eval import run_evaluation  # noqa: E402
from negcompbench.eval.schema_validation import validate_known_result_files  # noqa: E402


PUBLIC_RESULTS = [
    Path("results/model_matrix_summary/dog_grass_by_model.csv"),
    Path("results/model_matrix_summary/final_contextneg_by_model.csv"),
    Path("results/model_matrix_summary/model_manifest.csv"),
    Path("results/model_matrix_summary/summary.md"),
    Path("results/human_sanity_table.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate lightweight demo outputs or validate frozen public result tables.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--small", action="store_true", help="Run a license-safe synthetic demo with no model download.")
    mode.add_argument("--full", action="store_true", help="Validate and fingerprint the frozen public result tables.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--model", default="text_bow_baseline")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.small:
        output = ROOT / (args.output or "results/sample_synthetic_demo")
        return run_small(output, args.model, args.batch_size)
    output = ROOT / (args.output or "results/reproducibility")
    return run_full(output)


def run_small(output: Path, model: str, batch_size: int) -> int:
    annotations = ROOT / "data" / "sample_synthetic" / "annotations.jsonl"
    if not annotations.exists():
        raise FileNotFoundError("data/sample_synthetic/annotations.jsonl is missing. Run scripts/generate_sample_synthetic_dataset.py.")
    start = time.perf_counter()
    rows = run_evaluation(model_name=model, annotations_path=annotations, output_dir=output, batch_size=batch_size, seed=0)
    elapsed = time.perf_counter() - start
    summary = {
        "mode": "small",
        "model": model,
        "n_rows": len(rows),
        "runtime_sec": round(elapsed, 4),
        "outputs": ["results.jsonl", "summary.csv"],
    }
    (output / "run_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Small reproduction complete: {output}")
    return 0


def run_full(output: Path) -> int:
    errors = validate_known_result_files(ROOT)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    output.mkdir(parents=True, exist_ok=True)
    fingerprints = []
    for relative_path in PUBLIC_RESULTS:
        path = ROOT / relative_path
        fingerprints.append({"path": relative_path.as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size})
    dog = pd.read_csv(ROOT / "results/model_matrix_summary/dog_grass_by_model.csv")
    final = pd.read_csv(ROOT / "results/model_matrix_summary/final_contextneg_by_model.csv")
    manifest = {
        "mode": "full",
        "description": "Frozen public result table validation and fingerprints.",
        "dog_grass_rows": int(len(dog)),
        "final_contextneg_rows": int(len(final)),
        "environment": environment_versions(),
        "fingerprints": fingerprints,
    }
    (output / "result_fingerprints.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Full result validation complete: {output}")
    return 0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
    }
    try:
        import torch

        versions["torch"] = torch.__version__
    except Exception:
        versions["torch"] = "not importable"
    try:
        import open_clip

        versions["open_clip"] = getattr(open_clip, "__version__", "unknown")
    except Exception:
        versions["open_clip"] = "not importable"
    return versions


if __name__ == "__main__":
    raise SystemExit(main())
