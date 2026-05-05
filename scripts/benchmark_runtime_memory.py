from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negcompbench.eval.run_eval import run_evaluation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark runtime for a small evaluation run.")
    parser.add_argument("--annotations", default="data/sample_synthetic/annotations.jsonl")
    parser.add_argument("--model", default="text_bow_baseline")
    parser.add_argument("--output", default="results/runtime_benchmark")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = ROOT / args.output
    start = time.perf_counter()
    rows = run_evaluation(
        model_name=args.model,
        annotations_path=ROOT / args.annotations,
        output_dir=output,
        batch_size=args.batch_size,
        seed=0,
    )
    elapsed = time.perf_counter() - start
    payload = {
        "model": args.model,
        "annotations": args.annotations,
        "n_samples": len(rows),
        "batch_size": args.batch_size,
        "runtime_sec": round(elapsed, 4),
        "runtime_per_sample_sec": round(elapsed / max(len(rows), 1), 6),
        "note": "For OpenCLIP models, runtime includes local model loading if weights are not cached.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
