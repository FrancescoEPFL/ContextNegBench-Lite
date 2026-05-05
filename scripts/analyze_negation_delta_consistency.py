from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negcompbench.eval.negation_delta_consistency import run_negation_delta_consistency_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze whether CLIP text negation deltas are consistent across objects.")
    parser.add_argument("--model", default="openclip_vit_b32")
    parser.add_argument("--output", default="results/negation_delta_consistency")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_negation_delta_consistency_analysis(
        model_name=args.model,
        output_dir=args.output,
        seed=args.seed,
        device=args.device,
    )
    print(f"Wrote negation delta consistency analysis to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
