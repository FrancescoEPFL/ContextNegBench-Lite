from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negcompbench.eval.dog_grass_false_negation import run_dog_grass_false_negation_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the dog/grass false-negation diagnostic.")
    parser.add_argument("--root", default="data/context_neg/dog_grass_false_negation")
    parser.add_argument("--model", default="openclip_vit_b32")
    parser.add_argument("--output", default="results/dog_grass_false_negation")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dog_grass_false_negation_analysis(
        root=args.root,
        model_name=args.model,
        output_dir=args.output,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"Wrote dog/grass false-negation analysis to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
