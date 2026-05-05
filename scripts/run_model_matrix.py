from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negcompbench.eval.context_neg_research_analysis import SCENARIO_CONFIGS
from negcompbench.eval.dog_grass_false_negation import run_dog_grass_false_negation_analysis
from negcompbench.eval.final_contextneg_analysis import run_final_contextneg_analysis
from negcompbench.eval.run_eval import MODEL_PRESETS


DEFAULT_MODELS = [
    "openclip_vit_b32",
    "openclip_vit_b32_openai",
    "openclip_rn50",
    "openclip_vit_b16_siglip",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the publication-oriented analyses across multiple model presets.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS, choices=sorted(MODEL_PRESETS))
    parser.add_argument("--analyses", nargs="+", default=["final", "dog_grass"], choices=["final", "dog_grass"])
    parser.add_argument("--scenarios", nargs="+", default=["kitchen_table", "street_car"], choices=sorted(SCENARIO_CONFIGS))
    parser.add_argument("--dog-root", default="data/context_neg/dog_grass_false_negation")
    parser.add_argument("--output-root", default="results/model_matrix")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    for model in args.models:
        model_output = ROOT / args.output_root / model
        if "final" in args.analyses:
            run_final_contextneg_analysis(
                scenarios=args.scenarios,
                model_name=model,
                output_dir=model_output / "final_contextneg_analysis",
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
                batch_size=args.batch_size,
                device=args.device,
            )
        if "dog_grass" in args.analyses:
            run_dog_grass_false_negation_analysis(
                root=ROOT / args.dog_root,
                model_name=model,
                output_dir=model_output / "dog_grass_false_negation",
                bootstrap_samples=args.bootstrap_samples,
                seed=args.seed,
                batch_size=args.batch_size,
                device=args.device,
            )
    print(f"Wrote model matrix results to {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
