from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negcompbench.eval.dog_grass_false_negation import dog_grass_prompt_pairs  # noqa: E402

TOKEN_RE = re.compile(r"[a-z]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute simple lexical baselines for dog/grass prompt pairs.")
    parser.add_argument("--output", default="results/lexical_bias_baselines/dog_grass_prompt_lexical_baselines.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for pair in dog_grass_prompt_pairs():
        candidates = {
            "false_negated": pair.false_negated,
            "partial_generic": pair.partial_generic,
            "true_detailed_generic": pair.true_detailed_generic,
            "positive_reference": pair.positive_reference,
        }
        for role, caption in candidates.items():
            tokens = TOKEN_RE.findall(caption.lower())
            rows.append(
                {
                    "pair_id": pair.pair_id,
                    "caption_role": role,
                    "caption_text": caption,
                    "token_count": len(tokens),
                    "contains_target_object": int("dog" in tokens),
                    "contains_scene_word": int(bool({"grass", "grassy", "field"}.intersection(tokens))),
                    "contains_negation_marker": int(bool({"no", "without"}.intersection(tokens))),
                }
            )
    frame = pd.DataFrame(rows)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Wrote lexical baselines to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
