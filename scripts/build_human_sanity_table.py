from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negcompbench.data.human_sanity import build_human_sanity_table
from negcompbench.eval.context_neg_research_analysis import SCENARIO_CONFIGS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a compact human sanity table for reviewed ContextNeg datasets.")
    parser.add_argument("--scenarios", nargs="+", default=["kitchen_table", "street_car"], choices=sorted(SCENARIO_CONFIGS))
    parser.add_argument("--output", default="results/human_sanity_table.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = [
        {
            "scenario": name,
            "root": str(Path(SCENARIO_CONFIGS[name].annotations).parent),
            "scene": SCENARIO_CONFIGS[name].scene,
            "object": SCENARIO_CONFIGS[name].object_name,
        }
        for name in args.scenarios
    ]
    frame = build_human_sanity_table(scenarios, ROOT / args.output)
    print(f"Wrote {len(frame)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
