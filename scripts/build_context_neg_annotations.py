from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negcompbench.data.context_neg import build_context_neg_annotations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ContextNeg-Test annotations from reviewed images.")
    parser.add_argument("--root", required=True, help="Root folder containing reviewed/with_<object> and reviewed/without_<object>.")
    parser.add_argument("--scene", required=True, help="Scene name, e.g. street.")
    parser.add_argument("--object", required=True, dest="object_name", help="Object name, e.g. car.")
    parser.add_argument("--languages", "--language", nargs="+", default=["en"], help="Languages to include: en it or en,it")
    parser.add_argument("--scene-it", default=None, help="Optional Italian scene translation.")
    parser.add_argument("--object-it", default=None, help="Optional Italian object translation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = build_context_neg_annotations(
        root=args.root,
        scene=args.scene,
        object_name=args.object_name,
        languages=args.languages,
        scene_it=args.scene_it,
        object_it=args.object_it,
        cwd=ROOT,
    )
    print(f"Wrote {len(rows)} ContextNeg-Test annotations to {Path(args.root) / 'annotations.jsonl'}")


if __name__ == "__main__":
    main()
