from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negcompbench.data.context_neg_dataset import make_review_gallery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a static ContextNeg-Test human review gallery.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--scene", default="kitchen")
    parser.add_argument("--object", default="table", dest="object_name")
    parser.add_argument("--output", default=None, help="Optional output HTML path. Defaults to root/review_gallery.html.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gallery = make_review_gallery(args.root, args.output, scene=args.scene, object_name=args.object_name)
    print(f"Wrote review gallery to {gallery}")


if __name__ == "__main__":
    main()
