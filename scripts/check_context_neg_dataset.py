from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negcompbench.data.context_neg_dataset import check_context_neg_dataset, collect_image_records, negative_folder, positive_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a human-reviewed ContextNeg-Test dataset.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--scene", default="kitchen")
    parser.add_argument("--object", default="table", dest="object_name")
    parser.add_argument("--min-per-condition", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = collect_image_records(args.root, args.object_name)
    warnings, report = check_context_neg_dataset(args.root, min_per_condition=args.min_per_condition, object_name=args.object_name)
    pos = positive_folder(args.object_name)
    neg = negative_folder(args.object_name)
    reviewed_with = len([record for record in records if record.area == "reviewed" and record.condition == pos])
    reviewed_without = len([record for record in records if record.area == "reviewed" and record.condition == neg])
    rejected = len([record for record in records if record.area == "reviewed" and record.condition == "rejected"])
    print("ContextNeg-Test dataset summary")
    print(f"- reviewed/{pos}: {reviewed_with}")
    print(f"- reviewed/{neg}: {reviewed_without}")
    print(f"- reviewed/rejected: {rejected}")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("\nNo warnings.")
    print(f"\nWrote dataset report to {report}")


if __name__ == "__main__":
    main()
