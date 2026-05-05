from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negcompbench.eval.schema_validation import validate_known_result_files  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate public result CSV schemas.")
    parser.add_argument("--root", default=".")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    errors = validate_known_result_files(ROOT / args.root)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("Result schema validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
