from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negcompbench.data.context_neg_download import search_context_neg_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search/download candidate images for ContextNeg-Test human review.")
    parser.add_argument("--root", required=True, help="ContextNeg root, e.g. data/context_neg/kitchen_table.")
    parser.add_argument("--scene", default="kitchen")
    parser.add_argument("--object", default="table", dest="object_name")
    parser.add_argument("--condition", required=True, help="with_object, without_object, or explicit folder alias such as with_car.")
    parser.add_argument("--queries", nargs="*", default=[], help="Image search queries.")
    parser.add_argument("--max-results-per-query", type=int, default=15)
    parser.add_argument("--limit-total", type=int, default=25)
    parser.add_argument("--url-file", default=None, help="Fallback text file with image URLs, one per line, or query,url rows.")
    parser.add_argument("--dry-run", action="store_true", help="Print candidate URLs without downloading or writing logs.")
    parser.add_argument("--timeout-seconds", "--timeout", type=float, default=15.0)
    parser.add_argument("--sleep-seconds", type=float, default=2.0)
    parser.add_argument("--jitter-seconds", type=float, default=1.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-rate-limit", action="store_true")
    parser.add_argument("--max-downloads-per-minute", type=int, default=20)
    parser.add_argument(
        "--user-agent",
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) NegCompBench-Lite/0.1 local research dataset builder",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.queries and args.url_file is None:
        print("Provide --queries, --url-file, or both.", file=sys.stderr)
        return 2
    try:
        rows = search_context_neg_images(
            root=args.root,
            condition=args.condition,
            queries=args.queries,
            scene=args.scene,
            object_name=args.object_name,
            max_results_per_query=args.max_results_per_query,
            limit_total=args.limit_total,
            url_file=args.url_file,
            dry_run=args.dry_run,
            timeout=args.timeout_seconds,
            sleep_seconds=args.sleep_seconds,
            jitter_seconds=args.jitter_seconds,
            max_retries=args.max_retries,
            resume=args.resume,
            stop_on_rate_limit=args.stop_on_rate_limit,
            max_downloads_per_minute=args.max_downloads_per_minute,
            user_agent=args.user_agent,
        )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.dry_run:
        return 0

    downloaded = len([row for row in rows if row["action"] == "downloaded"])
    duplicates = len([row for row in rows if row["action"] == "skipped_duplicate"])
    failed = len([row for row in rows if row["action"] == "failed"])
    unsupported = len([row for row in rows if row["action"] == "unsupported"])
    print(f"Downloaded: {downloaded}")
    print(f"Skipped duplicates: {duplicates}")
    print(f"Failed: {failed}")
    print(f"Unsupported: {unsupported}")
    print(f"Log: {Path(args.root) / 'metadata' / 'image_download_log.csv'}")
    print(f"Review gallery: {Path(args.root) / 'review_gallery.html'}")
    print("\nNext commands:")
    scenario = Path(args.root).name
    print(f"python scripts/check_context_neg_dataset.py --root {args.root} --scene {args.scene} --object {args.object_name}")
    print(
        "python scripts/build_context_neg_annotations.py "
        f"--root {args.root} --scene {args.scene} --object {args.object_name} --languages en"
    )
    print(
        "python scripts/run_context_neg_eval.py "
        f"--annotations {Path(args.root) / 'annotations.jsonl'} "
        "--model openclip_vit_b32 "
        f"--output results/context_neg/{scenario} "
        "--languages en --batch-size 8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
