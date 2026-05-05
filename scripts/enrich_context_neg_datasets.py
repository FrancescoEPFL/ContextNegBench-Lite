from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from negcompbench.data.context_neg_download import DEFAULT_USER_AGENT
from negcompbench.data.context_neg_enrichment import enrich_context_neg_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely enrich all ContextNeg reviewed image folders with candidate images.")
    parser.add_argument("--target-per-folder", type=int, default=250)
    parser.add_argument("--batch-size-per-job", type=int, default=25)
    parser.add_argument("--sleep-seconds", type=float, default=3.0)
    parser.add_argument("--jitter-seconds", type=float, default=2.0)
    parser.add_argument("--max-downloads-per-minute", type=int, default=12)
    parser.add_argument("--max-results-per-query", type=int, default=15)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-rate-limit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = enrich_context_neg_datasets(
        target_per_folder=args.target_per_folder,
        batch_size_per_job=args.batch_size_per_job,
        sleep_seconds=args.sleep_seconds,
        jitter_seconds=args.jitter_seconds,
        max_downloads_per_minute=args.max_downloads_per_minute,
        max_results_per_query=args.max_results_per_query,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        resume=args.resume,
        stop_on_rate_limit=args.stop_on_rate_limit,
        dry_run=args.dry_run,
        user_agent=args.user_agent,
        base_dir=ROOT,
    )
    if summary.rate_limited:
        print("\nRate limit detected. Re-run later with --resume.")
        return 0
    print("\nEnrichment pass complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
