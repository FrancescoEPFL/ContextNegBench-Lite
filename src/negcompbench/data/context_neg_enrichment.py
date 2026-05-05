from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from negcompbench.data.context_neg_dataset import (
    ensure_context_neg_structure,
    file_sha256,
    make_review_gallery,
    normalize_condition,
    read_csv_rows,
    reviewed_conditions,
    scan_supported_images,
)
from negcompbench.data.context_neg_download import (
    DEFAULT_USER_AGENT,
    DownloadState,
    ImageCandidate,
    append_download_log,
    collect_candidates,
    download_candidate,
    enforce_download_rate,
    is_rate_limit_row,
    logged_resume_keys,
    log_row,
    repeated_rate_limit_for_query,
    sleep_between_requests,
    update_sources_csv,
)
from negcompbench.utils.io import ensure_dir


ENRICHMENT_LOG_COLUMNS = [
    "timestamp",
    "scenario_id",
    "root",
    "scene",
    "object",
    "condition",
    "query",
    "source_url",
    "local_path",
    "action",
    "file_hash",
    "width",
    "height",
    "message",
]


@dataclass(frozen=True)
class EnrichmentJob:
    scenario_id: str
    root: str
    scene: str
    object_name: str
    condition: str
    queries: tuple[str, ...]

    @property
    def condition_folder(self) -> str:
        return normalize_condition(self.condition, self.object_name)


@dataclass
class EnrichmentJobSummary:
    scenario_id: str
    condition: str
    root: str
    scene: str
    object_name: str
    destination: str
    count_before: int
    count_after: int = 0
    attempted: int = 0
    downloaded: int = 0
    skipped_duplicates: int = 0
    failures: int = 0
    rate_limited: bool = False


@dataclass
class EnrichmentRunSummary:
    start_time: datetime
    end_time: datetime | None = None
    target_per_folder: int = 250
    batch_size_per_job: int = 25
    dry_run: bool = False
    rate_limited: bool = False
    jobs: list[EnrichmentJobSummary] = field(default_factory=list)


DEFAULT_ENRICHMENT_JOBS = [
    EnrichmentJob(
        scenario_id="kitchen_table",
        root="data/context_neg/kitchen_table",
        scene="kitchen",
        object_name="table",
        condition="with_object",
        queries=(
            "kitchen with dining table",
            "kitchen with table and chairs",
            "modern kitchen with dining table",
            "kitchen dining area",
            "small kitchen with dining table",
            "open kitchen with dining table",
        ),
    ),
    EnrichmentJob(
        scenario_id="kitchen_table",
        root="data/context_neg/kitchen_table",
        scene="kitchen",
        object_name="table",
        condition="without_object",
        queries=(
            "galley kitchen",
            "compact kitchen interior",
            "small kitchen interior",
            "modern kitchen without table",
            "kitchen cabinets interior",
            "kitchen interior no dining table",
            "small kitchen no dining table",
        ),
    ),
    EnrichmentJob(
        scenario_id="street_car",
        root="data/context_neg/street_car",
        scene="street",
        object_name="car",
        condition="with_object",
        queries=(
            "street with cars",
            "city street with cars",
            "urban street traffic",
            "road with cars",
            "cars parked on street",
            "busy street with cars",
        ),
    ),
    EnrichmentJob(
        scenario_id="street_car",
        root="data/context_neg/street_car",
        scene="street",
        object_name="car",
        condition="without_object",
        queries=(
            "empty street no cars",
            "street without cars",
            "empty city street",
            "quiet street no cars",
            "road without cars",
            "empty road no cars",
            "deserted street no cars",
        ),
    ),
    EnrichmentJob(
        scenario_id="cat_sofa",
        root="data/context_neg/cat_sofa",
        scene="living room",
        object_name="cat",
        condition="with_object",
        queries=(
            "cat on sofa living room",
            "cat sitting on couch",
            "living room sofa with cat",
            "cat sleeping on sofa",
            "cat on couch indoors",
        ),
    ),
    EnrichmentJob(
        scenario_id="cat_sofa",
        root="data/context_neg/cat_sofa",
        scene="living room",
        object_name="cat",
        condition="without_object",
        queries=(
            "living room sofa no cat",
            "empty sofa living room",
            "living room couch without pets",
            "clean sofa living room",
            "modern living room sofa",
        ),
    ),
    EnrichmentJob(
        scenario_id="person_beach",
        root="data/context_neg/person_beach",
        scene="beach",
        object_name="person",
        condition="with_object",
        queries=(
            "person on beach",
            "beach with one person",
            "people walking on beach",
            "person standing on beach",
            "beach portrait full body",
        ),
    ),
    EnrichmentJob(
        scenario_id="person_beach",
        root="data/context_neg/person_beach",
        scene="beach",
        object_name="person",
        condition="without_object",
        queries=(
            "empty beach no people",
            "beach without people",
            "deserted beach",
            "empty sandy beach",
            "quiet beach no people",
        ),
    ),
    EnrichmentJob(
        scenario_id="bicycle_street",
        root="data/context_neg/bicycle_street",
        scene="street",
        object_name="bicycle",
        condition="with_object",
        queries=(
            "bicycle on street",
            "bike parked on street",
            "city street with bicycle",
            "person riding bicycle on street",
            "bicycle lane street",
        ),
    ),
    EnrichmentJob(
        scenario_id="bicycle_street",
        root="data/context_neg/bicycle_street",
        scene="street",
        object_name="bicycle",
        condition="without_object",
        queries=(
            "empty street no bicycles",
            "street without bicycles",
            "city street no bikes",
            "quiet street without bikes",
            "urban street no bicycle",
        ),
    ),
]


def enrich_context_neg_datasets(
    target_per_folder: int = 250,
    batch_size_per_job: int = 25,
    sleep_seconds: float = 3.0,
    jitter_seconds: float = 2.0,
    max_downloads_per_minute: int = 12,
    max_results_per_query: int = 15,
    timeout_seconds: float = 20.0,
    max_retries: int = 2,
    resume: bool = False,
    stop_on_rate_limit: bool = False,
    dry_run: bool = False,
    user_agent: str = DEFAULT_USER_AGENT,
    jobs: list[EnrichmentJob] | None = None,
    base_dir: str | Path = ".",
) -> EnrichmentRunSummary:
    base_path = Path(base_dir)
    active_jobs = jobs or DEFAULT_ENRICHMENT_JOBS
    for job in active_jobs:
        ensure_context_neg_structure(base_path / job.root, job.object_name)

    log_path = base_path / "data" / "context_neg" / "enrichment_log.csv"
    summary_path = base_path / "data" / "context_neg" / "enrichment_run_summary.md"
    run_summary = EnrichmentRunSummary(
        start_time=datetime.now(),
        target_per_folder=target_per_folder,
        batch_size_per_job=batch_size_per_job,
        dry_run=dry_run,
    )
    global_urls, global_hashes = enrichment_resume_keys(log_path, active_jobs, base_path) if resume else (set(), set())
    global_hashes |= global_reviewed_hashes(active_jobs, base_path)
    state = DownloadState()
    recent_rows: list[dict[str, str]] = []

    try:
        round_index = 0
        while True:
            round_index += 1
            round_downloaded = 0
            all_at_target = True
            print(f"\n=== Enrichment round {round_index} ===")
            for job in active_jobs:
                root = base_path / job.root
                destination = root / "reviewed" / job.condition_folder
                count_before = count_images(destination)
                to_download = max(0, min(batch_size_per_job, target_per_folder - count_before))
                job_summary = EnrichmentJobSummary(
                    scenario_id=job.scenario_id,
                    condition=job.condition_folder,
                    root=job.root,
                    scene=job.scene,
                    object_name=job.object_name,
                    destination=destination.as_posix(),
                    count_before=count_before,
                    count_after=count_before,
                )
                run_summary.jobs.append(job_summary)
                print(f"\n[{job.scenario_id}/{job.condition_folder}] current={count_before} target={target_per_folder} batch={to_download}")
                if to_download <= 0:
                    print("Already at target; skipping.")
                    regenerate_gallery_and_print_check(root, job)
                    continue
                all_at_target = False

                rotated_queries = rotate_queries(job.queries, read_csv_rows(log_path), job)
                try:
                    candidates = collect_enrichment_candidates(
                        rotated_queries,
                        max_results_per_query,
                        max(to_download * 4, to_download),
                        sleep_seconds=sleep_seconds,
                        jitter_seconds=jitter_seconds,
                    )
                except RuntimeError as exc:
                    row = enriched_log_row(
                        job,
                        ImageCandidate("", rotated_queries[0] if rotated_queries else ""),
                        "rate_limited",
                        "",
                        "",
                        "",
                        "",
                        str(exc),
                    )
                    append_enrichment_log(log_path, [row])
                    recent_rows.append(row)
                    job_summary.rate_limited = True
                    run_summary.rate_limited = True
                    print("Rate limit detected. Re-run later with --resume.")
                    break

                downloaded_for_job = 0
                for index, candidate in enumerate(candidates, start=1):
                    if downloaded_for_job >= to_download:
                        break
                    if resume and candidate.url in global_urls:
                        row = enriched_log_row(
                            job,
                            candidate,
                            "skipped_duplicate_url",
                            "",
                            "",
                            "",
                            "",
                            "Skipped by --resume: source_url already in enrichment log",
                        )
                        append_enrichment_log(log_path, [row])
                        recent_rows.append(row)
                        update_job_summary(job_summary, row)
                        print_enrichment_progress(job, index, job_summary)
                        continue
                    if dry_run:
                        row = enriched_log_row(job, candidate, "dry_run", "", "", "", "", "Dry run: candidate not downloaded")
                        append_enrichment_log(log_path, [row])
                        recent_rows.append(row)
                        update_job_summary(job_summary, row)
                        print_enrichment_progress(job, index, job_summary)
                        if job_summary.attempted >= to_download:
                            break
                        continue

                    enforce_download_rate(state, max_downloads_per_minute)
                    sleep_between_requests(sleep_seconds, jitter_seconds)
                    raw_row = download_candidate(
                        candidate=candidate,
                        root=root,
                        destination=destination,
                        condition=job.condition_folder,
                        existing_hashes=global_hashes,
                        timeout=timeout_seconds,
                        max_retries=max_retries,
                        stop_on_rate_limit=stop_on_rate_limit,
                        user_agent=user_agent,
                    )
                    row = enriched_log_row(
                        job,
                        candidate,
                        normalize_download_action(raw_row),
                        raw_row["local_path"],
                        raw_row["file_hash"],
                        raw_row["width"],
                        raw_row["height"],
                        raw_row["message"],
                    )
                    append_enrichment_log(log_path, [row])
                    append_download_log(root / "metadata" / "image_download_log.csv", [download_log_row(row)])
                    recent_rows.append(row)
                    update_job_summary(job_summary, row)
                    global_urls.add(candidate.url)
                    if row["file_hash"]:
                        global_hashes.add(row["file_hash"])
                    if row["action"] == "downloaded":
                        downloaded_for_job += 1
                    if is_rate_limited_enrichment_row(row):
                        job_summary.rate_limited = True
                        run_summary.rate_limited = True
                        print("Rate limit detected. Re-run later with --resume.")
                        if stop_on_rate_limit or repeated_rate_limit_for_query(download_log_rows(recent_rows), candidate.query):
                            break
                    print_enrichment_progress(job, index, job_summary)
                update_sources_csv(root, [download_log_row(row) for row in recent_rows if row["scenario_id"] == job.scenario_id])
                job_summary.count_after = count_images(destination)
                round_downloaded += job_summary.downloaded
                regenerate_gallery_and_print_check(root, job)
                if run_summary.rate_limited and stop_on_rate_limit:
                    break
            if all_at_target:
                break
            if dry_run:
                break
            if run_summary.rate_limited:
                break
            if round_downloaded == 0:
                print("No new images were downloaded in this round; stopping to avoid repeated duplicate/search loops.")
                break
    finally:
        run_summary.end_time = datetime.now()
        write_run_summary(summary_path, run_summary)
        print(f"\nWrote global enrichment log to {log_path}")
        print(f"Wrote run summary to {summary_path}")
    return run_summary


def count_images(path: str | Path) -> int:
    return len(scan_supported_images(path))


def collect_enrichment_candidates(
    queries: list[str],
    max_results_per_query: int,
    limit_total: int,
    sleep_seconds: float = 3.0,
    jitter_seconds: float = 2.0,
) -> list[ImageCandidate]:
    candidates: list[ImageCandidate] = []
    for index, query in enumerate(queries):
        if index > 0:
            sleep_between_requests(sleep_seconds, jitter_seconds)
        candidates.extend(collect_candidates([query], max_results_per_query, max_results_per_query))
        if len(candidates) >= limit_total:
            break
    return unique_by_url(candidates)[:limit_total]


def unique_by_url(candidates: list[ImageCandidate]) -> list[ImageCandidate]:
    seen: set[str] = set()
    unique: list[ImageCandidate] = []
    for candidate in candidates:
        if candidate.url in seen:
            continue
        seen.add(candidate.url)
        unique.append(candidate)
    return unique


def rotate_queries(queries: tuple[str, ...], log_rows: list[dict[str, str]], job: EnrichmentJob) -> list[str]:
    attempts = {query: 0 for query in queries}
    for row in log_rows:
        if row.get("scenario_id") == job.scenario_id and row.get("condition") == job.condition_folder and row.get("query") in attempts:
            attempts[row["query"]] += 1
    return sorted(queries, key=lambda query: (attempts[query], queries.index(query)))


def enrichment_resume_keys(path: Path, jobs: list[EnrichmentJob], base_path: Path) -> tuple[set[str], set[str]]:
    urls, hashes = logged_resume_keys(path)
    seen_roots: set[str] = set()
    for job in jobs:
        if job.root in seen_roots:
            continue
        seen_roots.add(job.root)
        root = base_path / job.root
        root_urls, root_hashes = logged_resume_keys(root / "metadata" / "image_download_log.csv")
        urls |= root_urls
        hashes |= root_hashes
        for row in read_csv_rows(root / "metadata" / "sources.csv"):
            if row.get("source_url"):
                urls.add(row["source_url"])
            if row.get("sha256"):
                hashes.add(row["sha256"])
    return urls, hashes


def global_reviewed_hashes(jobs: list[EnrichmentJob], base_path: Path) -> set[str]:
    hashes: set[str] = set()
    seen_roots: set[tuple[str, str]] = set()
    for job in jobs:
        key = (job.root, job.object_name)
        if key in seen_roots:
            continue
        seen_roots.add(key)
        root = base_path / job.root
        for folder in reviewed_conditions(job.object_name):
            for path in scan_supported_images(root / "reviewed" / folder):
                hashes.add(file_sha256(path))
    return hashes


def normalize_download_action(row: dict[str, str]) -> str:
    action = row["action"]
    message = row.get("message", "")
    if is_rate_limit_row(row):
        return "rate_limited"
    if action == "downloaded":
        return "downloaded"
    if action == "skipped_duplicate":
        return "skipped_duplicate_hash"
    if action == "unsupported":
        return "skipped_unsupported"
    if action == "failed" and looks_like_decode_failure(message):
        return "failed_decode"
    return "failed_download"


def looks_like_decode_failure(message: str) -> bool:
    lowered = message.lower()
    return "cannot identify image file" in lowered or "truncated" in lowered or "image.verify" in lowered


def enriched_log_row(
    job: EnrichmentJob,
    candidate: ImageCandidate,
    action: str,
    local_path: str,
    file_hash: str,
    width: str,
    height: str,
    message: str,
) -> dict[str, str]:
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "scenario_id": job.scenario_id,
        "root": job.root,
        "scene": job.scene,
        "object": job.object_name,
        "condition": job.condition_folder,
        "query": candidate.query,
        "source_url": candidate.url,
        "local_path": local_path,
        "action": action,
        "file_hash": file_hash,
        "width": width,
        "height": height,
        "message": message,
    }


def append_enrichment_log(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ENRICHMENT_LOG_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def download_log_row(row: dict[str, str]) -> dict[str, str]:
    action_map = {
        "downloaded": "downloaded",
        "skipped_duplicate_hash": "skipped_duplicate",
        "skipped_duplicate_url": "skipped_duplicate",
        "skipped_unsupported": "unsupported",
        "failed_download": "failed",
        "failed_decode": "failed",
        "rate_limited": "failed",
        "dry_run": "failed",
    }
    return log_row(
        ImageCandidate(row["source_url"], row["query"]),
        row["condition"],
        action_map.get(row["action"], "failed"),
        row["local_path"],
        row["file_hash"],
        row["width"],
        row["height"],
        row["message"],
    )


def download_log_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [download_log_row(row) for row in rows]


def update_job_summary(summary: EnrichmentJobSummary, row: dict[str, str]) -> None:
    summary.attempted += 1
    if row["action"] == "downloaded":
        summary.downloaded += 1
    elif row["action"] in {"skipped_duplicate_url", "skipped_duplicate_hash"}:
        summary.skipped_duplicates += 1
    elif row["action"] in {"failed_download", "failed_decode", "skipped_unsupported"}:
        summary.failures += 1
    elif row["action"] == "rate_limited":
        summary.failures += 1
        summary.rate_limited = True


def is_rate_limited_enrichment_row(row: dict[str, str]) -> bool:
    return row["action"] == "rate_limited" or "rate limit" in row.get("message", "").lower() or "429" in row.get("message", "")


def print_enrichment_progress(job: EnrichmentJob, index: int, summary: EnrichmentJobSummary) -> None:
    print(
        f"scenario={job.scenario_id!r} condition={job.condition_folder!r} result={index} "
        f"downloaded={summary.downloaded} skipped_duplicates={summary.skipped_duplicates} failures={summary.failures}"
    )


def regenerate_gallery_and_print_check(root: Path, job: EnrichmentJob) -> None:
    make_review_gallery(root, scene=job.scene, object_name=job.object_name)
    print(f"Review gallery: {root / 'review_gallery.html'}")
    print(f"Check command: python scripts/check_context_neg_dataset.py --root {job.root} --scene {job.scene} --object {job.object_name}")


def write_run_summary(path: Path, summary: EnrichmentRunSummary) -> None:
    end_time = summary.end_time or datetime.now()
    lines = [
        "# ContextNeg Dataset Enrichment Run Summary",
        "",
        f"- Start time: `{summary.start_time.isoformat(timespec='seconds')}`",
        f"- End time: `{end_time.isoformat(timespec='seconds')}`",
        f"- Target per folder: `{summary.target_per_folder}`",
        f"- Batch size per job: `{summary.batch_size_per_job}`",
        f"- Dry run: `{summary.dry_run}`",
        f"- Rate limit occurred: `{summary.rate_limited}`",
        "",
        "## Jobs",
        "",
        "| scenario_id | condition | before | after | downloaded | skipped duplicates | failures | rate limited |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for job in summary.jobs:
        lines.append(
            f"| {job.scenario_id} | {job.condition} | {job.count_before} | {job.count_after} | "
            f"{job.downloaded} | {job.skipped_duplicates} | {job.failures} | {job.rate_limited} |"
        )
    lines.extend(
        [
            "",
            "## Resume",
            "",
            "```powershell",
            "python scripts/enrich_context_neg_datasets.py --target-per-folder 250 --batch-size-per-job 25 --sleep-seconds 3 --jitter-seconds 2 --max-downloads-per-minute 12 --resume --stop-on-rate-limit",
            "```",
            "",
        ]
    )
    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")
