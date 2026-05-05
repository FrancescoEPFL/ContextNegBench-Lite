from __future__ import annotations

import csv
import mimetypes
import random
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from negcompbench.data.context_neg_dataset import (
    IMAGE_EXTENSIONS,
    SOURCE_COLUMNS,
    ensure_context_neg_structure,
    file_sha256,
    image_dimensions,
    make_review_gallery,
    negative_folder,
    normalize_filename,
    normalize_condition,
    positive_folder,
    read_csv_rows,
    reviewed_conditions,
    scan_supported_images,
    to_relative_path,
    unique_path,
    write_csv_rows,
    write_review_status,
)
from negcompbench.utils.io import ensure_dir

DOWNLOAD_LOG_COLUMNS = [
    "query",
    "condition",
    "source_url",
    "local_path",
    "action",
    "file_hash",
    "width",
    "height",
    "message",
]

TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
BACKOFF_SECONDS = [10.0, 30.0, 45.0]
DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) NegCompBench-Lite/0.1 local research dataset builder"


@dataclass(frozen=True)
class ImageCandidate:
    url: str
    query: str


@dataclass
class DownloadState:
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    unsupported: int = 0
    stop_requested: bool = False
    download_timestamps: list[float] = field(default_factory=list)

    def update(self, action: str) -> None:
        if action == "downloaded":
            self.downloaded += 1
        elif action == "skipped_duplicate":
            self.skipped += 1
        elif action == "unsupported":
            self.unsupported += 1
        elif action == "failed":
            self.failed += 1


def search_context_neg_images(
    root: str | Path,
    condition: str,
    queries: list[str],
    object_name: str = "table",
    scene: str = "kitchen",
    max_results_per_query: int = 25,
    limit_total: int = 80,
    url_file: str | Path | None = None,
    dry_run: bool = False,
    timeout: float = 15.0,
    sleep_seconds: float = 2.0,
    jitter_seconds: float = 1.0,
    max_retries: int = 2,
    resume: bool = False,
    stop_on_rate_limit: bool = False,
    max_downloads_per_minute: int = 20,
    user_agent: str = DEFAULT_USER_AGENT,
) -> list[dict[str, str]]:
    condition_folder = normalize_condition(condition, object_name)
    if condition_folder not in {positive_folder(object_name), negative_folder(object_name)}:
        raise ValueError(f"condition must map to with_object or without_object for object {object_name!r}")

    root_path = Path(root)
    ensure_context_neg_structure(root_path, object_name)
    candidates = collect_candidates(queries, max_results_per_query, limit_total, url_file)
    if dry_run:
        print_dry_run(candidates, limit_total)
        return []

    log_path = root_path / "metadata" / "image_download_log.csv"
    logged_urls, logged_hashes = logged_resume_keys(log_path) if resume else (set(), set())
    existing_hashes = reviewed_hashes(root_path, object_name) | logged_hashes
    destination = ensure_dir(root_path / "reviewed" / condition_folder)
    log_rows: list[dict[str, str]] = []
    state = DownloadState()
    blocked_queries: set[str] = set()

    for result_index, candidate in enumerate(candidates[:limit_total], start=1):
        if candidate.query in blocked_queries:
            continue
        if resume and candidate.url in logged_urls:
            row = log_row(candidate, condition, "skipped_duplicate", "", "", "", "", "Skipped by --resume: source_url already in log")
            append_download_log(log_path, [row])
            log_rows.append(row)
            state.update(row["action"])
            print_progress(candidate.query, result_index, state)
            continue

        enforce_download_rate(state, max_downloads_per_minute)
        sleep_between_requests(sleep_seconds, jitter_seconds)
        row = download_candidate(
            candidate=candidate,
            root=root_path,
            destination=destination,
            condition=condition_folder,
            existing_hashes=existing_hashes,
            timeout=timeout,
            max_retries=max_retries,
            stop_on_rate_limit=stop_on_rate_limit,
            user_agent=user_agent,
        )
        append_download_log(log_path, [row])
        log_rows.append(row)
        state.update(row["action"])
        if row["file_hash"]:
            logged_hashes.add(row["file_hash"])
        if is_rate_limit_row(row):
            if stop_on_rate_limit:
                state.stop_requested = True
                print("Rate limit detected. Re-run later with --resume.")
            elif repeated_rate_limit_for_query(log_rows, candidate.query):
                print(f"Repeated rate limit detected for query {candidate.query!r}; stopping current query.")
                blocked_queries.add(candidate.query)
        print_progress(candidate.query, result_index, state)
        if state.stop_requested:
            break

    update_sources_csv(root_path, log_rows)
    write_review_status(root_path, object_name)
    make_review_gallery(root_path, scene=scene, object_name=object_name)
    return log_rows


def collect_candidates(
    queries: list[str],
    max_results_per_query: int,
    limit_total: int,
    url_file: str | Path | None = None,
) -> list[ImageCandidate]:
    candidates: list[ImageCandidate] = []
    if url_file is not None:
        candidates.extend(read_url_file(url_file))
    if queries:
        candidates.extend(search_duckduckgo(queries, max_results_per_query))
    return unique_candidates(candidates)[:limit_total]


def read_url_file(path: str | Path) -> list[ImageCandidate]:
    candidates = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            value = line.strip().lstrip("\ufeff")
            if not value or value.startswith("#"):
                continue
            if "," in value:
                query, url = value.split(",", 1)
                candidates.append(ImageCandidate(url=url.strip(), query=query.strip() or "url_file"))
            else:
                candidates.append(ImageCandidate(url=value, query="url_file"))
    return candidates


def search_duckduckgo(queries: list[str], max_results_per_query: int) -> list[ImageCandidate]:
    DDGS = import_ddgs()
    candidates: list[ImageCandidate] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*duckduckgo_search.*renamed.*")
        with DDGS() as ddgs:
            for query in queries:
                try:
                    results = ddgs.images(
                        keywords=query,
                        max_results=max_results_per_query,
                        safesearch="moderate",
                    )
                except TypeError:
                    try:
                        results = ddgs.images(query, max_results=max_results_per_query)
                    except Exception as exc:
                        if looks_like_rate_limit(exc):
                            print("Rate limit detected during image search. Re-run later with --resume.")
                            break
                        raise RuntimeError(f"Image search failed for query {query!r}: {exc}") from exc
                except Exception as exc:
                    if looks_like_rate_limit(exc):
                        print("Rate limit detected during image search. Re-run later with --resume.")
                        break
                    raise RuntimeError(f"Image search failed for query {query!r}: {exc}") from exc
                for result in results:
                    url = result.get("image") or result.get("url") or result.get("thumbnail")
                    if url:
                        candidates.append(ImageCandidate(url=str(url), query=query))
    return candidates


def import_ddgs():
    try:
        from ddgs import DDGS

        return DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS

            return DDGS
        except ImportError as exc:
            raise RuntimeError(
                "DuckDuckGo image search requires an optional package.\n"
                "Install it with:\n\n"
                "  pip install ddgs\n\n"
                "or:\n\n"
                "  pip install duckduckgo_search\n\n"
                "Or use fallback mode with --url-file urls.txt."
            ) from exc


def looks_like_rate_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    return "ratelimit" in text or "rate limit" in text or "too many requests" in text or " 403 " in text or " 429 " in text


def unique_candidates(candidates: list[ImageCandidate]) -> list[ImageCandidate]:
    seen = set()
    unique = []
    for candidate in candidates:
        normalized = candidate.url.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(candidate)
    return unique


def reviewed_hashes(root: Path, object_name: str = "table") -> set[str]:
    hashes = set()
    for condition in reviewed_conditions(object_name):
        for path in scan_supported_images(root / "reviewed" / condition):
            hashes.add(file_sha256(path))
    return hashes


def download_candidate(
    candidate: ImageCandidate,
    root: Path,
    destination: Path,
    condition: str,
    existing_hashes: set[str],
    timeout: float,
    max_retries: int,
    stop_on_rate_limit: bool,
    user_agent: str,
) -> dict[str, str]:
    local_path = ""
    digest = ""
    width = ""
    height = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp_path = Path(temp.name)
        try:
            content_type, suggested_name = download_url_with_retries(
                candidate.url,
                temp_path,
                timeout=timeout,
                max_retries=max_retries,
                stop_on_rate_limit=stop_on_rate_limit,
                user_agent=user_agent,
            )
            suffix = infer_suffix(candidate.url, content_type)
            if suffix not in IMAGE_EXTENSIONS:
                return log_row(candidate, condition, "unsupported", "", "", "", "", f"Unsupported extension/content type: {suffix}")

            with Image.open(temp_path) as image:
                image.verify()
            digest = file_sha256(temp_path)
            if digest in existing_hashes:
                return log_row(candidate, condition, "skipped_duplicate", "", digest, "", "", "Duplicate content hash")

            filename = normalize_filename(suggested_name or url_filename(candidate.url) or f"{condition}_{digest[:12]}{suffix}")
            if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
                filename = f"{Path(filename).stem}{suffix}"
            target = unique_path(destination, filename)
            temp_path.replace(target)
            existing_hashes.add(digest)
            width_int, height_int = image_dimensions(target)
            width = str(width_int)
            height = str(height_int)
            local_path = to_relative_path(target)
            return log_row(candidate, condition, "downloaded", local_path, digest, width, height, "Downloaded")
        finally:
            if temp_path.exists():
                temp_path.unlink()
    except Exception as exc:
        return log_row(candidate, condition, "failed", local_path, digest, width, height, str(exc))


def download_url_with_retries(
    url: str,
    target: Path,
    timeout: float,
    max_retries: int,
    stop_on_rate_limit: bool,
    user_agent: str,
) -> tuple[str, str]:
    attempts = max_retries + 1
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return download_url(url, target, timeout, user_agent)
        except RateLimitError:
            if stop_on_rate_limit or attempt >= max_retries:
                raise
            time.sleep(backoff_seconds(attempt))
        except TransientDownloadError as exc:
            last_error = exc
            if attempt >= max_retries:
                raise
            time.sleep(backoff_seconds(attempt))
    if last_error is not None:
        raise last_error
    raise RuntimeError("Download failed for unknown reason")


def download_url(url: str, target: Path, timeout: float, user_agent: str) -> tuple[str, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "image/avif,image/webp,image/png,image/jpeg,image/*,*/*;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = getattr(response, "status", 200)
            if status in TRANSIENT_STATUS_CODES:
                raise transient_error(status, url)
            content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
            disposition = response.headers.get("Content-Disposition", "")
            suggested_name = filename_from_disposition(disposition) or url_filename(response.geturl())
            with target.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 256)
                    if not chunk:
                        break
                    handle.write(chunk)
            return content_type, suggested_name
    except urllib.error.HTTPError as exc:
        if exc.code == 429:
            raise RateLimitError(f"Rate limit detected for {url}: HTTP 429") from exc
        if exc.code in TRANSIENT_STATUS_CODES:
            raise TransientDownloadError(f"Transient HTTP {exc.code} for {url}") from exc
        raise RuntimeError(f"Download failed: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Download failed: {exc}") from exc


class RateLimitError(RuntimeError):
    pass


class TransientDownloadError(RuntimeError):
    pass


def transient_error(status: int, url: str) -> Exception:
    if status == 429:
        return RateLimitError(f"Rate limit detected for {url}: HTTP 429")
    return TransientDownloadError(f"Transient HTTP {status} for {url}")


def backoff_seconds(attempt: int) -> float:
    if attempt < len(BACKOFF_SECONDS):
        return BACKOFF_SECONDS[attempt]
    return BACKOFF_SECONDS[-1]


def infer_suffix(url: str, content_type: str) -> str:
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if suffix == ".jpeg":
        suffix = ".jpg"
    if suffix in IMAGE_EXTENSIONS:
        return suffix
    guessed = mimetypes.guess_extension(content_type or "")
    if guessed == ".jpe":
        guessed = ".jpg"
    if guessed == ".jpeg":
        guessed = ".jpg"
    return guessed or ""


def url_filename(url: str) -> str:
    path = urllib.parse.urlparse(url).path
    name = Path(urllib.parse.unquote(path)).name
    return name


def filename_from_disposition(disposition: str) -> str:
    marker = "filename="
    if marker not in disposition.lower():
        return ""
    _, value = disposition.split(marker, 1)
    return value.strip().strip('"')


def log_row(
    candidate: ImageCandidate,
    condition: str,
    action: str,
    local_path: str,
    file_hash: str,
    width: str,
    height: str,
    message: str,
) -> dict[str, str]:
    return {
        "query": candidate.query,
        "condition": condition,
        "source_url": candidate.url,
        "local_path": local_path,
        "action": action,
        "file_hash": file_hash,
        "width": width,
        "height": height,
        "message": message,
    }


def append_download_log(path: Path, rows: list[dict[str, str]]) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DOWNLOAD_LOG_COLUMNS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def logged_resume_keys(path: Path) -> tuple[set[str], set[str]]:
    urls: set[str] = set()
    hashes: set[str] = set()
    for row in read_csv_rows(path):
        if row.get("source_url"):
            urls.add(row["source_url"])
        if row.get("file_hash"):
            hashes.add(row["file_hash"])
    return urls, hashes


def sleep_between_requests(sleep_seconds: float, jitter_seconds: float) -> None:
    delay = max(0.0, sleep_seconds) + random.uniform(0.0, max(0.0, jitter_seconds))
    if delay > 0:
        time.sleep(delay)


def enforce_download_rate(state: DownloadState, max_downloads_per_minute: int) -> None:
    if max_downloads_per_minute <= 0:
        return
    now = time.monotonic()
    state.download_timestamps = [stamp for stamp in state.download_timestamps if now - stamp < 60.0]
    if len(state.download_timestamps) >= max_downloads_per_minute:
        wait = 60.0 - (now - state.download_timestamps[0])
        if wait > 0:
            time.sleep(wait)
    state.download_timestamps.append(time.monotonic())


def is_rate_limit_row(row: dict[str, str]) -> bool:
    message = row.get("message", "").lower()
    return "429" in message or "rate limit" in message or "too many requests" in message


def repeated_rate_limit_for_query(rows: list[dict[str, str]], query: str) -> bool:
    recent = [row for row in rows[-3:] if row["query"] == query]
    return len(recent) >= 2 and all(is_rate_limit_row(row) for row in recent)


def print_progress(query: str, result_index: int, state: DownloadState) -> None:
    print(
        f"query={query!r} result={result_index} "
        f"downloaded={state.downloaded} skipped={state.skipped} "
        f"failed={state.failed} unsupported={state.unsupported}"
    )


def update_sources_csv(root: Path, download_rows: list[dict[str, str]]) -> None:
    path = root / "metadata" / "sources.csv"
    existing = read_csv_rows(path)
    by_path = {row.get("relative_path", ""): row for row in existing if row.get("relative_path")}
    for row in download_rows:
        if row["action"] != "downloaded" or not row["local_path"]:
            continue
        by_path[row["local_path"]] = {
            "relative_path": row["local_path"],
            "original_path": "",
            "source_url": row["source_url"],
            "license": "",
            "condition_hint": row["condition"],
            "sha256": row["file_hash"],
            "width": row["width"],
            "height": row["height"],
        }
    write_csv_rows(path, SOURCE_COLUMNS, list(by_path.values()))


def print_dry_run(candidates: list[ImageCandidate], limit_total: int) -> None:
    print(f"Dry run: found {len(candidates)} unique candidate URLs. Showing up to {limit_total}.")
    for candidate in candidates[:limit_total]:
        print(f"- [{candidate.query}] {candidate.url}")
