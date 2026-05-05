from __future__ import annotations

import html
import shutil
from pathlib import Path

import pandas as pd


def make_failure_gallery(
    results_path: str | Path,
    annotations_root: str | Path,
    output_dir: str | Path,
    max_failures: int = 50,
) -> Path:
    output = Path(output_dir)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_json(results_path, lines=True)
    failures = frame[~frame["is_correct"]].copy()
    failures = failures.sort_values("margin").head(max_failures)

    rows = []
    for _, row in failures.iterrows():
        src = resolve_image_path(Path(annotations_root), str(row["image_path"]))
        copied = images_dir / src.name
        if src.exists():
            shutil.copy2(src, copied)
            image_tag = f'<img src="images/{html.escape(copied.name)}" alt="{html.escape(row["image_id"])}">'
        else:
            image_tag = "<p>image missing</p>"

        rows.append(
            "<section>"
            f"{image_tag}"
            f"<h2>{html.escape(str(row['image_id']))}</h2>"
            f"<p><strong>Task:</strong> {html.escape(str(row['task_type']))}</p>"
            f"<p><strong>Failure type:</strong> {html.escape(str(row.get('failure_type', 'unknown')))}</p>"
            f"<p><strong>Correct:</strong> {html.escape(str(row['correct_caption']))}</p>"
            f"<p><strong>Selected:</strong> {html.escape(str(row['selected_caption']))}</p>"
            f"<p><strong>Score(correct):</strong> {float(row['score_correct']):.4f}</p>"
            f"<p><strong>Score(wrong):</strong> {float(row['max_negative_score']):.4f}</p>"
            f"<p><strong>Margin:</strong> {float(row['margin']):.4f}</p>"
            f"<p><strong>Model:</strong> {html.escape(str(row['model_name']))}</p>"
            "</section>"
        )

    gallery = output / "index.html"
    gallery.write_text(render_html(rows), encoding="utf-8")
    markdown = output / "failures.md"
    markdown.write_text(render_markdown(failures), encoding="utf-8")
    return gallery


def resolve_image_path(root: Path, image_path: str) -> Path:
    path = Path(image_path)
    return path if path.is_absolute() else root / path


def render_html(rows: list[str]) -> str:
    body = "\n".join(rows) if rows else "<p>No failures found.</p>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>NegCompBench-Lite Failure Gallery</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1d1d1d; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 18px; }}
    section {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; background: #fff; }}
    img {{ width: 100%; max-width: 224px; image-rendering: auto; display: block; margin-bottom: 10px; }}
    h1 {{ font-size: 24px; }}
    h2 {{ font-size: 16px; margin: 8px 0; }}
    p {{ margin: 6px 0; font-size: 14px; }}
  </style>
</head>
<body>
  <h1>NegCompBench-Lite Failure Gallery</h1>
  <main>
    {body}
  </main>
</body>
</html>
"""


def render_markdown(failures: pd.DataFrame) -> str:
    if failures.empty:
        return "# Failure Gallery\n\nNo failures found.\n"
    lines = ["# Failure Gallery", ""]
    for _, row in failures.iterrows():
        lines.extend(
            [
                f"## {row['image_id']}",
                "",
                f"- Task: {row['task_type']}",
                f"- Failure type: {row.get('failure_type', 'unknown')}",
                f"- Correct: {row['correct_caption']}",
                f"- Selected: {row['selected_caption']}",
                f"- Score(correct): {float(row['score_correct']):.4f}",
                f"- Score(wrong): {float(row['max_negative_score']):.4f}",
                f"- Margin: {float(row['margin']):.4f}",
                f"- Model: {row['model_name']}",
                "",
            ]
        )
    return "\n".join(lines)
