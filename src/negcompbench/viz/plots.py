from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def make_plots(results_path: str | Path, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_json(results_path, lines=True)
    if frame.empty:
        return

    accuracy = frame.groupby("task_type")["is_correct"].mean().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    accuracy.plot(kind="bar", ax=ax, color="#3568a8")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Task")
    ax.set_title("Accuracy by Task")
    fig.tight_layout()
    fig.savefig(output / "accuracy_by_task.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    frame["margin"].astype(float).hist(ax=ax, bins=24, color="#d17b35")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Score(correct) - max score(negative)")
    ax.set_ylabel("Samples")
    ax.set_title("Margin Distribution")
    fig.tight_layout()
    fig.savefig(output / "margin_histogram.png", dpi=160)
    plt.close(fig)
