from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".mplconfig").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_history(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _float_column(rows: list[dict[str, str]], name: str) -> list[float]:
    values = []
    for row in rows:
        raw = row.get(name, "")
        values.append(float(raw) if raw not in ("", "nan") else float("nan"))
    return values


def plot_history(run_dir: Path) -> None:
    rows = _read_history(run_dir / "history.csv")
    epochs = [int(row["epoch"]) for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes[0, 0].plot(epochs, _float_column(rows, "loss"), label="train loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("epoch")

    axes[0, 1].plot(epochs, _float_column(rows, "test_accuracy"), label="test accuracy")
    axes[0, 1].set_title("Test Accuracy")
    axes[0, 1].set_ylim(0.0, 1.0)
    axes[0, 1].set_xlabel("epoch")

    axes[1, 0].plot(epochs, _float_column(rows, "post_switch_recovery"))
    axes[1, 0].set_title("Post-Switch Recovery")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("steps")

    an_dist = _float_column(rows, "an_dist")
    n_move = _float_column(rows, "n_move")
    if any(value == value for value in an_dist):
        axes[1, 1].plot(epochs, an_dist, label="A/N distance")
        axes[1, 1].plot(epochs, n_move, label="N movement")
        axes[1, 1].legend()
    axes[1, 1].set_title("CPA Diagnostics")
    axes[1, 1].set_xlabel("epoch")

    fig.tight_layout()
    fig.savefig(run_dir / "history.png", dpi=160)
    plt.close(fig)


def plot_noise(run_dirs: list[Path], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for run_dir in run_dirs:
        with (run_dir / "metrics.json").open() as f:
            metrics = json.load(f)
        xs = []
        ys = []
        for noise, result in sorted(metrics["noise_sweep"].items(), key=lambda item: float(item[0])):
            xs.append(float(noise))
            ys.append(float(result["accuracy"]))
        label = metrics["args"]["model"]
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_title("Accuracy vs Noise")
    ax.set_xlabel("noise probability")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CPA experiment runs")
    parser.add_argument("run_dirs", type=Path, nargs="+")
    parser.add_argument("--noise-out", type=Path, default=Path("runs/noise_comparison.png"))
    args = parser.parse_args()
    for run_dir in args.run_dirs:
        plot_history(run_dir)
    if len(args.run_dirs) > 1:
        args.noise_out.parent.mkdir(parents=True, exist_ok=True)
        plot_noise(args.run_dirs, args.noise_out)


if __name__ == "__main__":
    main()
