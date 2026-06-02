from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CPA diff hidden-size capacity sweep")
    parser.add_argument("--out-dir", type=Path, default=Path("runs_capacity"))
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[64, 96, 128])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def run_hidden_dim(args: argparse.Namespace, hidden_dim: int) -> Path:
    out_dir = args.out_dir / f"h{hidden_dim}"
    cmd = [
        sys.executable,
        "train.py",
        "--model",
        "cpa",
        "--regime-set",
        "lagged",
        "--use-distractors",
        "--distractor-prob",
        "0.006",
        "--min-distractor",
        "20",
        "--max-distractor",
        "40",
        "--epochs",
        str(args.epochs),
        "--seed",
        str(args.seed),
        "--hidden-dim",
        str(hidden_dim),
        "--rho",
        "0.10",
        "--lambda-band",
        "0.001",
        "--no-detach-a-to-n",
        "--use-diff-to-n",
        "--stateful-eval",
        "--device",
        args.device,
        "--out-dir",
        str(out_dir),
    ]
    print(f"\n=== hidden_dim={hidden_dim} ===", flush=True)
    subprocess.run(cmd, check=True)
    return out_dir / f"cpa_seed{args.seed}"


def read_row(run_dir: Path, hidden_dim: int) -> dict[str, str | int | float]:
    with (run_dir / "metrics.json").open() as f:
        data = json.load(f)
    final = data["final"]
    stateful = data.get("stateful") or {}
    return {
        "hidden_dim": hidden_dim,
        "parameters": data["parameters"],
        "accuracy": final["accuracy"],
        "post_switch_recovery": final["post_switch_recovery"],
        "distractor_recovery": final["distractor_recovery"],
        "post_distractor_accuracy": final["post_distractor_accuracy"],
        "distractor_accuracy": final["distractor_accuracy"],
        "stateful_accuracy": stateful.get("accuracy", ""),
        "stateful_post_switch_recovery": stateful.get("post_switch_recovery", ""),
        "stateful_distractor_recovery": stateful.get("distractor_recovery", ""),
        "run_dir": str(run_dir),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for hidden_dim in args.hidden_dims:
        run_dir = run_hidden_dim(args, hidden_dim)
        rows.append(read_row(run_dir, hidden_dim))

    summary_path = args.out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    best = max(rows, key=lambda row: float(row["accuracy"]))
    print("\nBest by accuracy:")
    for key, value in best.items():
        print(f"{key}: {value}")
    print(f"\nwrote: {summary_path}")


if __name__ == "__main__":
    main()
