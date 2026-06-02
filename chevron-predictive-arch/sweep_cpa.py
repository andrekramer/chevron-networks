from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact CPA diagnostic sweep")
    parser.add_argument("--out-dir", type=Path, default=Path("runs_lagged_sweep"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-sequences", type=int, default=64)
    parser.add_argument("--test-sequences", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=1200)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def run_config(args: argparse.Namespace, *, rho: float, lambda_band: float, detach: bool) -> Path:
    name = f"rho{rho:g}_band{lambda_band:g}_{'detach' if detach else 'nodetach'}"
    run_out = args.out_dir / name
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
        "--train-sequences",
        str(args.train_sequences),
        "--test-sequences",
        str(args.test_sequences),
        "--sequence-length",
        str(args.sequence_length),
        "--context-length",
        str(args.context_length),
        "--hidden-dim",
        str(args.hidden_dim),
        "--batch-size",
        str(args.batch_size),
        "--rho",
        str(rho),
        "--lambda-band",
        str(lambda_band),
        "--stateful-eval",
        "--device",
        args.device,
        "--out-dir",
        str(run_out),
    ]
    cmd.append("--detach-a-to-n" if detach else "--no-detach-a-to-n")
    print(f"\n=== {name} ===", flush=True)
    subprocess.run(cmd, check=True)
    return run_out / "cpa_seed0"


def read_summary(run_dir: Path, *, rho: float, lambda_band: float, detach: bool) -> dict[str, float | str | bool]:
    with (run_dir / "metrics.json").open() as f:
        metrics = json.load(f)
    with (run_dir / "history.csv").open() as f:
        history = list(csv.DictReader(f))
    last = history[-1]
    final = metrics["final"]
    stateful = metrics.get("stateful") or {}
    return {
        "rho": rho,
        "lambda_band": lambda_band,
        "detach_a_to_n": detach,
        "accuracy": final["accuracy"],
        "switch_recovery": final["post_switch_recovery"],
        "distractor_recovery": final["distractor_recovery"],
        "post_distractor_accuracy": final["post_distractor_accuracy"],
        "distractor_accuracy": final["distractor_accuracy"],
        "stateful_accuracy": stateful.get("accuracy", ""),
        "stateful_switch_recovery": stateful.get("post_switch_recovery", ""),
        "an_dist": last.get("an_dist", ""),
        "an_cos": last.get("an_cos", ""),
        "a_move": last.get("a_move", ""),
        "n_move": last.get("n_move", ""),
        "run_dir": str(run_dir),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        (0.05, 0.0, True),
        (0.05, 0.001, True),
        (0.10, 0.0, True),
        (0.10, 0.001, True),
        (0.20, 0.0, True),
        (0.20, 0.001, True),
        (0.10, 0.001, False),
        (0.20, 0.001, False),
    ]
    rows = []
    for rho, lambda_band, detach in configs:
        run_dir = run_config(args, rho=rho, lambda_band=lambda_band, detach=detach)
        rows.append(read_summary(run_dir, rho=rho, lambda_band=lambda_band, detach=detach))

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
