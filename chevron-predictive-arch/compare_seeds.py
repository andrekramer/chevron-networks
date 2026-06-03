from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path


CONDITIONS = {
    "mlp": ["--model", "mlp"],
    "transformer": ["--model", "transformer"],
    "cpa_tuned": [
        "--model",
        "cpa",
        "--rho",
        "0.10",
        "--lambda-band",
        "0.001",
        "--no-detach-a-to-n",
        "--stateful-eval",
    ],
    "cpa_diff": [
        "--model",
        "cpa",
        "--rho",
        "0.10",
        "--lambda-band",
        "0.001",
        "--no-detach-a-to-n",
        "--use-diff-to-n",
        "--stateful-eval",
    ],
}


METRICS = (
    "accuracy",
    "post_switch_recovery",
    "distractor_recovery",
    "post_distractor_accuracy",
    "distractor_accuracy",
    "switch_accuracy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare key models across random seeds")
    parser.add_argument("--out-dir", type=Path, default=Path("runs_seed_compare"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--mlp-hidden-dim", type=int, default=64)
    parser.add_argument("--cpa-hidden-dim", type=int, default=64)
    return parser.parse_args()


def run_condition(args: argparse.Namespace, condition: str, seed: int) -> Path:
    condition_out = args.out_dir / condition
    cmd = [
        sys.executable,
        "train.py",
        *CONDITIONS[condition],
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
        str(seed),
        "--device",
        args.device,
        "--out-dir",
        str(condition_out),
    ]
    if condition == "mlp":
        cmd.extend(["--hidden-dim", str(args.mlp_hidden_dim)])
    elif condition.startswith("cpa"):
        cmd.extend(["--hidden-dim", str(args.cpa_hidden_dim)])
    print(f"\n=== {condition} seed={seed} ===", flush=True)
    subprocess.run(cmd, check=True)
    model = "cpa" if condition.startswith("cpa") else condition
    return condition_out / f"{model}_seed{seed}"


def read_row(condition: str, seed: int, run_dir: Path) -> dict[str, str | int | float]:
    with (run_dir / "metrics.json").open() as f:
        data = json.load(f)
    row: dict[str, str | int | float] = {
        "condition": condition,
        "seed": seed,
        "parameters": data["parameters"],
        "run_dir": str(run_dir),
    }
    for metric in METRICS:
        row[metric] = data["final"][metric]
    stateful = data.get("stateful")
    if stateful:
        row["stateful_accuracy"] = stateful["accuracy"]
        row["stateful_post_switch_recovery"] = stateful["post_switch_recovery"]
        row["stateful_distractor_recovery"] = stateful["distractor_recovery"]
    else:
        row["stateful_accuracy"] = ""
        row["stateful_post_switch_recovery"] = ""
        row["stateful_distractor_recovery"] = ""
    return row


def write_aggregate(rows: list[dict[str, str | int | float]], path: Path) -> None:
    aggregate_rows = []
    for condition in sorted({str(row["condition"]) for row in rows}):
        subset = [row for row in rows if row["condition"] == condition]
        out: dict[str, str | float] = {"condition": condition}
        for metric in METRICS:
            values = [float(row[metric]) for row in subset]
            out[f"{metric}_mean"] = statistics.mean(values)
            out[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregate_rows.append(out)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for condition in args.conditions:
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition!r}; choose from {sorted(CONDITIONS)}")
        for seed in args.seeds:
            run_dir = run_condition(args, condition, seed)
            rows.append(read_row(condition, seed, run_dir))

    per_seed_path = args.out_dir / "per_seed.csv"
    with per_seed_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    aggregate_path = args.out_dir / "aggregate.csv"
    write_aggregate(rows, aggregate_path)
    print(f"\nwrote: {per_seed_path}")
    print(f"wrote: {aggregate_path}")


if __name__ == "__main__":
    main()
