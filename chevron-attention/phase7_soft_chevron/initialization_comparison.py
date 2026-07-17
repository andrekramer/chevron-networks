"""Phase 7.2: answer-only learning from varied theta and sharpness initializations."""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

import torch

from phase7_soft_chevron.experiment import (
    SOFT_CHEVRON,
    CategoryMatchingTask,
    TaskConfig,
    TrainConfig,
    evaluate,
    summarize,
    train_model,
)


INITIALIZATIONS: Dict[str, Tuple[float, float]] = {
    "calibrated": (0.10, 30.0),
    "low_theta": (0.02, 30.0),
    "high_theta": (0.30, 30.0),
    "soft_gate": (0.10, 5.0),
    "sharp_gate": (0.10, 80.0),
    "closed_saturated": (0.02, 80.0),
    "open_saturated": (0.30, 80.0),
}


def run_condition(
    condition: str,
    seeds: Sequence[int],
    steps: int,
    device: torch.device,
) -> List[Dict[str, float]]:
    theta, sharpness = INITIALIZATIONS[condition]
    task = CategoryMatchingTask(TaskConfig())
    config = replace(
        TrainConfig(steps=steps),
        retrieval_weight=0.0,
        gate_weight=0.0,
        theta_init=theta,
        sharpness_init=sharpness,
    )
    rows: List[Dict[str, float]] = []
    for seed in seeds:
        model = train_model(SOFT_CHEVRON, task, config, seed, device)
        rows.append(evaluate(SOFT_CHEVRON, model, task, seed, device))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=INITIALIZATIONS,
        default=list(INITIALIZATIONS),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    for condition in args.conditions:
        initial_theta, initial_sharpness = INITIALIZATIONS[condition]
        rows = run_condition(condition, args.seeds, args.steps, device)
        print(
            "\n%s theta_init=%.3f sharpness_init=%.1f"
            % (condition, initial_theta, initial_sharpness)
        )
        for key, (average, spread) in summarize(rows).items():
            print("  %-24s %.4f +/- %.4f" % (key, average, spread))


if __name__ == "__main__":
    main()

