"""Phase 7.1: remove retrieval and gate supervision from the learned network."""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

import torch

from phase7_soft_chevron.experiment import (
    JOINT_ATTENTION,
    SOFT_CHEVRON,
    CategoryMatchingTask,
    TaskConfig,
    TrainConfig,
    evaluate,
    parameter_count,
    summarize,
    train_model,
)


METHODS = (SOFT_CHEVRON, JOINT_ATTENTION)
CONDITIONS = {
    "full_aux": (0.5, 0.5),
    "retrieval_only": (0.5, 0.0),
    "answer_only": (0.0, 0.0),
}


def run_condition(
    condition: str,
    method: str,
    seeds: Sequence[int],
    steps: int,
    device: torch.device,
) -> Tuple[List[Dict[str, float]], int]:
    retrieval_weight, gate_weight = CONDITIONS[condition]
    task = CategoryMatchingTask(TaskConfig())
    config = replace(
        TrainConfig(steps=steps),
        retrieval_weight=retrieval_weight,
        gate_weight=gate_weight,
    )
    rows: List[Dict[str, float]] = []
    parameters = 0
    for seed in seeds:
        model = train_model(method, task, config, seed, device)
        parameters = parameter_count(model)
        rows.append(evaluate(method, model, task, seed, device))
    return rows, parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    for condition in args.conditions:
        for method in args.methods:
            rows, parameters = run_condition(
                condition, method, args.seeds, args.steps, device
            )
            print("\n%s %s parameters=%d" % (condition, method, parameters))
            for key, (average, spread) in summarize(rows).items():
                print("  %-24s %.4f +/- %.4f" % (key, average, spread))


if __name__ == "__main__":
    main()

