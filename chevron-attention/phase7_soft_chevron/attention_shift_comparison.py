"""Phase 7.4: Soft Chevron versus joint attention under representation shift."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
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


@dataclass(frozen=True)
class Shift:
    train_match_noise: float
    train_template_noise: float
    eval_match_noise: float
    eval_template_noise: float


SHIFTS: Dict[str, Shift] = {
    "clean": Shift(0.0, 0.015, 0.0, 0.015),
    "trained_noise05": Shift(0.05, 0.05, 0.05, 0.05),
    "unseen_noise08": Shift(0.0, 0.015, 0.08, 0.08),
}


def run_condition(
    shift_name: str,
    method: str,
    seeds: Sequence[int],
    steps: int,
    device: torch.device,
) -> Tuple[List[Dict[str, float]], int]:
    shift = SHIFTS[shift_name]
    train_task = CategoryMatchingTask(
        replace(
            TaskConfig(),
            match_noise=shift.train_match_noise,
            template_noise=shift.train_template_noise,
        )
    )
    eval_task = CategoryMatchingTask(
        replace(
            TaskConfig(),
            match_noise=shift.eval_match_noise,
            template_noise=shift.eval_template_noise,
        )
    )
    # Chevron receives the harder independent A/N initialization. Joint
    # attention retains its existing shared Q_N/K_N initialization.
    config = replace(
        TrainConfig(steps=steps),
        retrieval_weight=0.0,
        gate_weight=0.0,
        match_init="independent_random",
    )
    rows: List[Dict[str, float]] = []
    parameters = 0
    for seed in seeds:
        model = train_model(method, train_task, config, seed, device)
        parameters = parameter_count(model)
        rows.append(evaluate(method, model, eval_task, seed, device))
    return rows, parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shifts", nargs="+", choices=SHIFTS, default=list(SHIFTS))
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    for shift_name in args.shifts:
        shift = SHIFTS[shift_name]
        for method in args.methods:
            rows, parameters = run_condition(
                shift_name, method, args.seeds, args.steps, device
            )
            print(
                "\n%s %s parameters=%d train_noise=(%.3f,%.3f) "
                "eval_noise=(%.3f,%.3f)"
                % (
                    shift_name,
                    method,
                    parameters,
                    shift.train_match_noise,
                    shift.train_template_noise,
                    shift.eval_match_noise,
                    shift.eval_template_noise,
                )
            )
            for key, (average, spread) in summarize(rows).items():
                print("  %-24s %.4f +/- %.4f" % (key, average, spread))


if __name__ == "__main__":
    main()
