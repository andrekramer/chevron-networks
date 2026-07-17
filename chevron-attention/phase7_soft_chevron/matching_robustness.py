"""Phase 7.3: random matching projections and representation-noise tests."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from typing import Dict, List, Sequence

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


@dataclass(frozen=True)
class Condition:
    match_init: str
    train_match_noise: float = 0.0
    train_template_noise: float = 0.015
    eval_match_noise: float = 0.0
    eval_template_noise: float = 0.015


CONDITIONS: Dict[str, Condition] = {
    "identity_clean": Condition("identity"),
    "shared_random_clean": Condition("shared_random"),
    "independent_random_clean": Condition("independent_random"),
    "identity_noise05": Condition(
        "identity", 0.05, 0.05, 0.05, 0.05
    ),
    "independent_noise05": Condition(
        "independent_random", 0.05, 0.05, 0.05, 0.05
    ),
    "identity_ood08": Condition(
        "identity", 0.0, 0.015, 0.08, 0.08
    ),
    "independent_ood08": Condition(
        "independent_random", 0.0, 0.015, 0.08, 0.08
    ),
}


def run_condition(
    condition: str,
    seeds: Sequence[int],
    steps: int,
    device: torch.device,
) -> List[Dict[str, float]]:
    spec = CONDITIONS[condition]
    train_task = CategoryMatchingTask(
        replace(
            TaskConfig(),
            match_noise=spec.train_match_noise,
            template_noise=spec.train_template_noise,
        )
    )
    eval_task = CategoryMatchingTask(
        replace(
            TaskConfig(),
            match_noise=spec.eval_match_noise,
            template_noise=spec.eval_template_noise,
        )
    )
    config = replace(
        TrainConfig(steps=steps),
        retrieval_weight=0.0,
        gate_weight=0.0,
        match_init=spec.match_init,
    )
    rows: List[Dict[str, float]] = []
    for seed in seeds:
        model = train_model(SOFT_CHEVRON, train_task, config, seed, device)
        rows.append(evaluate(SOFT_CHEVRON, model, eval_task, seed, device))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS)
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    for condition in args.conditions:
        rows = run_condition(condition, args.seeds, args.steps, device)
        spec = CONDITIONS[condition]
        print(
            "\n%s init=%s train_noise=(%.3f,%.3f) eval_noise=(%.3f,%.3f)"
            % (
                condition,
                spec.match_init,
                spec.train_match_noise,
                spec.train_template_noise,
                spec.eval_match_noise,
                spec.eval_template_noise,
            )
        )
        for key, (average, spread) in summarize(rows).items():
            print("  %-24s %.4f +/- %.4f" % (key, average, spread))


if __name__ == "__main__":
    main()

