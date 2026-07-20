"""Locked Phase 8 confirmation on new seeds.

No candidate, evidence-calibration, architecture, or threshold parameter is
selected from these results.  Rows are checkpointed after every seed so the
long run can be resumed without changing the protocol.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

from phase7_soft_chevron.continual_closure import ClosureConfig, prepare_models
from phase7_soft_chevron.experiment import CategoryMatchingTask, TaskConfig
from phase8_consolidation.experiment import (
    ALPHA_QUARANTINE,
    CHEVRON_IMMEDIATE,
    CHEVRON_QUARANTINE,
    JOINT_IMMEDIATE,
    JOINT_QUARANTINE,
    METHODS,
    ORACLE,
    DevelopmentConfig,
    Scenario,
    run_method,
    selected_development_config,
)
from phase8_consolidation.robustness import PRIMARY_METHODS, summarize


CONFIRMATION_SEEDS = tuple(range(1007, 1207, 10))
ROBUSTNESS_SEEDS = CONFIRMATION_SEEDS[:10]


@dataclass(frozen=True)
class ConfirmationCondition:
    name: str
    scenario: Scenario
    observation_noise: float = 0.035
    retained_capacity: int = 9
    candidate_capacity: int = 3


PRIMARY_CONDITIONS = (
    ConfirmationCondition(
        "blocked_default", Scenario("blocked_default", "sustained", 8)
    ),
    ConfirmationCondition(
        "near", Scenario("near", "sustained", 8, novel_flips=1)
    ),
)


ROBUSTNESS_CONDITIONS = (
    ConfirmationCondition(
        "interleaved",
        Scenario("interleaved", "sustained", 8, interleaved=True),
    ),
    ConfirmationCondition("noise06", Scenario("noise06", "sustained", 8), 0.06),
    ConfirmationCondition("noise08", Scenario("noise08", "sustained", 8), 0.08),
    ConfirmationCondition("noise10", Scenario("noise10", "sustained", 8), 0.10),
    ConfirmationCondition(
        "distance3", Scenario("distance3", "sustained", 8, novel_flips=3)
    ),
    ConfirmationCondition("transient1", Scenario("transient1", "transient", 1)),
    ConfirmationCondition("transient3", Scenario("transient3", "transient", 3)),
    ConfirmationCondition("transient4", Scenario("transient4", "transient", 4)),
    ConfirmationCondition("transient5", Scenario("transient5", "transient", 5)),
    ConfirmationCondition("transient6", Scenario("transient6", "transient", 6)),
    ConfirmationCondition(
        "retained_capacity8",
        Scenario("retained_capacity8", "sustained", 8),
        retained_capacity=8,
    ),
    ConfirmationCondition(
        "long16", Scenario("long16", "sustained", 16)
    ),
    ConfirmationCondition(
        "bank2_interleaved",
        Scenario("bank2_interleaved", "sustained", 8, interleaved=True),
        candidate_capacity=2,
    ),
)


def bootstrap_ci(
    values: Sequence[float], *, samples: int = 5000, seed: int = 880_001
) -> Tuple[float, float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    generator = random.Random(seed)
    size = len(values)
    estimates = [
        statistics.mean(values[generator.randrange(size)] for _ in range(size))
        for _ in range(samples)
    ]
    estimates.sort()
    lower = estimates[int(0.025 * (samples - 1))]
    upper = estimates[int(0.975 * (samples - 1))]
    return lower, upper


def paired_comparison(
    rows: Sequence[Dict[str, object]],
    condition: str,
    left: str,
    right: str,
    metric: str,
    bootstrap_seed: int,
) -> Dict[str, object]:
    selected = [row for row in rows if row["condition"] == condition]
    by_method = {
        method: {int(row["seed"]): float(row[metric]) for row in selected if row["method"] == method}
        for method in (left, right)
    }
    seeds = sorted(set(by_method[left]).intersection(by_method[right]))
    differences = [by_method[left][seed] - by_method[right][seed] for seed in seeds]
    lower, upper = bootstrap_ci(differences, seed=bootstrap_seed)
    return {
        "condition": condition,
        "left": left,
        "right": right,
        "metric": metric,
        "difference": "left_minus_right",
        "n": len(differences),
        "mean": statistics.mean(differences),
        "sd": statistics.stdev(differences) if len(differences) > 1 else 0.0,
        "bootstrap_95_ci": [lower, upper],
        "left_wins": sum(value > 0.0 for value in differences),
        "ties": sum(value == 0.0 for value in differences),
        "right_wins": sum(value < 0.0 for value in differences),
        "per_seed": [
            {"seed": seed, "difference": difference}
            for seed, difference in zip(seeds, differences)
        ],
    }


def all_comparisons(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    conditions = sorted({str(row["condition"]) for row in rows})
    comparisons: List[Dict[str, object]] = []
    index = 0
    for condition in conditions:
        available = {str(row["method"]) for row in rows if row["condition"] == condition}
        pairs = (
            (CHEVRON_QUARANTINE, CHEVRON_IMMEDIATE),
            (JOINT_QUARANTINE, JOINT_IMMEDIATE),
            (CHEVRON_QUARANTINE, JOINT_QUARANTINE),
        )
        for left, right in pairs:
            if left not in available or right not in available:
                continue
            for metric in (
                "final_old_accuracy",
                "final_new_accuracy",
                "novel_categories_retained",
                "false_consolidations",
                "cross_write_mass",
                "template_mse",
            ):
                comparisons.append(
                    paired_comparison(
                        rows,
                        condition,
                        left,
                        right,
                        metric,
                        881_000 + index,
                    )
                )
                index += 1
        if ALPHA_QUARANTINE in available:
            for metric in ("final_old_accuracy", "cross_write_mass", "template_mse"):
                comparisons.append(
                    paired_comparison(
                        rows,
                        condition,
                        CHEVRON_QUARANTINE,
                        ALPHA_QUARANTINE,
                        metric,
                        881_000 + index,
                    )
                )
                index += 1
    return comparisons


def comparison_lookup(
    comparisons: Sequence[Dict[str, object]],
    condition: str,
    left: str,
    right: str,
    metric: str,
) -> Dict[str, object]:
    return next(
        item
        for item in comparisons
        if item["condition"] == condition
        and item["left"] == left
        and item["right"] == right
        and item["metric"] == metric
    )


def evaluate_hypotheses(
    summary: Dict[str, object], comparisons: Sequence[Dict[str, object]]
) -> Dict[str, object]:
    transient_pass = all(
        summary[condition][method]["false_consolidations"]["mean"] == 0.0
        for condition in ("transient1", "transient3", "transient4")
        for method in (CHEVRON_QUARANTINE, JOINT_QUARANTINE)
    )
    blocked_acquisition = {
        method: summary["blocked_default"][method]["novel_categories_retained"]["mean"] / 3.0
        for method in (CHEVRON_QUARANTINE, JOINT_QUARANTINE)
    }
    interleaved_acquisition = {
        method: summary["interleaved"][method]["novel_categories_retained"]["mean"] / 3.0
        for method in (CHEVRON_QUARANTINE, JOINT_QUARANTINE)
    }
    alpha_old = comparison_lookup(
        comparisons,
        "near",
        CHEVRON_QUARANTINE,
        ALPHA_QUARANTINE,
        "final_old_accuracy",
    )
    alpha_cross = comparison_lookup(
        comparisons,
        "near",
        CHEVRON_QUARANTINE,
        ALPHA_QUARANTINE,
        "cross_write_mass",
    )
    near_joint = comparison_lookup(
        comparisons,
        "near",
        CHEVRON_QUARANTINE,
        JOINT_QUARANTINE,
        "final_new_accuracy",
    )
    near_immediate = comparison_lookup(
        comparisons,
        "near",
        CHEVRON_QUARANTINE,
        CHEVRON_IMMEDIATE,
        "final_new_accuracy",
    )
    default_joint = comparison_lookup(
        comparisons,
        "blocked_default",
        CHEVRON_QUARANTINE,
        JOINT_QUARANTINE,
        "final_new_accuracy",
    )
    return {
        "H1_temporal_quarantine_protects_stability": {
            "passed": transient_pass,
            "criterion": "zero false consolidation for quarantine at durations 1, 3, and 4",
        },
        "H2_quarantine_preserves_plasticity": {
            "passed": (
                blocked_acquisition[CHEVRON_QUARANTINE] >= 0.75
                and blocked_acquisition[JOINT_QUARANTINE] >= 0.75
                and interleaved_acquisition[CHEVRON_QUARANTINE] >= 0.60
                and interleaved_acquisition[JOINT_QUARANTINE] >= 0.60
            ),
            "blocked_acquisition": blocked_acquisition,
            "interleaved_acquisition": interleaved_acquisition,
        },
        "H3_assent_remains_causal": {
            "passed": (
                alpha_old["bootstrap_95_ci"][0] > 0.0
                and alpha_cross["bootstrap_95_ci"][1] < 0.0
            ),
            "near_old_accuracy_comparison": alpha_old,
            "near_cross_write_comparison": alpha_cross,
        },
        "H4_near_category_advantage_replicates": {
            "passed": (
                near_joint["bootstrap_95_ci"][0] > 0.0
                and near_immediate["bootstrap_95_ci"][0] > 0.0
            ),
            "versus_joint_quarantine": near_joint,
            "versus_chevron_immediate": near_immediate,
        },
        "H5_general_superiority_not_assumed": {
            "default_chevron_minus_joint": default_joint,
            "criterion": "report the paired interval; no pass is assigned",
        },
    }


def checkpoint(path: Path, rows: Sequence[Dict[str, object]], completed: Sequence[int]) -> None:
    payload = {
        "protocol": "phase8_locked_confirmation",
        "completed_seeds": list(completed),
        "rows": list(rows),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(CONFIRMATION_SEEDS))
    parser.add_argument("--robustness-seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--controller-steps", type=int, default=350)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("phase8_consolidation/confirmation-results.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("phase8_consolidation/confirmation-results.partial.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if any(seed < 1000 for seed in args.seeds):
        raise ValueError("confirmation seeds must be disjoint from development seeds")
    task_config = TaskConfig()
    task = CategoryMatchingTask(task_config)
    locked = selected_development_config()
    rows: List[Dict[str, object]] = []
    completed: List[int] = []
    robust_seed_set = set(args.seeds[: args.robustness_seeds])

    for seed in args.seeds:
        prepared = prepare_models(
            [seed],
            task,
            args.steps,
            ClosureConfig(controller_steps=args.controller_steps),
            torch.device(args.device),
        )[seed]
        seed_conditions = list(PRIMARY_CONDITIONS)
        if seed in robust_seed_set:
            seed_conditions.extend(ROBUSTNESS_CONDITIONS)
        for condition in seed_conditions:
            methods = METHODS if condition in PRIMARY_CONDITIONS else PRIMARY_METHODS
            provisional = replace(locked, capacity=condition.candidate_capacity)
            development = DevelopmentConfig(
                observation_noise=condition.observation_noise,
                memory_capacity=condition.retained_capacity,
            )
            for method in methods:
                row = run_method(
                    method,
                    prepared,
                    task_config,
                    condition.scenario,
                    seed,
                    provisional,
                    development,
                )
                row["condition"] = condition.name
                row["observation_noise"] = condition.observation_noise
                row["retained_capacity"] = condition.retained_capacity
                row["candidate_capacity"] = condition.candidate_capacity
                rows.append(row)
        completed.append(seed)
        checkpoint(args.checkpoint, rows, completed)
        print("completed seed %d (%d/%d)" % (seed, len(completed), len(args.seeds)), flush=True)

    summary = summarize(rows)
    comparisons = all_comparisons(rows)
    hypotheses = evaluate_hypotheses(summary, comparisons)
    payload = {
        "protocol": "phase8_locked_confirmation",
        "confirmation_seeds": args.seeds,
        "robustness_seeds": args.seeds[: args.robustness_seeds],
        "train_steps": args.steps,
        "controller_steps": args.controller_steps,
        "locked_config": asdict(locked),
        "primary_conditions": [asdict(item) for item in PRIMARY_CONDITIONS],
        "robustness_conditions": [asdict(item) for item in ROBUSTNESS_CONDITIONS],
        "rows": rows,
        "summary": summary,
        "paired_comparisons": comparisons,
        "hypotheses": hypotheses,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(hypotheses, indent=2))
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()

