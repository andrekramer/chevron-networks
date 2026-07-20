"""Phase 8 pre-lock robustness matrix on development seeds.

The selected candidate parameters are held fixed except where candidate-bank
capacity is itself the declared robustness variable.  These are still
development seeds; the script decides only whether the mechanism is ready to
be frozen for a later confirmation run.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from phase7_soft_chevron.continual_closure import ClosureConfig, prepare_models
from phase7_soft_chevron.experiment import CategoryMatchingTask, TaskConfig
from phase8_consolidation.experiment import (
    CHEVRON_IMMEDIATE,
    CHEVRON_QUARANTINE,
    JOINT_IMMEDIATE,
    JOINT_QUARANTINE,
    DevelopmentConfig,
    Scenario,
    run_method,
    selected_development_config,
)
from phase8_consolidation.provisional_memory import ProvisionalConfig


PRIMARY_METHODS = (
    CHEVRON_IMMEDIATE,
    CHEVRON_QUARANTINE,
    JOINT_IMMEDIATE,
    JOINT_QUARANTINE,
)


@dataclass(frozen=True)
class Condition:
    name: str
    scenario: Scenario
    observation_noise: float = 0.035
    candidate_capacity: int = 3


def conditions() -> Sequence[Condition]:
    return (
        Condition("blocked_default", Scenario("blocked_default", "sustained", 8)),
        Condition(
            "interleaved_default",
            Scenario("interleaved_default", "sustained", 8, interleaved=True),
        ),
        Condition("noise06", Scenario("noise06", "sustained", 8), 0.06),
        Condition("noise08", Scenario("noise08", "sustained", 8), 0.08),
        Condition("noise10", Scenario("noise10", "sustained", 8), 0.10),
        Condition("distance1", Scenario("distance1", "sustained", 8, novel_flips=1)),
        Condition("distance3", Scenario("distance3", "sustained", 8, novel_flips=3)),
        Condition(
            "bank1_interleaved",
            Scenario("bank1_interleaved", "sustained", 8, interleaved=True),
            candidate_capacity=1,
        ),
        Condition(
            "bank2_interleaved",
            Scenario("bank2_interleaved", "sustained", 8, interleaved=True),
            candidate_capacity=2,
        ),
        *(
            Condition(
                f"transient{duration}",
                Scenario(f"transient{duration}", "transient", duration),
            )
            for duration in (1, 3, 4, 5, 6)
        ),
    )


def numeric_summary(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    keys = (
        "final_old_accuracy",
        "final_new_accuracy",
        "novel_categories_retained",
        "false_consolidations",
        "false_splits",
        "cross_write_mass",
        "template_mse",
        "mean_acquisition_delay",
        "candidate_replacements",
        "candidate_retained_match_rejections",
        "event_remaining_mass",
        "event_novelty",
        "event_eligible_mass",
        "familiar_remaining_mass",
        "familiar_novelty",
        "familiar_eligible_mass",
    )
    summary: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if values:
            summary[key] = {
                "mean": statistics.mean(values),
                "sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
    return summary


def summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    grouped: Dict[str, Dict[str, List[Dict[str, object]]]] = {}
    for row in rows:
        grouped.setdefault(str(row["condition"]), {}).setdefault(
            str(row["method"]), []
        ).append(row)
    return {
        condition: {
            method: numeric_summary(method_rows)
            for method, method_rows in method_groups.items()
        }
        for condition, method_groups in grouped.items()
    }


def lock_checks(summary: Dict[str, object]) -> Dict[str, object]:
    """Predeclared readiness checks, not statistical confirmation tests."""

    failures: List[str] = []
    for condition in ("transient1", "transient3", "transient4"):
        methods = summary[condition]
        for method in (CHEVRON_QUARANTINE, JOINT_QUARANTINE):
            false_mean = methods[method]["false_consolidations"]["mean"]
            if false_mean > 0.0:
                failures.append(f"{condition}/{method} falsely consolidated")

    for condition in ("blocked_default", "interleaved_default", "distance1"):
        methods = summary[condition]
        for method in (CHEVRON_QUARANTINE, JOINT_QUARANTINE):
            old_accuracy = methods[method]["final_old_accuracy"]["mean"]
            if old_accuracy < 0.95:
                failures.append(f"{condition}/{method} old accuracy below 0.95")

    blocked = summary["blocked_default"]
    for method in (CHEVRON_QUARANTINE, JOINT_QUARANTINE):
        acquisition = blocked[method]["novel_categories_retained"]["mean"] / 3.0
        if acquisition < 0.75:
            failures.append(f"blocked_default/{method} acquisition below 0.75")

    interleaved = summary["interleaved_default"]
    for method in (CHEVRON_QUARANTINE, JOINT_QUARANTINE):
        acquisition = interleaved[method]["novel_categories_retained"]["mean"] / 3.0
        if acquisition < 0.60:
            failures.append(f"interleaved_default/{method} acquisition below 0.60")

    chevron_near = (
        summary["distance1"][CHEVRON_QUARANTINE]["novel_categories_retained"]["mean"]
        / 3.0
    )
    if chevron_near < 0.60:
        failures.append("distance1/chevron_quarantine acquisition below 0.60")

    return {
        "ready_to_lock": not failures,
        "failures": failures,
        "criteria": {
            "no_false_consolidation_through_duration": 4,
            "minimum_old_accuracy": 0.95,
            "minimum_blocked_acquisition": 0.75,
            "minimum_interleaved_acquisition": 0.60,
            "minimum_chevron_near_acquisition": 0.60,
        },
    }


def print_summary(summary: Dict[str, object]) -> None:
    for condition, methods_value in summary.items():
        methods = methods_value
        print("\n" + condition.upper())
        for method in PRIMARY_METHODS:
            metrics = methods[method]
            old = metrics["final_old_accuracy"]["mean"]
            new = metrics["final_new_accuracy"]["mean"]
            retained = metrics["novel_categories_retained"]["mean"]
            false = metrics["false_consolidations"]["mean"]
            delay = metrics.get("mean_acquisition_delay", {}).get("mean")
            delay_text = "-" if delay is None else f"{delay:.2f}"
            print(
                f"{method:22s} old={old:.3f} new={new:.3f} "
                f"retained={retained:.2f} false={false:.2f} delay={delay_text}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[107, 117, 127, 137, 147]
    )
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--controller-steps", type=int, default=350)
    parser.add_argument("--consolidation-threshold", type=float, default=None)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("phase8_consolidation/robustness-results.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_config = TaskConfig()
    task = CategoryMatchingTask(task_config)
    prepared = prepare_models(
        args.seeds,
        task,
        args.steps,
        ClosureConfig(controller_steps=args.controller_steps),
        torch.device(args.device),
    )
    selected = selected_development_config()
    if args.consolidation_threshold is not None:
        selected = replace(
            selected, consolidation_threshold=args.consolidation_threshold
        )
    rows: List[Dict[str, object]] = []
    for condition in conditions():
        provisional: ProvisionalConfig = replace(
            selected, capacity=condition.candidate_capacity
        )
        development = DevelopmentConfig(observation_noise=condition.observation_noise)
        for seed in args.seeds:
            for method in PRIMARY_METHODS:
                row = run_method(
                    method,
                    prepared[seed],
                    task_config,
                    condition.scenario,
                    seed,
                    provisional,
                    development,
                )
                row["condition"] = condition.name
                row["candidate_capacity"] = condition.candidate_capacity
                row["observation_noise"] = condition.observation_noise
                rows.append(row)
    summary = summarize(rows)
    checks = lock_checks(summary)
    payload = {
        "development_seeds": args.seeds,
        "train_steps": args.steps,
        "controller_steps": args.controller_steps,
        "selected_config": asdict(selected),
        "conditions": [asdict(item) for item in conditions()],
        "rows": rows,
        "summary": summary,
        "lock_checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print_summary(summary)
    print("\nLOCK READINESS: %s" % checks["ready_to_lock"])
    for failure in checks["failures"]:
        print("  " + failure)
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()
