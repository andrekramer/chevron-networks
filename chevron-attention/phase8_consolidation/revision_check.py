"""Focused Phase 8 check after the pre-lock state-management revision."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List

import torch

from phase7_soft_chevron.continual_closure import ClosureConfig, prepare_models
from phase7_soft_chevron.experiment import CategoryMatchingTask, TaskConfig
from phase8_consolidation.experiment import (
    CHEVRON_QUARANTINE,
    JOINT_QUARANTINE,
    DevelopmentConfig,
    run_method,
    selected_development_config,
)
from phase8_consolidation.robustness import (
    PRIMARY_METHODS,
    conditions,
    print_summary,
    summarize,
)


FOCUSED_CONDITIONS = (
    "blocked_default",
    "interleaved_default",
    "distance1",
    "transient1",
    "transient3",
    "transient4",
    "transient5",
)


def readiness(summary: Dict[str, object]) -> Dict[str, object]:
    failures: List[str] = []
    for condition in ("transient1", "transient3", "transient4"):
        for method in (CHEVRON_QUARANTINE, JOINT_QUARANTINE):
            value = summary[condition][method]["false_consolidations"]["mean"]
            if value > 0.0:
                failures.append(f"{condition}/{method} falsely consolidated")

    for condition in ("blocked_default", "interleaved_default", "distance1"):
        for method in (CHEVRON_QUARANTINE, JOINT_QUARANTINE):
            value = summary[condition][method]["final_old_accuracy"]["mean"]
            if value < 0.95:
                failures.append(f"{condition}/{method} old accuracy below 0.95")

    for condition, minimum in (("blocked_default", 0.75), ("interleaved_default", 0.60)):
        for method in (CHEVRON_QUARANTINE, JOINT_QUARANTINE):
            value = (
                summary[condition][method]["novel_categories_retained"]["mean"]
                / 3.0
            )
            if value < minimum:
                failures.append(f"{condition}/{method} acquisition below {minimum:.2f}")

    chevron_near = (
        summary["distance1"][CHEVRON_QUARANTINE]["novel_categories_retained"]["mean"]
        / 3.0
    )
    if chevron_near < 0.60:
        failures.append("distance1/chevron_quarantine acquisition below 0.60")

    replacements = summary["interleaved_default"][CHEVRON_QUARANTINE][
        "candidate_replacements"
    ]["mean"]
    if replacements > 20.0:
        failures.append("interleaved Chevron candidate replacements above 20")

    return {
        "ready_to_lock": not failures,
        "failures": failures,
        "criteria": {
            "no_false_consolidation_through_duration": 4,
            "minimum_old_accuracy": 0.95,
            "minimum_blocked_acquisition": 0.75,
            "minimum_interleaved_acquisition": 0.60,
            "minimum_chevron_near_acquisition": 0.60,
            "maximum_interleaved_chevron_replacements": 20.0,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[107, 117, 127, 137, 147]
    )
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--controller-steps", type=int, default=350)
    parser.add_argument("--consolidation-threshold", type=float, default=0.20)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("phase8_consolidation/revision-results.json"),
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
    selected = replace(
        selected_development_config(),
        consolidation_threshold=args.consolidation_threshold,
    )
    selected_conditions = [
        item for item in conditions() if item.name in FOCUSED_CONDITIONS
    ]
    rows: List[Dict[str, object]] = []
    for condition in selected_conditions:
        provisional = replace(selected, capacity=condition.candidate_capacity)
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
                rows.append(row)
    summary = summarize(rows)
    checks = readiness(summary)
    payload = {
        "development_seeds": args.seeds,
        "selected_config": asdict(selected),
        "joint_evidence_calibration": "clipped score-odds / threshold-odds",
        "mature_retained_match_action": "reject and clear",
        "conditions": [asdict(item) for item in selected_conditions],
        "rows": rows,
        "summary": summary,
        "readiness": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print_summary(summary)
    print("\nREVISION READINESS: %s" % checks["ready_to_lock"])
    for failure in checks["failures"]:
        print("  " + failure)
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()
