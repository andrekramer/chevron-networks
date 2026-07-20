"""Screen Phase 8 persistence parameters on development seeds only.

The objective is declared in code before results are examined.  It penalizes
false consolidation, failure to acquire sustained categories, excessive
acquisition delay, and loss of old-category accuracy.  It does not optimize
final accuracy alone.
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Sequence

import torch

from phase7_soft_chevron.continual_closure import ClosureConfig, prepare_models
from phase7_soft_chevron.experiment import CategoryMatchingTask, TaskConfig
from phase8_consolidation.experiment import (
    CHEVRON_QUARANTINE,
    JOINT_QUARANTINE,
    SCENARIOS,
    DevelopmentConfig,
    run_method,
)
from phase8_consolidation.provisional_memory import ProvisionalConfig


DEVELOPMENT_METHODS = (CHEVRON_QUARANTINE, JOINT_QUARANTINE)


def objective(rows: Sequence[Dict[str, object]]) -> Dict[str, float]:
    transient = [row for row in rows if not bool(row["should_consolidate"])]
    sustained = [row for row in rows if bool(row["should_consolidate"])]
    false_consolidation = sum(float(row["false_consolidations"]) for row in transient) / max(
        len(transient), 1
    )
    failure = sum(
        1.0
        - float(row["novel_categories_retained"])
        / max(float(row["novel_categories_required"]), 1.0)
        for row in sustained
    ) / max(len(sustained), 1)
    delay = sum(
        min(
            float(row["mean_acquisition_delay"])
            if row["mean_acquisition_delay"] is not None
            else 9.0,
            9.0,
        )
        / 8.0
        for row in sustained
    ) / max(len(sustained), 1)
    old_loss = sum(1.0 - float(row["final_old_accuracy"]) for row in rows) / max(
        len(rows), 1
    )
    false_split = sum(min(float(row["false_splits"]), 1.0) for row in rows) / max(
        len(rows), 1
    )
    # Stability receives the largest explicit penalty. Failure remains more
    # costly than delay so a system cannot win merely by refusing to learn.
    score = (
        2.0 * false_consolidation
        + failure
        + 0.25 * delay
        + old_loss
        + 0.5 * false_split
    )
    return {
        "score": score,
        "false_consolidation": false_consolidation,
        "sustained_failure": failure,
        "normalized_delay": delay,
        "old_accuracy_loss": old_loss,
        "false_split": false_split,
    }


def configs() -> List[ProvisionalConfig]:
    base = ProvisionalConfig(
        minimum_distinct_mismatch=0.04,
        minimum_eligible_mass=0.10,
    )
    return [
        replace(
            base,
            persistence_beta=beta,
            consolidation_threshold=threshold,
            minimum_support=support,
        )
        for beta, threshold, support in itertools.product(
            (0.80, 0.90), (0.25, 0.40), (3, 5)
        )
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[107, 117])
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--controller-steps", type=int, default=350)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("phase8_consolidation/development-sweep.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_config = TaskConfig()
    task = CategoryMatchingTask(task_config)
    development = DevelopmentConfig()
    prepared = prepare_models(
        args.seeds,
        task,
        args.steps,
        ClosureConfig(controller_steps=args.controller_steps),
        torch.device(args.device),
    )
    results: List[Dict[str, object]] = []
    for index, config in enumerate(configs()):
        rows: List[Dict[str, object]] = []
        for seed in args.seeds:
            for scenario in SCENARIOS:
                for method in DEVELOPMENT_METHODS:
                    rows.append(
                        run_method(
                            method,
                            prepared[seed],
                            task_config,
                            scenario,
                            seed,
                            config,
                            development,
                        )
                    )
        result = {
            "index": index,
            "config": asdict(config),
            "objective": objective(rows),
            "rows": rows,
        }
        results.append(result)
        print(
            "config=%d beta=%.2f tau=%.2f support=%d score=%.4f false=%.3f failure=%.3f delay=%.3f"
            % (
                index,
                config.persistence_beta,
                config.consolidation_threshold,
                config.minimum_support,
                result["objective"]["score"],
                result["objective"]["false_consolidation"],
                result["objective"]["sustained_failure"],
                result["objective"]["normalized_delay"],
            )
        )
    results.sort(key=lambda item: item["objective"]["score"])
    payload = {
        "development_seeds": args.seeds,
        "train_steps": args.steps,
        "controller_steps": args.controller_steps,
        "objective_weights": {
            "false_consolidation": 2.0,
            "sustained_failure": 1.0,
            "normalized_delay": 0.25,
            "old_accuracy_loss": 1.0,
            "false_split": 0.5,
        },
        "results": results,
        "selected": results[0],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    best = results[0]
    print("\nselected config %d: %s" % (best["index"], json.dumps(best["config"])))
    print("wrote %s" % args.output)


if __name__ == "__main__":
    main()

