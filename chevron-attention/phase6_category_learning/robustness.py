"""Phase 6 final robustness suite with paired twenty-seed comparisons."""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

from phase6_category_learning.ambiguity_comparison import (
    AmbiguityConfig,
    run_comparison as run_ambiguity,
)
from phase6_category_learning.drift_comparison import (
    DriftConfig,
    DriftStream,
    DriftTaskConfig,
    metrics as drift_metrics,
    run_method as run_drift_method,
)
from phase6_category_learning.experiment import (
    CategoryStream,
    MLPConfig,
    MemoryConfig,
    TaskConfig,
    metrics as category_metrics,
    run_method as run_category_method,
)


SEEDS = tuple(7 + 10 * index for index in range(20))
CATEGORY_METHODS = ("chevron_art", "persistent_attention", "standard_attention")
DRIFT_METHODS = (
    "chevron_soft",
    "standard_dual_attention",
    "chevron_dual",
    "persistent_single",
)
AMBIGUITY_METHODS = (
    "joint_softmax",
    "joint_top1",
    "soft_vigilance",
    "sharp_vigilance",
    "masked_attention",
    "hard_search",
)


def mean_ci(values: Sequence[float]) -> Tuple[float, float]:
    average = sum(values) / len(values)
    if len(values) < 2:
        return average, 0.0
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return average, 1.96 * math.sqrt(variance / len(values))


def cell(values: Sequence[float]) -> str:
    average, ci = mean_ci(values)
    return "%.3f+/-%.3f" % (average, ci)


def category_conditions() -> List[Tuple[str, TaskConfig, MemoryConfig]]:
    task = TaskConfig()
    memory = MemoryConfig()
    return [
        ("default", task, memory),
        ("noise=.05", replace(task, noise_std=0.05), memory),
        ("noise=.06", replace(task, noise_std=0.06), memory),
        ("noise=.08", replace(task, noise_std=0.08), memory),
        ("noise=.12", replace(task, noise_std=0.12), memory),
        ("separation=3", replace(task, min_code_distance=3), memory),
        ("brief=2", replace(task, transient_length=2), memory),
        ("brief=6", replace(task, transient_length=6), memory),
        ("brief=7", replace(task, transient_length=7), memory),
        ("capacity=3", task, replace(memory, retained_slots=3)),
        ("capacity=5", task, replace(memory, retained_slots=5)),
    ]


def run_category(seeds: Sequence[int]) -> None:
    print("CATEGORY CREATION (mean +/- 95% CI)")
    print("condition method old_after_brief final spurious_categories")
    for name, task, memory in category_conditions():
        rows: Dict[str, List[Dict[str, float]]] = {
            method: [] for method in CATEGORY_METHODS
        }
        for seed in seeds:
            stream = CategoryStream(task, seed).build()
            for method in CATEGORY_METHODS:
                result = run_category_method(
                    method, stream, task, memory, MLPConfig(), seed
                )
                rows[method].append(category_metrics(result, task))
        for method in CATEGORY_METHODS:
            print(
                "%-13s %-22s %-15s %-15s %-15s"
                % (
                    name,
                    method,
                    cell([row["old_retention_after_transients"] for row in rows[method]]),
                    cell([row["final_accuracy"] for row in rows[method]]),
                    cell([row["transient_categories_retained"] for row in rows[method]]),
                )
            )


def drift_conditions() -> List[Tuple[str, DriftTaskConfig, DriftConfig]]:
    task = DriftTaskConfig()
    config = DriftConfig()
    return [
        ("default", task, config),
        ("noise=.04", replace(task, noise_std=0.04), config),
        ("noise=.07", replace(task, noise_std=0.07), config),
        ("shift_y=.55", replace(task, shift_y=0.55), config),
        ("shift_y=.95", replace(task, shift_y=0.95), config),
        ("short=5", replace(task, short_shift=5), config),
        ("short=15", replace(task, short_shift=15), config),
        ("threshold=.40", task, replace(config, threshold=0.40)),
        ("threshold=.70", task, replace(config, threshold=0.70)),
    ]


def run_drift(seeds: Sequence[int]) -> None:
    print("DRIFT (mean +/- 95% CI)")
    print("condition method short_cycle long_online adapt_steps retained_final")
    for name, task, config in drift_conditions():
        rows: Dict[str, List[Dict[str, float]]] = {
            method: [] for method in DRIFT_METHODS
        }
        for seed in seeds:
            stream = DriftStream(task, seed).build()
            for method in DRIFT_METHODS:
                rows[method].append(
                    drift_metrics(run_drift_method(method, stream, task, config))
                )
        for method in DRIFT_METHODS:
            print(
                "%-13s %-24s %-15s %-15s %-15s %-15s"
                % (
                    name,
                    method,
                    cell([row["short_cycle_min"] for row in rows[method]]),
                    cell([row["long_shift_online"] for row in rows[method]]),
                    cell([row["long_adaptation_steps"] for row in rows[method]]),
                    cell([row["retained_shift_probe"] for row in rows[method]]),
                )
            )


def ambiguity_conditions() -> List[Tuple[str, AmbiguityConfig, int]]:
    config = AmbiguityConfig()
    return [
        ("decoys=7", config, 7),
        ("decoys=15", config, 15),
        ("decoys=31", config, 31),
        ("decoys=63", config, 63),
        ("target_noise=.12", replace(config, target_a_noise=0.12), 31),
        ("template_noise=.04", replace(config, template_noise=0.04), 31),
        ("template_noise=.07", replace(config, template_noise=0.07), 31),
        ("vigilance=.04", replace(config, vigilance=0.04), 31),
        ("vigilance=.08", replace(config, vigilance=0.08), 31),
        ("sharpness=40", replace(config, soft_sharpness=40.0), 31),
        ("sharpness=120", replace(config, soft_sharpness=120.0), 31),
    ]


def run_ambiguity_section(seeds: Sequence[int], episodes: int) -> None:
    print("AMBIGUITY (mean +/- 95% CI)")
    print("condition method accuracy")
    for name, config, decoys in ambiguity_conditions():
        results = run_ambiguity(config, [decoys], seeds, episodes)
        for method in AMBIGUITY_METHODS:
            average, spread = results[method][decoys]["accuracy"]
            # run_ambiguity returns SD; recomputing CI would require retaining
            # rows, so convert the reported seed SD to a normal 95% interval.
            ci = 1.96 * spread / math.sqrt(len(seeds))
            print("%-19s %-22s %.3f+/-%.3f" % (name, method, average, ci))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--section", choices=("all", "category", "drift", "ambiguity"), default="all"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--ambiguity-episodes", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.section in ("all", "category"):
        run_category(args.seeds)
    if args.section in ("all", "drift"):
        run_drift(args.seeds)
    if args.section in ("all", "ambiguity"):
        run_ambiguity_section(args.seeds, args.ambiguity_episodes)


if __name__ == "__main__":
    main()
