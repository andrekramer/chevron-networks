"""Parameter sweeps for the integrated Phase 3 experiment."""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Dict, Iterable, List, Tuple

from phase3_integrated.experiment import Config, Schedule, run_seed, summarize


DEFAULT_SEEDS = [7, 17, 27, 37, 47]


def mean_metric(
    configs: Iterable[Tuple[str, Config]],
    seeds: List[int],
    method: str,
    metric_names: List[str],
) -> Dict[str, Dict[str, float]]:
    result = {}
    for label, config in configs:
        runs = [run_seed(config, seed)[method] for seed in seeds]
        result[label] = {
            metric_name: summarize(run[metric_name] for run in runs)[0]
            for metric_name in metric_names
        }
    return result


def print_table(
    title: str,
    rows: Dict[str, Dict[str, float]],
    metric_names: List[str],
) -> None:
    print(title)
    print("label " + " ".join(metric_names))
    for label, metrics in rows.items():
        formatted = " ".join("%.4f" % metrics[name] for name in metric_names)
        print("%s %s" % (label, formatted))
    print()


def transient_duration_configs(base: Config) -> List[Tuple[str, Config]]:
    return [
        (
            str(duration),
            replace(
                base,
                schedule=replace(base.schedule, transient_revoke=duration),
            ),
        )
        for duration in [2, 5, 10, 20, 40, 80, 120]
    ]


def sustained_duration_configs(base: Config) -> List[Tuple[str, Config]]:
    return [
        (
            str(duration),
            replace(
                base,
                schedule=replace(
                    base.schedule,
                    sustained_revoke=duration,
                    sustained_restore=duration,
                ),
            ),
        )
        for duration in [20, 40, 60, 90, 120, 180]
    ]


def fixed_slow_rate_configs(base: Config) -> List[Tuple[str, Config]]:
    return [
        (
            "%.3f" % eta_n_low,
            replace(base, eta_n_low=eta_n_low),
        )
        for eta_n_low in [0.002, 0.004, 0.008, 0.012, 0.020, 0.040, 0.080]
    ]


def idl_threshold_configs(base: Config) -> List[Tuple[str, Config]]:
    return [
        (
            "%.2f" % threshold,
            replace(base, threshold=threshold),
        )
        for threshold in [0.15, 0.25, 0.35, 0.45, 0.55]
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Config()

    print_table(
        "transient_duration integrated_idl",
        mean_metric(
            transient_duration_configs(base),
            args.seeds,
            "integrated_idl",
            [
                "transient_retained_drift",
                "recovery_active_accuracy",
                "transient_update_gate_mean",
                "post_revoke_retained_accuracy",
                "final_active_accuracy",
            ],
        ),
        [
            "transient_retained_drift",
            "recovery_active_accuracy",
            "transient_update_gate_mean",
            "post_revoke_retained_accuracy",
            "final_active_accuracy",
        ],
    )

    print_table(
        "sustained_duration integrated_idl",
        mean_metric(
            sustained_duration_configs(base),
            args.seeds,
            "integrated_idl",
            [
                "post_revoke_retained_accuracy",
                "final_active_accuracy",
                "revoke_consolidation_steps",
                "restore_consolidation_steps",
            ],
        ),
        [
            "post_revoke_retained_accuracy",
            "final_active_accuracy",
            "revoke_consolidation_steps",
            "restore_consolidation_steps",
        ],
    )

    print_table(
        "fixed_slow_rate fixed_slow",
        mean_metric(
            fixed_slow_rate_configs(base),
            args.seeds,
            "fixed_slow",
            [
                "transient_retained_drift",
                "recovery_active_accuracy",
                "post_revoke_retained_accuracy",
                "final_active_accuracy",
                "revoke_consolidation_steps",
                "restore_consolidation_steps",
            ],
        ),
        [
            "transient_retained_drift",
            "recovery_active_accuracy",
            "post_revoke_retained_accuracy",
            "final_active_accuracy",
            "revoke_consolidation_steps",
            "restore_consolidation_steps",
        ],
    )

    print_table(
        "idl_threshold integrated_idl",
        mean_metric(
            idl_threshold_configs(base),
            args.seeds,
            "integrated_idl",
            [
                "transient_retained_drift",
                "recovery_active_accuracy",
                "post_revoke_retained_accuracy",
                "final_active_accuracy",
                "revoke_consolidation_steps",
                "restore_consolidation_steps",
            ],
        ),
        [
            "transient_retained_drift",
            "recovery_active_accuracy",
            "post_revoke_retained_accuracy",
            "final_active_accuracy",
            "revoke_consolidation_steps",
            "restore_consolidation_steps",
        ],
    )


if __name__ == "__main__":
    main()
