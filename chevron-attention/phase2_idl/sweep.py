"""One-factor-at-a-time sensitivity sweeps for phase-two IDL."""

import argparse
import statistics
from dataclasses import replace
from typing import Dict, Iterable, List, Tuple

from phase2_idl.experiment import Config, Schedule, run_seed


DEFAULT_SEEDS = [7, 17, 27, 37, 47]


def mean(values: Iterable[float]) -> float:
    return statistics.mean(values)


def aggregate(
    config: Config, seeds: List[int], shift_magnitude: float = 2.0
) -> Dict[str, Dict[str, float]]:
    runs = [run_seed(config, seed, shift_magnitude) for seed in seeds]
    result: Dict[str, Dict[str, float]] = {}
    for method in runs[0]:
        result[method] = {}
        for metric in runs[0][method]:
            result[method][metric] = mean(run[method][metric] for run in runs)
    return result


def tradeoff_row(result: Dict[str, Dict[str, float]]) -> Tuple[float, ...]:
    idl = result["idl"]
    always = result["always_slow"]
    low = result["fixed_slow_low"]
    protection = 1.0 - idl["transient_n_drift"] / always["transient_n_drift"]
    return (
        idl["transient_n_drift"],
        always["transient_n_drift"],
        100.0 * protection,
        idl["sustained_adaptation_steps"],
        always["sustained_adaptation_steps"],
        low["sustained_adaptation_steps"],
        idl["sustained_final_error"],
        idl["transient_gate_mean"],
        idl["sustained_early_gate_mean"],
    )


def print_table(
    title: str,
    parameter: str,
    conditions: Iterable[Tuple[str, Config, float]],
    seeds: List[int],
) -> None:
    print("\n## %s" % title)
    print(
        "| %s | IDL brief drift | Full-rate drift | Protection | "
        "IDL adapt | Full-rate adapt | Low-rate adapt | IDL final error | "
        "Brief gate | Sustained gate |" % parameter
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, config, shift_magnitude in conditions:
        row = tradeoff_row(aggregate(config, seeds, shift_magnitude))
        print(
            "| %s | %.4f | %.4f | %.1f%% | %.1f | %.1f | %.1f | %.4f | %.3f | %.3f |"
            % ((label,) + row)
        )


def fixed_rate_table(base: Config, rates: List[float], seeds: List[int]) -> None:
    print("\n## Fixed low-rate frontier")
    print("| eta_N | Brief drift | Adaptation steps | Final error |")
    print("|---:|---:|---:|---:|")
    for rate in rates:
        config = replace(base, eta_n_low=rate)
        result = aggregate(config, seeds)["fixed_slow_low"]
        print(
            "| %.4f | %.4f | %.1f | %.4f |"
            % (
                rate,
                result["transient_n_drift"],
                result["sustained_adaptation_steps"],
                result["sustained_final_error"],
            )
        )


def scale_comparison_table(
    base: Config, magnitudes: List[float], seeds: List[int]
) -> None:
    print("\n## Absolute versus scale-aware magnitude response")
    print(
        "| RMS shift | Absolute drift | Scaled drift | Absolute adapt | "
        "Scaled adapt | Absolute final | Scaled final | Scaled brief gate | "
        "Scaled sustained gate |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for magnitude in magnitudes:
        result = aggregate(base, seeds, magnitude)
        absolute = result["idl"]
        scaled = result["idl_scaled"]
        print(
            "| %.2f | %.4f | %.4f | %.1f | %.1f | %.4f | %.4f | %.3f | %.3f |"
            % (
                magnitude,
                absolute["transient_n_drift"],
                scaled["transient_n_drift"],
                absolute["sustained_adaptation_steps"],
                scaled["sustained_adaptation_steps"],
                absolute["sustained_final_error"],
                scaled["sustained_final_error"],
                scaled["transient_gate_mean"],
                scaled["sustained_early_gate_mean"],
            )
        )


def scale_parameter_table(
    title: str,
    parameter: str,
    conditions: Iterable[Tuple[str, Config]],
    seeds: List[int],
) -> None:
    print("\n## %s" % title)
    print(
        "| %s | Brief drift at shift 2.0 | Adapt at shift 0.25 | "
        "Final error at 0.25 | Adapt at shift 2.0 | Final error at 2.0 |"
        % parameter
    )
    print("|---:|---:|---:|---:|---:|---:|")
    for label, config in conditions:
        small = aggregate(config, seeds, 0.25)["idl_scaled"]
        default = aggregate(config, seeds, 2.0)["idl_scaled"]
        print(
            "| %s | %.4f | %.1f | %.4f | %.1f | %.4f |"
            % (
                label,
                default["transient_n_drift"],
                small["sustained_adaptation_steps"],
                small["sustained_final_error"],
                default["sustained_adaptation_steps"],
                default["sustained_final_error"],
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = Config()
    print("# Phase-two IDL sensitivity sweeps")
    print("seeds: %s" % ", ".join(str(seed) for seed in args.seeds))
    print("All sweeps vary one factor from the documented default configuration.")

    transient_lengths = [2, 5, 10, 20, 40, 80]
    print_table(
        "Brief-disturbance duration",
        "steps",
        (
            (
                str(length),
                replace(base, schedule=replace(base.schedule, transient=length)),
                2.0,
            )
            for length in transient_lengths
        ),
        args.seeds,
    )

    shift_magnitudes = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
    print_table(
        "Regime-shift magnitude",
        "RMS shift",
        (("%.2f" % value, base, value) for value in shift_magnitudes),
        args.seeds,
    )

    betas = [0.98, 0.99, 0.995, 0.9975, 0.999]
    print_table(
        "Persistence decay",
        "beta",
        ((str(value), replace(base, beta=value), 2.0) for value in betas),
        args.seeds,
    )

    thresholds = [0.05, 0.10, 0.20, 0.40, 0.80]
    print_table(
        "Retention threshold",
        "theta",
        (("%.2f" % value, replace(base, threshold=value), 2.0) for value in thresholds),
        args.seeds,
    )

    fixed_rate_table(base, [0.002, 0.004, 0.008, 0.012, 0.020], args.seeds)
    scale_comparison_table(base, shift_magnitudes, args.seeds)
    scale_parameter_table(
        "Scale-aware noise margin",
        "margin",
        ((str(value), replace(base, scale_margin=value)) for value in [2.0, 3.0, 4.0, 6.0]),
        args.seeds,
    )
    scale_parameter_table(
        "Scale estimator update rate",
        "rate",
        (
            (str(value), replace(base, scale_update_rate=value))
            for value in [0.002, 0.005, 0.01, 0.05]
        ),
        args.seeds,
    )


if __name__ == "__main__":
    main()
