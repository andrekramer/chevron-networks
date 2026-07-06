"""Test whether persistent difference can regulate retained-state updates.

All methods share the same fast online learner A. They differ only in how the
retained state N follows A:

* IDL: N updates through a persistence-sensitive gate.
* idl_scaled: persistence tracks discrepancy relative to an online noise floor.
* always_slow: N follows A at every step with the same maximum update rate.
* fixed_slow_low: N always follows A at a lower rate chosen to reduce brief drift.
* fast_only: the current prediction is A; there is no distinct retained state.
"""

import argparse
import random
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class Schedule:
    stable: int = 600
    transient: int = 10
    recovery: int = 200
    sustained: int = 300

    @property
    def total(self) -> int:
        return self.stable + self.transient + self.recovery + self.sustained

    @property
    def transient_start(self) -> int:
        return self.stable

    @property
    def transient_end(self) -> int:
        return self.stable + self.transient

    @property
    def sustained_start(self) -> int:
        return self.stable + self.transient + self.recovery

    def phase(self, step: int) -> str:
        if step < self.transient_start:
            return "stable"
        if step < self.transient_end:
            return "transient"
        if step < self.sustained_start:
            return "recovery"
        return "sustained"


@dataclass(frozen=True)
class Config:
    dimensions: int = 16
    batch_size: int = 64
    noise_std: float = 0.05
    eta_a: float = 0.08
    eta_n: float = 0.02
    eta_n_low: float = 0.008
    beta: float = 0.995
    threshold: float = 0.20
    sharpness: float = 30.0
    scale_update_rate: float = 0.01
    scale_margin: float = 3.0
    scale_epsilon: float = 1e-6
    schedule: Schedule = Schedule()


@dataclass
class Trace:
    method: str
    a: List[Tensor]
    n: List[Tensor]
    difference: List[float]
    evidence: List[float]
    difference_scale: List[float]
    persistence: List[float]
    gate: List[float]
    current_error: List[float]
    retained_error: List[float]


def rms(vector: Tensor) -> float:
    return vector.square().mean().sqrt().item()


def make_regimes(dimensions: int, generator: torch.Generator) -> Tuple[Tensor, Tensor]:
    """Create two opposed mappings with unit RMS magnitude."""
    base = torch.randn(dimensions, generator=generator)
    base = base / base.square().mean().sqrt()
    return base, -base


def regime_at(step: int, schedule: Schedule, base: Tensor, changed: Tensor) -> Tensor:
    phase = schedule.phase(step)
    return changed if phase in ("transient", "sustained") else base


def online_fast_update(
    a: Tensor,
    target: Tensor,
    config: Config,
    generator: torch.Generator,
) -> Tensor:
    """One analytical SGD step for a linear squared-error predictor."""
    # Unit-variance features give the fast learner an O(1/eta_a) adaptation
    # timescale independent of dimensionality.
    x = torch.randn(config.batch_size, config.dimensions, generator=generator)
    noise = torch.randn(config.batch_size, generator=generator) * config.noise_std
    y = x.mv(target) + noise
    residual = x.mv(a) - y
    gradient = (2.0 / config.batch_size) * x.t().mv(residual)
    return a - config.eta_a * gradient


def run_method(
    method: str,
    config: Config,
    seed: int,
    regimes: Optional[Tuple[Tensor, Tensor]] = None,
) -> Trace:
    if method not in (
        "idl",
        "idl_scaled",
        "always_slow",
        "fixed_slow_low",
        "fast_only",
    ):
        raise ValueError("unknown method: %s" % method)

    regime_generator = torch.Generator().manual_seed(seed)
    data_generator = torch.Generator().manual_seed(seed + 1_000_000)
    base, changed = regimes or make_regimes(config.dimensions, regime_generator)
    # Phase two starts after the base mapping has been learned. The stable phase
    # measures ordinary online variation; it is not an acquisition benchmark.
    a = base.clone()
    n = base.clone()
    persistence = 0.0
    difference_scale: Optional[float] = None
    trace = Trace(method, [], [], [], [], [], [], [], [], [])

    for step in range(config.schedule.total):
        target = regime_at(step, config.schedule, base, changed)
        a = online_fast_update(a, target, config, data_generator)
        difference = rms(a - n)
        if method == "idl_scaled":
            if difference_scale is None:
                difference_scale = max(difference, config.scale_epsilon)
            ratio = difference / (difference_scale + config.scale_epsilon)
            # Zero inside the learned noise band, rising to one by twice the
            # margin. Freeze the scale while an anomaly is being accumulated.
            evidence = min(
                1.0,
                max(0.0, ratio / config.scale_margin - 1.0),
            )
            if evidence == 0.0:
                difference_scale += config.scale_update_rate * (
                    difference - difference_scale
                )
        else:
            if difference_scale is None:
                difference_scale = max(difference, config.scale_epsilon)
            evidence = difference
        persistence = (
            config.beta * persistence + (1.0 - config.beta) * evidence
        )

        if method in ("idl", "idl_scaled"):
            gate = torch.sigmoid(
                torch.tensor(config.sharpness * (persistence - config.threshold))
            ).item()
            n = n + config.eta_n * gate * (a - n)
            output = n
        elif method == "always_slow":
            gate = 1.0
            n = n + config.eta_n * (a - n)
            output = n
        elif method == "fixed_slow_low":
            gate = 1.0
            n = n + config.eta_n_low * (a - n)
            output = n
        else:
            gate = 1.0
            n = a.clone()
            output = a

        trace.a.append(a.clone())
        trace.n.append(n.clone())
        trace.difference.append(difference)
        trace.evidence.append(evidence)
        trace.difference_scale.append(difference_scale)
        trace.persistence.append(persistence)
        trace.gate.append(gate)
        trace.current_error.append(rms(a - target))
        trace.retained_error.append(rms(output - target))

    return trace


def adaptation_steps(trace: Trace, config: Config, base: Tensor, changed: Tensor) -> int:
    """Steps until retained error is at most 10% of shift magnitude."""
    tolerance = 0.10 * rms(changed - base)
    start = config.schedule.sustained_start
    for offset, state in enumerate(trace.n[start:]):
        if rms(state - changed) <= tolerance:
            return offset + 1
    return config.schedule.sustained + 1


def mean_slice(values: List[float], start: int, end: int) -> float:
    segment = values[start:end]
    return sum(segment) / len(segment)


def metrics(trace: Trace, config: Config, base: Tensor, changed: Tensor) -> Dict[str, float]:
    schedule = config.schedule
    before_transient = trace.n[schedule.transient_start - 1]
    after_transient = trace.n[schedule.transient_end - 1]
    before_sustained = trace.n[schedule.sustained_start - 1]
    final = trace.n[-1]
    persistent_early_end = min(schedule.total, schedule.sustained_start + 50)
    return {
        "pre_transient_base_error": rms(before_transient - base),
        "transient_n_drift": rms(after_transient - before_transient),
        "recovery_end_base_error": rms(before_sustained - base),
        "sustained_adaptation_steps": float(
            adaptation_steps(trace, config, base, changed)
        ),
        "sustained_final_error": rms(final - changed),
        "sustained_mean_error": mean_slice(
            trace.retained_error, schedule.sustained_start, schedule.total
        ),
        "transient_gate_mean": mean_slice(
            trace.gate, schedule.transient_start, schedule.transient_end
        ),
        "sustained_early_gate_mean": mean_slice(
            trace.gate, schedule.sustained_start, persistent_early_end
        ),
    }


def run_seed(
    config: Config, seed: int, shift_magnitude: float = 2.0
) -> Dict[str, Dict[str, float]]:
    generator = torch.Generator().manual_seed(seed)
    base, default_changed = make_regimes(config.dimensions, generator)
    direction = default_changed - base
    direction = direction / direction.square().mean().sqrt()
    regimes = (base, base + shift_magnitude * direction)
    result = {}
    for method in (
        "idl",
        "idl_scaled",
        "always_slow",
        "fixed_slow_low",
        "fast_only",
    ):
        trace = run_method(method, config, seed, regimes)
        result[method] = metrics(trace, config, regimes[0], regimes[1])
    return result


def summarize(values: Iterable[float]) -> Tuple[float, float]:
    items = list(values)
    spread = statistics.stdev(items) if len(items) > 1 else 0.0
    return statistics.mean(items), spread


def print_seed(seed: int, result: Dict[str, Dict[str, float]]) -> None:
    print("seed=%d" % seed)
    for method, values in result.items():
        formatted = " ".join("%s=%.4f" % item for item in values.items())
        print("  %s %s" % (method, formatted))


def print_summary(results: List[Dict[str, Dict[str, float]]]) -> None:
    print("summary mean+/-sd")
    for method in results[0]:
        print(method)
        for metric_name in results[0][method]:
            mean, spread = summarize(
                result[method][metric_name] for result in results
            )
            print("  %s=%.4f+/-%.4f" % (metric_name, mean, spread))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--dimensions", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--noise-std", type=float, default=0.05)
    parser.add_argument("--eta-a", type=float, default=0.08)
    parser.add_argument("--eta-n", type=float, default=0.02)
    parser.add_argument("--eta-n-low", type=float, default=0.008)
    parser.add_argument("--beta", type=float, default=0.995)
    parser.add_argument("--threshold", type=float, default=0.20)
    parser.add_argument("--sharpness", type=float, default=30.0)
    parser.add_argument("--scale-update-rate", type=float, default=0.01)
    parser.add_argument("--scale-margin", type=float, default=3.0)
    parser.add_argument("--stable", type=int, default=600)
    parser.add_argument("--transient", type=int, default=10)
    parser.add_argument("--recovery", type=int, default=200)
    parser.add_argument("--sustained", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        dimensions=args.dimensions,
        batch_size=args.batch_size,
        noise_std=args.noise_std,
        eta_a=args.eta_a,
        eta_n=args.eta_n,
        eta_n_low=args.eta_n_low,
        beta=args.beta,
        threshold=args.threshold,
        sharpness=args.sharpness,
        scale_update_rate=args.scale_update_rate,
        scale_margin=args.scale_margin,
        schedule=Schedule(
            stable=args.stable,
            transient=args.transient,
            recovery=args.recovery,
            sustained=args.sustained,
        ),
    )
    results = []
    for seed in args.seeds:
        result = run_seed(config, seed)
        results.append(result)
        print_seed(seed, result)
    print_summary(results)


if __name__ == "__main__":
    main()
