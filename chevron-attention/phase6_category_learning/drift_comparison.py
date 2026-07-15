"""Phase 6.1: recurring category drift and the value of two traces.

A known category moves to a temporary surface form, returns to its retained
form, and later makes the same move persistently.  The experiment asks whether
one representation can be plastic now and stable later, or whether separate A
and N traces are needed.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from phase6_category_learning.experiment import complement_code, contrast_match


UNKNOWN = -1
METHODS = (
    "chevron_dual",
    "chevron_soft",
    "standard_dual_attention",
    "persistent_single",
    "fast_single",
    "slow_single",
)


@dataclass(frozen=True)
class DriftTaskConfig:
    dimension: int = 2
    categories: int = 3
    warmup_per_category: int = 30
    mixed_warmup: int = 60
    short_shift: int = 10
    short_current_probe: int = 40
    base_recovery_probe: int = 40
    base_relearn: int = 18
    long_shift: int = 70
    final_shift_probe: int = 60
    retained_shift_probe: int = 60
    noise_std: float = 0.018
    shift_x: float = 0.50
    shift_y: float = 0.75


@dataclass(frozen=True)
class DriftConfig:
    eta_fast: float = 0.35
    eta_slow: float = 0.025
    eta_retained: float = 0.16
    beta: float = 0.95
    threshold: float = 0.55
    sharpness: float = 28.0
    evidence_margin: float = 0.30
    current_match: float = 0.10
    retained_match: float = 0.12
    attention_temperature: float = 0.025


@dataclass(frozen=True)
class DriftItem:
    phase: str
    x: Tensor
    label: int
    learn: bool


@dataclass(frozen=True)
class DriftRecord:
    phase: str
    target: int
    prediction: int
    correct: bool


def squared_distances(x: Tensor, prototypes: Tensor) -> Tensor:
    return ((prototypes - x.unsqueeze(0)) ** 2).sum(-1)


class DriftStream:
    """Three categories; category zero has base and shifted surface forms."""

    def __init__(self, config: DriftTaskConfig, seed: int):
        self.config = config
        self.rng = random.Random(seed + 40_000)
        self.generator = torch.Generator().manual_seed(seed + 50_000)
        self.base = torch.tensor([[0.10, 0.20], [0.50, 0.20], [0.90, 0.20]])
        self.shifted = torch.tensor([config.shift_x, config.shift_y])

    def sample(self, label: int, shifted: bool = False) -> Tensor:
        center = self.shifted if shifted else self.base[label]
        noise = torch.randn(self.config.dimension, generator=self.generator)
        return (center + self.config.noise_std * noise).clamp(0.0, 1.0)

    def items(
        self, phase: str, labels: Iterable[int], learn: bool, shifted: bool = False
    ) -> List[DriftItem]:
        return [
            DriftItem(phase, self.sample(label, shifted and label == 0), label, learn)
            for label in labels
        ]

    def build(self) -> List[DriftItem]:
        c = self.config
        stream: List[DriftItem] = []
        for label in range(c.categories):
            stream += self.items("warmup", [label] * c.warmup_per_category, True)
        mixed = [i % c.categories for i in range(c.mixed_warmup)]
        self.rng.shuffle(mixed)
        stream += self.items("mixed_warmup", mixed, True)
        stream += self.items("short_shift", [0] * c.short_shift, True, shifted=True)
        stream += self.items(
            "short_current_probe", [0] * c.short_current_probe, False, shifted=True
        )
        stream += self.items(
            "base_recovery_probe", [0] * c.base_recovery_probe, False
        )
        stream += self.items("base_relearn", [0] * c.base_relearn, True)
        stream += self.items("long_shift", [0] * c.long_shift, True, shifted=True)
        stream += self.items(
            "final_shift_probe", [0] * c.final_shift_probe, False, shifted=True
        )
        stream += self.items(
            "retained_shift_probe", [0] * c.retained_shift_probe, False, shifted=True
        )
        return stream


class SinglePrototypeMemory:
    def __init__(self, categories: int, rate: float):
        self.categories = categories
        self.rate = rate
        self.prototypes: List[Optional[Tensor]] = [None] * categories

    def predict(self, x: Tensor) -> int:
        active = [i for i, value in enumerate(self.prototypes) if value is not None]
        if not active:
            return UNKNOWN
        matrix = torch.stack([self.prototypes[i] for i in active])
        return active[int(squared_distances(x, matrix).argmin().item())]

    def observe(self, x: Tensor, label: int) -> None:
        if self.prototypes[label] is None:
            self.prototypes[label] = x.clone()
        else:
            value = self.prototypes[label]
            self.prototypes[label] = (1.0 - self.rate) * value + self.rate * x

    def predict_retained(self, x: Tensor) -> int:
        return self.predict(x)


class PersistentSingleMemory(SinglePrototypeMemory):
    """One retained template; persistent discrepancy gates its only update."""

    def __init__(self, categories: int, config: DriftConfig):
        super().__init__(categories, config.eta_retained)
        self.config = config
        self.persistence = [0.0] * categories

    def observe(self, x: Tensor, label: int) -> None:
        if self.prototypes[label] is None:
            self.prototypes[label] = x.clone()
            return
        n = self.prototypes[label]
        mismatch, _gate = contrast_match(complement_code(x), complement_code(n))
        evidence = min(1.0, mismatch / self.config.evidence_margin)
        self.persistence[label] = (
            self.config.beta * self.persistence[label]
            + (1.0 - self.config.beta) * evidence
        )
        update_gate = torch.sigmoid(
            torch.tensor(
                self.config.sharpness
                * (self.persistence[label] - self.config.threshold)
            )
        ).item()
        self.prototypes[label] = (
            n + self.config.eta_retained * update_gate * (x - n)
        )


class DualTraceMemory:
    """Fast A prototypes, retained N templates, and IDL consolidation."""

    def __init__(self, categories: int, config: DriftConfig, routing: str):
        self.categories = categories
        self.config = config
        self.routing = routing
        self.a: List[Optional[Tensor]] = [None] * categories
        self.n: List[Optional[Tensor]] = [None] * categories
        self.persistence = [0.0] * categories
        self.direction: List[Optional[Tensor]] = [None] * categories

    def _active(self) -> List[int]:
        return [i for i, value in enumerate(self.a) if value is not None]

    def predict(self, x: Tensor) -> int:
        active = self._active()
        if not active:
            return UNKNOWN
        if self.routing in ("standard_attention", "soft_contrast"):
            keys: List[Tensor] = []
            labels: List[int] = []
            for label in active:
                keys.extend([self.a[label], self.n[label]])
                labels.extend([label, label])
            distances = squared_distances(x, torch.stack(keys))
            logits = -distances / self.config.attention_temperature
            if self.routing == "soft_contrast":
                gates = []
                encoded_x = complement_code(x)
                for key in keys:
                    _mismatch, gate = contrast_match(encoded_x, complement_code(key))
                    gates.append(max(gate, 1e-6))
                logits = logits + torch.tensor(gates).log()
            weights = logits.softmax(0)
            votes = torch.zeros(self.categories)
            for weight, label in zip(weights, labels):
                votes[label] += weight
            return int(votes.argmax().item())

        # Chevron form: A determines the search order. Each category can
        # resonate through its current A surface or retained N template.
        a_matrix = torch.stack([self.a[i] for i in active])
        order = squared_distances(x, a_matrix).argsort().tolist()
        for position in order:
            label = active[position]
            a_mismatch, _ = contrast_match(
                complement_code(x), complement_code(self.a[label])
            )
            n_mismatch, _ = contrast_match(
                complement_code(x), complement_code(self.n[label])
            )
            if (
                a_mismatch <= self.config.current_match
                or n_mismatch <= self.config.retained_match
            ):
                return label
        return UNKNOWN

    def observe(self, x: Tensor, label: int) -> None:
        if self.a[label] is None:
            self.a[label] = x.clone()
            self.n[label] = x.clone()
            return

        old_a = self.a[label]
        n = self.n[label]
        self.a[label] = (1.0 - self.config.eta_fast) * old_a + self.config.eta_fast * x
        delta = self.a[label] - n
        norm = float(delta.norm().item())
        direction = delta / max(norm, 1e-8)
        previous = self.direction[label]
        if previous is not None and float(torch.dot(previous, direction).item()) < 0.0:
            self.persistence[label] = 0.0
        self.direction[label] = direction

        mismatch, _ = contrast_match(
            complement_code(self.a[label]), complement_code(n)
        )
        evidence = min(1.0, mismatch / self.config.evidence_margin)
        self.persistence[label] = (
            self.config.beta * self.persistence[label]
            + (1.0 - self.config.beta) * evidence
        )
        update_gate = torch.sigmoid(
            torch.tensor(
                self.config.sharpness
                * (self.persistence[label] - self.config.threshold)
            )
        ).item()
        self.n[label] = n + self.config.eta_retained * update_gate * (
            self.a[label] - n
        )

    def predict_retained(self, x: Tensor) -> int:
        active = self._active()
        if not active:
            return UNKNOWN
        matrix = torch.stack([self.n[i] for i in active])
        return active[int(squared_distances(x, matrix).argmin().item())]


def build_method(method: str, task: DriftTaskConfig, config: DriftConfig):
    if method == "chevron_dual":
        return DualTraceMemory(task.categories, config, routing="hard_search")
    if method == "chevron_soft":
        return DualTraceMemory(task.categories, config, routing="soft_contrast")
    if method == "standard_dual_attention":
        return DualTraceMemory(task.categories, config, routing="standard_attention")
    if method == "persistent_single":
        return PersistentSingleMemory(task.categories, config)
    if method == "fast_single":
        return SinglePrototypeMemory(task.categories, config.eta_fast)
    if method == "slow_single":
        return SinglePrototypeMemory(task.categories, config.eta_slow)
    raise ValueError("unknown method: %s" % method)


def run_method(
    method: str,
    stream: Sequence[DriftItem],
    task: DriftTaskConfig,
    config: DriftConfig,
) -> List[DriftRecord]:
    learner = build_method(method, task, config)
    records: List[DriftRecord] = []
    for item in stream:
        prediction = (
            learner.predict_retained(item.x)
            if item.phase == "retained_shift_probe"
            else learner.predict(item.x)
        )
        records.append(
            DriftRecord(item.phase, item.label, prediction, prediction == item.label)
        )
        if item.learn:
            learner.observe(item.x, item.label)
    return records


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def phase_accuracy(records: Sequence[DriftRecord], phase: str) -> float:
    return mean(float(record.correct) for record in records if record.phase == phase)


def adaptation_steps(records: Sequence[DriftRecord], phase: str, window: int = 5) -> float:
    selected = [record.correct for record in records if record.phase == phase]
    for index in range(len(selected) - window + 1):
        if all(selected[index : index + window]):
            return float(index)
    return math.inf


def metrics(records: Sequence[DriftRecord]) -> Dict[str, float]:
    current = phase_accuracy(records, "short_current_probe")
    recovery = phase_accuracy(records, "base_recovery_probe")
    return {
        "short_shift_online": phase_accuracy(records, "short_shift"),
        "short_current_probe": current,
        "base_recovery_probe": recovery,
        "short_cycle_min": min(current, recovery),
        "long_shift_online": phase_accuracy(records, "long_shift"),
        "long_adaptation_steps": adaptation_steps(records, "long_shift"),
        "final_shift_probe": phase_accuracy(records, "final_shift_probe"),
        "retained_shift_probe": phase_accuracy(records, "retained_shift_probe"),
    }


def summarize(rows: Sequence[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    return {
        key: (
            statistics.mean(row[key] for row in rows),
            statistics.stdev(row[key] for row in rows) if len(rows) > 1 else 0.0,
        )
        for key in rows[0]
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = DriftTaskConfig()
    config = DriftConfig()
    results: Dict[str, List[Dict[str, float]]] = {method: [] for method in METHODS}
    for seed in args.seeds:
        stream = DriftStream(task, seed).build()
        for method in METHODS:
            results[method].append(metrics(run_method(method, stream, task, config)))
    for method in METHODS:
        print("\n" + method)
        for key, (average, spread) in summarize(results[method]).items():
            print("  %-28s %.4f +/- %.4f" % (key, average, spread))


if __name__ == "__main__":
    main()
