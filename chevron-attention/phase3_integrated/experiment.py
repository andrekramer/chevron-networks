"""Integrated retrieval, contextual gating, and retained-policy update test.

Phase one showed a clean split between A retrieval and N permission gating.
Phase two showed that persistent A/N discrepancy can regulate retained-state
updates. This phase combines those claims in one controlled online task:

* A always retrieves the queried fact value.
* Contextual N can immediately override current behavior.
* Retained N is updated only when that contextual override persists.

The task schedule is:

1. active fact use with no control context,
2. a brief revoke burst,
3. no context again, where the original active policy should survive,
4. a sustained revoke, where the retained policy should consolidate,
5. no context again, where revocation should persist,
6. a sustained restore, where the retained policy should consolidate back,
7. final no-context active behavior.
"""

from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor


METHODS = (
    "integrated_idl",
    "always_update",
    "fixed_slow",
    "context_only",
)


@dataclass(frozen=True)
class Schedule:
    stable: int = 80
    transient_revoke: int = 10
    recovery: int = 50
    sustained_revoke: int = 180
    post_revoke: int = 50
    sustained_restore: int = 180
    final: int = 50

    @property
    def total(self) -> int:
        return (
            self.stable
            + self.transient_revoke
            + self.recovery
            + self.sustained_revoke
            + self.post_revoke
            + self.sustained_restore
            + self.final
        )

    @property
    def transient_start(self) -> int:
        return self.stable

    @property
    def transient_end(self) -> int:
        return self.stable + self.transient_revoke

    @property
    def sustained_revoke_start(self) -> int:
        return self.transient_end + self.recovery

    @property
    def sustained_revoke_end(self) -> int:
        return self.sustained_revoke_start + self.sustained_revoke

    @property
    def sustained_restore_start(self) -> int:
        return self.sustained_revoke_end + self.post_revoke

    @property
    def sustained_restore_end(self) -> int:
        return self.sustained_restore_start + self.sustained_restore

    def phase(self, step: int) -> str:
        if step < self.transient_start:
            return "stable"
        if step < self.transient_end:
            return "transient_revoke"
        if step < self.sustained_revoke_start:
            return "recovery"
        if step < self.sustained_revoke_end:
            return "sustained_revoke"
        if step < self.sustained_restore_start:
            return "post_revoke"
        if step < self.sustained_restore_end:
            return "sustained_restore"
        return "final"


@dataclass(frozen=True)
class Config:
    num_keys: int = 12
    num_values: int = 12
    target_key: int = 0
    idk_value: int = -1
    eta_n: float = 0.08
    eta_n_low: float = 0.012
    beta: float = 0.985
    threshold: float = 0.35
    sharpness: float = 24.0
    scale_update_rate: float = 0.02
    scale_margin: float = 3.0
    scale_epsilon: float = 1e-6
    schedule: Schedule = Schedule()


@dataclass
class StepRecord:
    step: int
    phase: str
    query_key: int
    retrieved_value: int
    answer: int
    expected_answer: int
    retained_gate: float
    current_gate: float
    contextual_gate: Optional[float]
    difference: float
    evidence: float
    difference_scale: float
    persistence: float
    update_gate: float


@dataclass
class Trace:
    method: str
    values: List[int]
    records: List[StepRecord]


def make_values(config: Config, seed: int) -> List[int]:
    rng = random.Random(seed)
    values = list(range(config.num_values))
    rng.shuffle(values)
    return values[: config.num_keys]


def context_gate_for_phase(phase: str) -> Optional[float]:
    if phase in ("transient_revoke", "sustained_revoke"):
        return 0.0
    if phase == "sustained_restore":
        return 1.0
    return None


def expected_retained_gate_for_phase(phase: str) -> float:
    if phase == "post_revoke":
        return 0.0
    return 1.0


def answer_from_gate(value: int, gate: float, idk_value: int) -> int:
    return value if gate >= 0.5 else idk_value


def update_idl_state(
    retained_gate: float,
    contextual_gate: Optional[float],
    persistence: float,
    difference_scale: Optional[float],
    config: Config,
) -> Tuple[float, float, float, float, float, float]:
    if contextual_gate is None:
        # No contextual override means no retained-policy training signal. The
        # persistence trace decays so old anomalies do not leak into later phases.
        persistence = config.beta * persistence
        scale = difference_scale if difference_scale is not None else config.scale_epsilon
        return retained_gate, 0.0, 0.0, scale, persistence, 0.0

    difference = abs(contextual_gate - retained_gate)
    if difference_scale is None:
        difference_scale = max(difference * 0.05, config.scale_epsilon)

    ratio = difference / (difference_scale + config.scale_epsilon)
    evidence = min(1.0, max(0.0, ratio / config.scale_margin - 1.0))
    if evidence == 0.0:
        difference_scale += config.scale_update_rate * (difference - difference_scale)

    persistence = config.beta * persistence + (1.0 - config.beta) * evidence
    update_gate = torch.sigmoid(
        torch.tensor(config.sharpness * (persistence - config.threshold))
    ).item()
    retained_gate += config.eta_n * update_gate * (contextual_gate - retained_gate)
    return (
        min(1.0, max(0.0, retained_gate)),
        difference,
        evidence,
        difference_scale,
        persistence,
        update_gate,
    )


def update_baseline_state(
    method: str,
    retained_gate: float,
    contextual_gate: Optional[float],
    config: Config,
) -> Tuple[float, float, float, float, float, float]:
    if contextual_gate is None:
        return retained_gate, 0.0, 0.0, config.scale_epsilon, 0.0, 0.0
    difference = abs(contextual_gate - retained_gate)
    if method == "always_update":
        update_gate = 1.0
        eta = config.eta_n
    elif method == "fixed_slow":
        update_gate = 1.0
        eta = config.eta_n_low
    elif method == "context_only":
        update_gate = 0.0
        eta = 0.0
    else:
        raise ValueError("unknown baseline method: %s" % method)
    retained_gate += eta * update_gate * (contextual_gate - retained_gate)
    return (
        min(1.0, max(0.0, retained_gate)),
        difference,
        difference,
        config.scale_epsilon,
        difference,
        update_gate,
    )


def run_method(method: str, config: Config, seed: int) -> Trace:
    if method not in METHODS:
        raise ValueError("unknown method: %s" % method)

    values = make_values(config, seed)
    retained_gate = 1.0
    persistence = 0.0
    difference_scale: Optional[float] = None
    records: List[StepRecord] = []

    for step in range(config.schedule.total):
        phase = config.schedule.phase(step)
        query_key = config.target_key
        retrieved_value = values[query_key]
        contextual_gate = context_gate_for_phase(phase)
        current_gate = retained_gate if contextual_gate is None else contextual_gate
        expected_gate = (
            contextual_gate
            if contextual_gate is not None
            else expected_retained_gate_for_phase(phase)
        )
        answer = answer_from_gate(retrieved_value, current_gate, config.idk_value)
        expected_answer = answer_from_gate(
            retrieved_value, expected_gate, config.idk_value
        )

        if method == "integrated_idl":
            (
                retained_gate,
                difference,
                evidence,
                scale,
                persistence,
                update_gate,
            ) = update_idl_state(
                retained_gate,
                contextual_gate,
                persistence,
                difference_scale,
                config,
            )
            difference_scale = scale
        else:
            (
                retained_gate,
                difference,
                evidence,
                scale,
                persistence,
                update_gate,
            ) = update_baseline_state(method, retained_gate, contextual_gate, config)

        records.append(
            StepRecord(
                step=step,
                phase=phase,
                query_key=query_key,
                retrieved_value=retrieved_value,
                answer=answer,
                expected_answer=expected_answer,
                retained_gate=retained_gate,
                current_gate=current_gate,
                contextual_gate=contextual_gate,
                difference=difference,
                evidence=evidence,
                difference_scale=scale,
                persistence=persistence,
                update_gate=update_gate,
            )
        )

    return Trace(method=method, values=values, records=records)


def phase_records(trace: Trace, phase: str) -> List[StepRecord]:
    return [record for record in trace.records if record.phase == phase]


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items)


def first_step_where(
    records: List[StepRecord], predicate
) -> int:
    for record in records:
        if predicate(record):
            return record.step - records[0].step + 1
    return len(records) + 1


def metrics(trace: Trace, config: Config) -> Dict[str, float]:
    records = trace.records
    by_phase = {phase: phase_records(trace, phase) for phase in (
        "stable",
        "transient_revoke",
        "recovery",
        "sustained_revoke",
        "post_revoke",
        "sustained_restore",
        "final",
    )}
    target_value = trace.values[config.target_key]
    return {
        "retrieval_accuracy": mean(
            float(record.retrieved_value == target_value) for record in records
        ),
        "answer_accuracy": mean(
            float(record.answer == record.expected_answer) for record in records
        ),
        "transient_current_revoke_accuracy": mean(
            float(record.answer == config.idk_value)
            for record in by_phase["transient_revoke"]
        ),
        "recovery_active_accuracy": mean(
            float(record.answer == target_value) for record in by_phase["recovery"]
        ),
        "transient_retained_drift": 1.0
        - by_phase["transient_revoke"][-1].retained_gate,
        "sustained_revoke_current_accuracy": mean(
            float(record.answer == config.idk_value)
            for record in by_phase["sustained_revoke"]
        ),
        "post_revoke_retained_accuracy": mean(
            float(record.answer == config.idk_value)
            for record in by_phase["post_revoke"]
        ),
        "restore_current_accuracy": mean(
            float(record.answer == target_value)
            for record in by_phase["sustained_restore"]
        ),
        "final_active_accuracy": mean(
            float(record.answer == target_value) for record in by_phase["final"]
        ),
        "revoke_consolidation_steps": float(
            first_step_where(
                by_phase["sustained_revoke"], lambda record: record.retained_gate < 0.5
            )
        ),
        "restore_consolidation_steps": float(
            first_step_where(
                by_phase["sustained_restore"], lambda record: record.retained_gate >= 0.5
            )
        ),
        "transient_update_gate_mean": mean(
            record.update_gate for record in by_phase["transient_revoke"]
        ),
        "sustained_revoke_update_gate_mean": mean(
            record.update_gate for record in by_phase["sustained_revoke"][:60]
        ),
        "sustained_restore_update_gate_mean": mean(
            record.update_gate for record in by_phase["sustained_restore"][:60]
        ),
    }


def run_seed(config: Config, seed: int) -> Dict[str, Dict[str, float]]:
    return {method: metrics(run_method(method, config, seed), config) for method in METHODS}


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
    for method in METHODS:
        print(method)
        for metric_name in results[0][method]:
            metric_values = [result[method][metric_name] for result in results]
            metric_mean, metric_sd = summarize(metric_values)
            print("  %s=%.4f+/-%.4f" % (metric_name, metric_mean, metric_sd))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--eta-n", type=float, default=0.08)
    parser.add_argument("--eta-n-low", type=float, default=0.012)
    parser.add_argument("--beta", type=float, default=0.985)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--sharpness", type=float, default=24.0)
    parser.add_argument("--stable", type=int, default=80)
    parser.add_argument("--transient-revoke", type=int, default=10)
    parser.add_argument("--recovery", type=int, default=50)
    parser.add_argument("--sustained-revoke", type=int, default=180)
    parser.add_argument("--post-revoke", type=int, default=50)
    parser.add_argument("--sustained-restore", type=int, default=180)
    parser.add_argument("--final", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        eta_n=args.eta_n,
        eta_n_low=args.eta_n_low,
        beta=args.beta,
        threshold=args.threshold,
        sharpness=args.sharpness,
        schedule=Schedule(
            stable=args.stable,
            transient_revoke=args.transient_revoke,
            recovery=args.recovery,
            sustained_revoke=args.sustained_revoke,
            post_revoke=args.post_revoke,
            sustained_restore=args.sustained_restore,
            final=args.final,
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
