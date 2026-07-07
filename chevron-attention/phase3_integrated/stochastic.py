"""Stochastic Phase 3 integrated experiment.

This version makes seeds matter beyond key/value assignment:

* multiple keys each have an independent retained permission gate;
* override episodes choose a random key, action, and duration;
* short overrides should affect current behavior but not retained policy;
* long overrides should consolidate into retained policy;
* no-context probes after each episode test what retained N remembers.
"""

from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from phase3_integrated.experiment import (
    METHODS,
    Config,
    answer_from_gate,
    make_values,
    update_baseline_state,
    update_idl_state,
)


@dataclass(frozen=True)
class StochasticConfig:
    base: Config = Config()
    episodes: int = 40
    pre_gap_min: int = 3
    pre_gap_max: int = 12
    probe_min: int = 6
    probe_max: int = 14
    short_min: int = 2
    short_max: int = 12
    long_min: int = 55
    long_max: int = 95
    distractor_probability: float = 0.30


@dataclass(frozen=True)
class StochasticRecord:
    step: int
    episode: int
    kind: str
    query_key: int
    controlled_key: Optional[int]
    contextual_gate: Optional[float]
    expected_retained_gate: float
    current_gate: float
    retained_gate_after_update: float
    answer: int
    expected_answer: int
    update_gate: float


@dataclass
class StochasticTrace:
    method: str
    values: List[int]
    records: List[StochasticRecord]


def random_query_key(
    rng: random.Random,
    controlled_key: Optional[int],
    config: StochasticConfig,
) -> int:
    if controlled_key is not None and rng.random() >= config.distractor_probability:
        return controlled_key
    return rng.randrange(config.base.num_keys)


def run_stochastic_method(
    method: str,
    config: StochasticConfig,
    seed: int,
) -> StochasticTrace:
    if method not in METHODS:
        raise ValueError("unknown method: %s" % method)

    rng = random.Random(seed + 20_000)
    values = make_values(config.base, seed)
    retained_gates = [1.0 for _ in range(config.base.num_keys)]
    expected_retained = [1.0 for _ in range(config.base.num_keys)]
    persistence = [0.0 for _ in range(config.base.num_keys)]
    difference_scale: List[Optional[float]] = [
        None for _ in range(config.base.num_keys)
    ]
    records: List[StochasticRecord] = []
    step = 0

    def append_record(
        episode: int,
        kind: str,
        query_key: int,
        controlled_key: Optional[int],
        contextual_gate: Optional[float],
    ) -> None:
        nonlocal step
        retained_before = retained_gates[query_key]
        current_gate = retained_before if contextual_gate is None else contextual_gate
        expected_current_gate = (
            expected_retained[query_key]
            if contextual_gate is None
            else contextual_gate
        )
        value = values[query_key]
        answer = answer_from_gate(value, current_gate, config.base.idk_value)
        expected_answer = answer_from_gate(
            value, expected_current_gate, config.base.idk_value
        )

        update_gate = 0.0
        if controlled_key is not None and contextual_gate is not None:
            if method == "integrated_idl":
                (
                    retained_gates[controlled_key],
                    _difference,
                    _evidence,
                    difference_scale[controlled_key],
                    persistence[controlled_key],
                    update_gate,
                ) = update_idl_state(
                    retained_gates[controlled_key],
                    contextual_gate,
                    persistence[controlled_key],
                    difference_scale[controlled_key],
                    config.base,
                )
            else:
                (
                    retained_gates[controlled_key],
                    _difference,
                    _evidence,
                    _scale,
                    _persistence,
                    update_gate,
                ) = update_baseline_state(
                    method,
                    retained_gates[controlled_key],
                    contextual_gate,
                    config.base,
                )
        else:
            # Decay persistence for all keys during gaps/probes. There is no
            # retained-policy target when no contextual control is present.
            if method == "integrated_idl":
                for key in range(config.base.num_keys):
                    persistence[key] *= config.base.beta

        records.append(
            StochasticRecord(
                step=step,
                episode=episode,
                kind=kind,
                query_key=query_key,
                controlled_key=controlled_key,
                contextual_gate=contextual_gate,
                expected_retained_gate=expected_retained[query_key],
                current_gate=current_gate,
                retained_gate_after_update=retained_gates[query_key],
                answer=answer,
                expected_answer=expected_answer,
                update_gate=update_gate,
            )
        )
        step += 1

    for episode in range(config.episodes):
        controlled_key = rng.randrange(config.base.num_keys)
        target_gate = 1.0 - expected_retained[controlled_key]
        is_long = episode % 2 == 1
        if rng.random() < 0.5:
            is_long = not is_long
        duration = rng.randint(
            config.long_min if is_long else config.short_min,
            config.long_max if is_long else config.short_max,
        )
        label = "long" if is_long else "short"

        for _ in range(rng.randint(config.pre_gap_min, config.pre_gap_max)):
            query_key = random_query_key(rng, controlled_key, config)
            append_record(episode, "gap", query_key, None, None)

        for _ in range(duration):
            query_key = random_query_key(rng, controlled_key, config)
            contextual_gate = target_gate if query_key == controlled_key else None
            append_record(
                episode,
                "%s_context" % label,
                query_key,
                controlled_key,
                contextual_gate,
            )

        if is_long:
            expected_retained[controlled_key] = target_gate

        for _ in range(rng.randint(config.probe_min, config.probe_max)):
            query_key = random_query_key(rng, controlled_key, config)
            append_record(episode, "%s_probe" % label, query_key, controlled_key, None)

    return StochasticTrace(method=method, values=values, records=records)


def accuracy(records: Iterable[StochasticRecord]) -> float:
    items = list(records)
    if not items:
        return 0.0
    return sum(record.answer == record.expected_answer for record in items) / len(items)


def mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def metrics(trace: StochasticTrace) -> Dict[str, float]:
    records = trace.records
    short_context = [
        record
        for record in records
        if record.kind == "short_context" and record.contextual_gate is not None
    ]
    long_context = [
        record
        for record in records
        if record.kind == "long_context" and record.contextual_gate is not None
    ]
    short_probe = [
        record
        for record in records
        if record.kind == "short_probe" and record.query_key == record.controlled_key
    ]
    long_probe = [
        record
        for record in records
        if record.kind == "long_probe" and record.query_key == record.controlled_key
    ]
    return {
        "answer_accuracy": accuracy(records),
        "short_context_accuracy": accuracy(short_context),
        "short_probe_preserve_accuracy": accuracy(short_probe),
        "long_context_accuracy": accuracy(long_context),
        "long_probe_consolidate_accuracy": accuracy(long_probe),
        "short_context_update_gate": mean(
            record.update_gate for record in short_context
        ),
        "long_context_update_gate": mean(record.update_gate for record in long_context),
        "records": float(len(records)),
    }


def run_stochastic_seed(
    config: StochasticConfig,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    return {
        method: metrics(run_stochastic_method(method, config, seed))
        for method in METHODS
    }


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
            values = [result[method][metric_name] for result in results]
            metric_mean, metric_sd = summarize(values)
            print("  %s=%.4f+/-%.4f" % (metric_name, metric_mean, metric_sd))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--short-max", type=int, default=12)
    parser.add_argument("--long-min", type=int, default=55)
    parser.add_argument("--long-max", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StochasticConfig(
        episodes=args.episodes,
        short_max=args.short_max,
        long_min=args.long_min,
        long_max=args.long_max,
    )
    results = []
    for seed in args.seeds:
        result = run_stochastic_seed(config, seed)
        results.append(result)
        print_seed(seed, result)
    print_summary(results)


if __name__ == "__main__":
    main()
