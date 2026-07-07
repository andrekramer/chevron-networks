"""Phase 4: learned contextual N gates driving IDL retention.

Phase 3 supplied the contextual gate directly. Phase 4 replaces that supplied
gate with a small learned sequence model. Retrieval remains algorithmic so the
experiment isolates one question:

Can a learned contextual N module infer revoke/restore state from tokens, and
can that inferred gate drive the same short/long retention behavior?
"""

from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from phase3_integrated.experiment import (
    METHODS,
    Config,
    answer_from_gate,
    make_values,
    update_baseline_state,
    update_idl_state,
)


PAD = 0
BOS = 1
QUERY = 2
REVOKE = 3
RESTORE = 4
KEY_OFFSET = 5
NO_CONTEXT = 0
REVOKED = 1
RESTORED = 2


@dataclass(frozen=True)
class GateTrainConfig:
    num_keys: int = 12
    max_controls: int = 4
    batch_size: int = 128
    steps: int = 700
    learning_rate: float = 3e-3
    d_model: int = 48
    hidden_size: int = 64
    grad_clip: float = 1.0

    @property
    def vocab_size(self) -> int:
        return KEY_OFFSET + self.num_keys

    @property
    def max_length(self) -> int:
        return 1 + 2 * self.max_controls + 2


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
    gate_confidence: float = 0.50


@dataclass(frozen=True)
class Record:
    step: int
    episode: int
    kind: str
    query_key: int
    controlled_key: Optional[int]
    oracle_contextual_gate: Optional[float]
    learned_contextual_gate: Optional[float]
    expected_retained_gate: float
    current_gate: float
    answer: int
    expected_answer: int
    update_gate: float
    gate_class: int
    expected_gate_class: int


@dataclass
class Trace:
    method: str
    records: List[Record]


class ContextGateModel(nn.Module):
    def __init__(self, config: GateTrainConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=PAD)
        self.rnn = nn.GRU(config.d_model, config.hidden_size, batch_first=True)
        self.output = nn.Linear(config.hidden_size, 3)

    def forward(self, tokens: Tensor) -> Tensor:
        embedded = self.embedding(tokens)
        _sequence, hidden = self.rnn(embedded)
        return self.output(hidden[-1])


def key_token(key: int) -> int:
    return KEY_OFFSET + key


def pad(tokens: List[int], max_length: int) -> List[int]:
    if len(tokens) > max_length:
        raise ValueError("token sequence exceeds max_length")
    return tokens + [PAD] * (max_length - len(tokens))


def make_gate_example(
    rng: random.Random,
    config: GateTrainConfig,
) -> Tuple[List[int], int]:
    query_key = rng.randrange(config.num_keys)
    target_class = rng.randrange(3)
    controls: List[Tuple[int, int]] = []

    if target_class == REVOKED:
        controls.append((REVOKE, query_key))
    elif target_class == RESTORED:
        controls.append((RESTORE, query_key))

    distractor_count = rng.randrange(config.max_controls)
    for _ in range(distractor_count):
        key = rng.randrange(config.num_keys)
        if key == query_key:
            key = (key + 1 + rng.randrange(config.num_keys - 1)) % config.num_keys
        controls.append((rng.choice([REVOKE, RESTORE]), key))
    rng.shuffle(controls)

    tokens = [BOS]
    for op, key in controls[: config.max_controls]:
        tokens.extend([op, key_token(key)])
    tokens.extend([QUERY, key_token(query_key)])
    return pad(tokens, config.max_length), target_class


def make_gate_batch(
    rng: random.Random,
    config: GateTrainConfig,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    rows = [make_gate_example(rng, config) for _ in range(config.batch_size)]
    tokens, targets = zip(*rows)
    return (
        torch.tensor(tokens, dtype=torch.long, device=device),
        torch.tensor(targets, dtype=torch.long, device=device),
    )


def build_runtime_tokens(
    gate_config: GateTrainConfig,
    query_key: int,
    controlled_key: Optional[int],
    oracle_contextual_gate: Optional[float],
) -> Tuple[List[int], int]:
    tokens = [BOS]
    expected_class = NO_CONTEXT
    if controlled_key is not None and oracle_contextual_gate is not None:
        op = RESTORE if oracle_contextual_gate >= 0.5 else REVOKE
        tokens.extend([op, key_token(controlled_key)])
        if query_key == controlled_key:
            expected_class = RESTORED if oracle_contextual_gate >= 0.5 else REVOKED
    tokens.extend([QUERY, key_token(query_key)])
    return pad(tokens, gate_config.max_length), expected_class


def train_gate_model(
    gate_config: GateTrainConfig,
    seed: int,
    device: torch.device,
) -> ContextGateModel:
    rng = random.Random(seed + 100_000)
    torch.manual_seed(seed + 200_000)
    model = ContextGateModel(gate_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=gate_config.learning_rate)
    model.train()
    for _ in range(gate_config.steps):
        tokens, targets = make_gate_batch(rng, gate_config, device)
        logits = model(tokens)
        loss = F.cross_entropy(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), gate_config.grad_clip)
        optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def predict_contextual_gate(
    model: ContextGateModel,
    gate_config: GateTrainConfig,
    stochastic_config: StochasticConfig,
    query_key: int,
    controlled_key: Optional[int],
    oracle_contextual_gate: Optional[float],
    device: torch.device,
) -> Tuple[Optional[float], int, int]:
    tokens, expected_class = build_runtime_tokens(
        gate_config, query_key, controlled_key, oracle_contextual_gate
    )
    token_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
    probabilities = model(token_tensor).softmax(dim=-1)[0]
    confidence, predicted_class = probabilities.max(dim=0)
    gate_class = int(predicted_class.item())
    if gate_class == NO_CONTEXT or confidence.item() < stochastic_config.gate_confidence:
        return None, gate_class, expected_class
    return (0.0 if gate_class == REVOKED else 1.0), gate_class, expected_class


def random_query_key(
    rng: random.Random,
    controlled_key: Optional[int],
    config: StochasticConfig,
) -> int:
    if controlled_key is not None and rng.random() >= config.distractor_probability:
        return controlled_key
    return rng.randrange(config.base.num_keys)


def run_method(
    method: str,
    gate_model: ContextGateModel,
    gate_config: GateTrainConfig,
    config: StochasticConfig,
    seed: int,
    device: torch.device,
) -> Trace:
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
    records: List[Record] = []
    step = 0

    def append_record(
        episode: int,
        kind: str,
        query_key: int,
        controlled_key: Optional[int],
        oracle_contextual_gate: Optional[float],
    ) -> None:
        nonlocal step
        learned_gate, gate_class, expected_gate_class = predict_contextual_gate(
            gate_model,
            gate_config,
            config,
            query_key,
            controlled_key,
            oracle_contextual_gate,
            device,
        )
        retained_before = retained_gates[query_key]
        current_gate = retained_before if learned_gate is None else learned_gate
        expected_current_gate = (
            expected_retained[query_key]
            if oracle_contextual_gate is None
            else oracle_contextual_gate
        )
        value = values[query_key]
        answer = answer_from_gate(value, current_gate, config.base.idk_value)
        expected_answer = answer_from_gate(
            value, expected_current_gate, config.base.idk_value
        )

        update_gate = 0.0
        if controlled_key is not None and learned_gate is not None:
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
                    learned_gate,
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
                    learned_gate,
                    config.base,
                )
        elif method == "integrated_idl":
            for key in range(config.base.num_keys):
                persistence[key] *= config.base.beta

        records.append(
            Record(
                step=step,
                episode=episode,
                kind=kind,
                query_key=query_key,
                controlled_key=controlled_key,
                oracle_contextual_gate=oracle_contextual_gate,
                learned_contextual_gate=learned_gate,
                expected_retained_gate=expected_retained[query_key],
                current_gate=current_gate,
                answer=answer,
                expected_answer=expected_answer,
                update_gate=update_gate,
                gate_class=gate_class,
                expected_gate_class=expected_gate_class,
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
            oracle_gate = target_gate if query_key == controlled_key else None
            append_record(
                episode,
                "%s_context" % label,
                query_key,
                controlled_key,
                oracle_gate,
            )

        if is_long:
            expected_retained[controlled_key] = target_gate

        for _ in range(rng.randint(config.probe_min, config.probe_max)):
            query_key = random_query_key(rng, controlled_key, config)
            append_record(episode, "%s_probe" % label, query_key, controlled_key, None)

    return Trace(method=method, records=records)


def accuracy(records: Iterable[Record]) -> float:
    items = list(records)
    if not items:
        return 0.0
    return sum(record.answer == record.expected_answer for record in items) / len(items)


def class_accuracy(records: Iterable[Record]) -> float:
    items = list(records)
    if not items:
        return 0.0
    return sum(record.gate_class == record.expected_gate_class for record in items) / len(items)


def mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)


def metrics(trace: Trace) -> Dict[str, float]:
    records = trace.records
    short_context = [
        r for r in records if r.kind == "short_context" and r.oracle_contextual_gate is not None
    ]
    long_context = [
        r for r in records if r.kind == "long_context" and r.oracle_contextual_gate is not None
    ]
    short_probe = [
        r for r in records if r.kind == "short_probe" and r.query_key == r.controlled_key
    ]
    long_probe = [
        r for r in records if r.kind == "long_probe" and r.query_key == r.controlled_key
    ]
    return {
        "answer_accuracy": accuracy(records),
        "gate_class_accuracy": class_accuracy(records),
        "context_gate_accuracy": class_accuracy(short_context + long_context),
        "short_context_accuracy": accuracy(short_context),
        "short_probe_preserve_accuracy": accuracy(short_probe),
        "long_context_accuracy": accuracy(long_context),
        "long_probe_consolidate_accuracy": accuracy(long_probe),
        "short_context_update_gate": mean(r.update_gate for r in short_context),
        "long_context_update_gate": mean(r.update_gate for r in long_context),
        "records": float(len(records)),
    }


def run_seed(
    stochastic_config: StochasticConfig,
    gate_config: GateTrainConfig,
    seed: int,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    gate_model = train_gate_model(gate_config, seed, device)
    return {
        method: metrics(
            run_method(method, gate_model, gate_config, stochastic_config, seed, device)
        )
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
            metric_values = [result[method][metric_name] for result in results]
            metric_mean, metric_sd = summarize(metric_values)
            print("  %s=%.4f+/-%.4f" % (metric_name, metric_mean, metric_sd))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--gate-steps", type=int, default=700)
    parser.add_argument("--device", choices=["cpu", "mps", "auto"], default="auto")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return torch.device("mps")
    if name == "auto" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    stochastic_config = StochasticConfig(episodes=args.episodes)
    gate_config = GateTrainConfig(steps=args.gate_steps)
    results = []
    for seed in args.seeds:
        result = run_seed(stochastic_config, gate_config, seed, device)
        results.append(result)
        print_seed(seed, result)
    print_summary(results)


if __name__ == "__main__":
    main()
