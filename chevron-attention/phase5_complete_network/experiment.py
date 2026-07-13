"""Phase 5: learned retrieval, permission, and persistent retention.

The fast A path supplies query/key attention.  The retained N path supplies
values and a learned contextual permission state.  A small explicit IDL state
machine decides whether contextual permission becomes the retained default.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


NO_CONTEXT = 0
REVOKED = 1
RESTORED = 2
PAD_OP = 0
REVOKE_OP = 1
RESTORE_OP = 2
METHODS = ("integrated_idl", "always_update", "fixed_slow", "context_only")


@dataclass(frozen=True)
class TaskConfig:
    num_keys: int = 12
    num_values: int = 12
    num_facts: int = 6
    max_controls: int = 4

    @property
    def idk_class(self) -> int:
        return self.num_values


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 700
    batch_size: int = 128
    d_model: int = 48
    learning_rate: float = 3e-3
    retrieval_weight: float = 1.0
    context_weight: float = 1.0
    grad_clip: float = 1.0


@dataclass(frozen=True)
class IDLConfig:
    beta: float = 0.985
    threshold: float = 0.35
    sharpness: float = 24.0
    eta_n: float = 0.08
    eta_fixed: float = 0.008
    evidence_margin: float = 0.25
    persistence_mode: str = "hard_reset"


@dataclass(frozen=True)
class CycleConfig:
    stable: int = 20
    short_revoke: int = 10
    short_probe: int = 20
    long_revoke: int = 70
    revoke_probe: int = 25
    long_restore: int = 70
    final_probe: int = 25
    target_query_probability: float = 0.75

    @property
    def phases(self) -> Tuple[Tuple[str, int], ...]:
        return (
            ("stable", self.stable),
            ("short_revoke", self.short_revoke),
            ("short_probe", self.short_probe),
            ("long_revoke", self.long_revoke),
            ("revoke_probe", self.revoke_probe),
            ("long_restore", self.long_restore),
            ("final_probe", self.final_probe),
        )


@dataclass(frozen=True)
class SignalNoise:
    """Perturbations applied only to the contextual signal used for retention."""

    gaussian_std: float = 0.0
    dropout_probability: float = 0.0
    flip_probability: float = 0.0


@dataclass
class Batch:
    fact_keys: Tensor
    fact_values: Tensor
    query_keys: Tensor
    control_keys: Tensor
    control_ops: Tensor
    retained_gates: Tensor
    context_classes: Tensor
    target_slots: Tensor
    answers: Tensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(**{name: value.to(device) for name, value in vars(self).items()})


class RecallAndControlTask:
    """Fresh associative memories with independently varied retained policy."""

    def __init__(self, config: TaskConfig):
        if config.num_facts > min(config.num_keys, config.num_values):
            raise ValueError("num_facts must fit within both key and value vocabularies")
        self.config = config

    def _row(self, rng: random.Random) -> Tuple[List[int], ...]:
        c = self.config
        keys = rng.sample(range(c.num_keys), c.num_facts)
        values = rng.sample(range(c.num_values), c.num_facts)
        retained = [float(rng.randrange(2)) for _ in keys]
        target_slot = rng.randrange(c.num_facts)

        context = [NO_CONTEXT for _ in keys]
        target_context = rng.randrange(3)
        controls: List[Tuple[int, int]] = []
        if target_context != NO_CONTEXT:
            context[target_slot] = target_context
            controls.append(
                (keys[target_slot], REVOKE_OP if target_context == REVOKED else RESTORE_OP)
            )

        available = [slot for slot in range(c.num_facts) if slot != target_slot]
        rng.shuffle(available)
        for slot in available[: rng.randrange(c.max_controls)]:
            state = rng.choice((REVOKED, RESTORED))
            context[slot] = state
            controls.append((keys[slot], REVOKE_OP if state == REVOKED else RESTORE_OP))
        rng.shuffle(controls)

        control_keys = [key for key, _op in controls]
        control_ops = [op for _key, op in controls]
        control_keys += [c.num_keys] * (c.max_controls - len(controls))
        control_ops += [PAD_OP] * (c.max_controls - len(controls))

        target_gate = (
            retained[target_slot]
            if target_context == NO_CONTEXT
            else float(target_context == RESTORED)
        )
        answer = values[target_slot] if target_gate >= 0.5 else c.idk_class
        return (
            keys,
            values,
            [keys[target_slot]],
            control_keys,
            control_ops,
            retained,
            context,
            [target_slot],
            [answer],
        )

    def batch(
        self, batch_size: int, rng: random.Random, device: Optional[torch.device] = None
    ) -> Batch:
        rows = [self._row(rng) for _ in range(batch_size)]
        fields = list(zip(*rows))
        batch = Batch(
            fact_keys=torch.tensor(fields[0], dtype=torch.long),
            fact_values=torch.tensor(fields[1], dtype=torch.long),
            query_keys=torch.tensor(fields[2], dtype=torch.long).squeeze(-1),
            control_keys=torch.tensor(fields[3], dtype=torch.long),
            control_ops=torch.tensor(fields[4], dtype=torch.long),
            retained_gates=torch.tensor(fields[5], dtype=torch.float32),
            context_classes=torch.tensor(fields[6], dtype=torch.long),
            target_slots=torch.tensor(fields[7], dtype=torch.long).squeeze(-1),
            answers=torch.tensor(fields[8], dtype=torch.long).squeeze(-1),
        )
        return batch.to(device) if device is not None else batch

    def runtime_batch(
        self,
        fact_keys: Sequence[int],
        fact_values: Sequence[int],
        query_key: int,
        retained_gates: Sequence[float],
        control: Optional[Tuple[int, int]],
        device: torch.device,
    ) -> Batch:
        c = self.config
        slot_by_key = {key: slot for slot, key in enumerate(fact_keys)}
        context = [NO_CONTEXT for _ in fact_keys]
        control_keys = [c.num_keys] * c.max_controls
        control_ops = [PAD_OP] * c.max_controls
        if control is not None:
            key, op = control
            control_keys[0], control_ops[0] = key, op
            context[slot_by_key[key]] = REVOKED if op == REVOKE_OP else RESTORED
        target_slot = slot_by_key[query_key]
        state = context[target_slot]
        gate = retained_gates[target_slot] if state == NO_CONTEXT else float(state == RESTORED)
        answer = fact_values[target_slot] if gate >= 0.5 else c.idk_class
        return Batch(
            fact_keys=torch.tensor([fact_keys], dtype=torch.long, device=device),
            fact_values=torch.tensor([fact_values], dtype=torch.long, device=device),
            query_keys=torch.tensor([query_key], dtype=torch.long, device=device),
            control_keys=torch.tensor([control_keys], dtype=torch.long, device=device),
            control_ops=torch.tensor([control_ops], dtype=torch.long, device=device),
            retained_gates=torch.tensor([retained_gates], dtype=torch.float32, device=device),
            context_classes=torch.tensor([context], dtype=torch.long, device=device),
            target_slots=torch.tensor([target_slot], dtype=torch.long, device=device),
            answers=torch.tensor([answer], dtype=torch.long, device=device),
        )


class ChevronMemoryNetwork(nn.Module):
    """A supplies Q/K; N supplies V and contextual permission."""

    def __init__(self, task: TaskConfig, d_model: int = 48):
        super().__init__()
        self.task = task
        self.scale = math.sqrt(d_model)

        self.a_keys = nn.Embedding(task.num_keys, d_model)
        self.q_a = nn.Linear(d_model, d_model, bias=False)
        self.k_a = nn.Linear(d_model, d_model, bias=False)

        self.n_values = nn.Embedding(task.num_values, d_model)
        self.v_n = nn.Linear(d_model, d_model, bias=False)

        self.n_control_keys = nn.Embedding(task.num_keys + 1, d_model, padding_idx=task.num_keys)
        self.n_control_ops = nn.Embedding(3, d_model, padding_idx=PAD_OP)
        self.q_n = nn.Linear(d_model, d_model, bias=False)
        self.k_n = nn.Linear(d_model, d_model, bias=False)
        self.null_control_logit = nn.Parameter(torch.tensor(1.0))
        self.null_control = nn.Parameter(torch.zeros(d_model))
        self.context_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 3)
        )

        self.null_value = nn.Parameter(torch.zeros(d_model))
        self.answer_head = nn.Linear(d_model, task.num_values + 1)

    def forward(self, batch: Batch) -> Dict[str, Tensor]:
        fact_a = self.a_keys(batch.fact_keys)
        query_a = self.a_keys(batch.query_keys)
        q = self.q_a(query_a).unsqueeze(1)
        k = self.k_a(fact_a)
        retrieval_logits = (q * k).sum(-1) / self.scale
        alpha = retrieval_logits.softmax(dim=-1)

        # N reads key-addressed control events.  A learned null event represents
        # the absence of context for a fact slot.
        control_key = self.n_control_keys(batch.control_keys)
        control_event = control_key + self.n_control_ops(batch.control_ops)
        q_context = self.q_n(self.n_control_keys(batch.fact_keys))
        k_context = self.k_n(control_key)
        event_logits = torch.einsum("bsd,bcd->bsc", q_context, k_context) / self.scale
        event_logits = event_logits.masked_fill(batch.control_ops.eq(PAD_OP).unsqueeze(1), -1e4)
        null_logits = self.null_control_logit.expand(
            batch.fact_keys.size(0), batch.fact_keys.size(1), 1
        )
        event_weights = torch.cat([event_logits, null_logits], dim=-1).softmax(dim=-1)
        null_event = self.null_control.expand(batch.fact_keys.size(0), 1, -1)
        all_events = torch.cat([control_event, null_event], dim=1)
        attended_control = torch.einsum("bsc,bcd->bsd", event_weights, all_events)
        context_logits = self.context_head(torch.cat([q_context, attended_control], dim=-1))
        context_probabilities = context_logits.softmax(dim=-1)

        # Soft selection keeps the entire offline objective differentiable.
        current_gates = (
            context_probabilities[..., NO_CONTEXT] * batch.retained_gates
            + context_probabilities[..., RESTORED]
        )
        values = self.v_n(self.n_values(batch.fact_values))
        admitted = alpha * current_gates
        admitted_mass = admitted.sum(-1, keepdim=True)
        output_value = (admitted.unsqueeze(-1) * values).sum(1)
        output_value = output_value + (1.0 - admitted_mass) * self.null_value
        return {
            "answer_logits": self.answer_head(output_value),
            "retrieval_logits": retrieval_logits,
            "alpha": alpha,
            "context_logits": context_logits,
            "context_probabilities": context_probabilities,
            "current_gates": current_gates,
            "admitted_mass": admitted_mass.squeeze(-1),
            "event_weights": event_weights,
        }


def losses(outputs: Dict[str, Tensor], batch: Batch, config: TrainConfig) -> Dict[str, Tensor]:
    answer = F.cross_entropy(outputs["answer_logits"], batch.answers)
    retrieval = F.cross_entropy(outputs["retrieval_logits"], batch.target_slots)
    context = F.cross_entropy(
        outputs["context_logits"].reshape(-1, 3), batch.context_classes.reshape(-1)
    )
    total = answer + config.retrieval_weight * retrieval + config.context_weight * context
    return {"total": total, "answer": answer, "retrieval": retrieval, "context": context}


@torch.no_grad()
def evaluate(
    model: ChevronMemoryNetwork,
    task: RecallAndControlTask,
    rng: random.Random,
    device: torch.device,
    batches: int = 20,
    batch_size: int = 128,
) -> Dict[str, float]:
    model.eval()
    totals = {"count": 0, "answer": 0, "retrieval": 0, "context": 0, "slots": 0, "alpha": 0.0}
    for _ in range(batches):
        batch = task.batch(batch_size, rng, device)
        outputs = model(batch)
        row = torch.arange(batch_size, device=device)
        totals["count"] += batch_size
        totals["answer"] += outputs["answer_logits"].argmax(-1).eq(batch.answers).sum().item()
        totals["retrieval"] += outputs["alpha"].argmax(-1).eq(batch.target_slots).sum().item()
        totals["context"] += outputs["context_logits"].argmax(-1).eq(batch.context_classes).sum().item()
        totals["slots"] += batch_size * task.config.num_facts
        totals["alpha"] += outputs["alpha"][row, batch.target_slots].sum().item()
    return {
        "answer_accuracy": totals["answer"] / totals["count"],
        "retrieval_accuracy": totals["retrieval"] / totals["count"],
        "context_accuracy": totals["context"] / totals["slots"],
        "target_alpha": totals["alpha"] / totals["count"],
    }


def train_model(
    task: RecallAndControlTask,
    config: TrainConfig,
    seed: int,
    device: torch.device,
) -> ChevronMemoryNetwork:
    random.seed(seed)
    torch.manual_seed(seed)
    rng = random.Random(seed + 1_000)
    model = ChevronMemoryNetwork(task.config, config.d_model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    model.train()
    for _ in range(config.steps):
        batch = task.batch(config.batch_size, rng, device)
        objective = losses(model(batch), batch, config)["total"]
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
    model.eval()
    return model


@dataclass
class RetainedState:
    gates: List[float]
    persistence: List[float]
    directions: List[int]
    positive_persistence: List[float]
    negative_persistence: List[float]


@dataclass(frozen=True)
class CycleRecord:
    phase: str
    query_is_target: bool
    answer_correct: bool
    retrieval_correct: bool
    context_correct: bool
    target_alpha: float
    retained_target: float
    update_gate: float


def update_retained(
    method: str,
    state: RetainedState,
    slot: int,
    contextual_gate: Optional[float],
    config: IDLConfig,
) -> float:
    if contextual_gate is None:
        if method == "integrated_idl":
            if config.persistence_mode == "two_trace":
                state.positive_persistence[slot] *= config.beta
                state.negative_persistence[slot] *= config.beta
            else:
                state.persistence[slot] *= config.beta
        return 0.0
    difference = contextual_gate - state.gates[slot]
    direction = 1 if difference > 0 else -1 if difference < 0 else 0
    if method == "integrated_idl":
        evidence = min(1.0, abs(difference) / config.evidence_margin)
        if config.persistence_mode == "hard_reset":
            if direction and state.directions[slot] not in (0, direction):
                state.persistence[slot] = 0.0
            state.directions[slot] = direction
            state.persistence[slot] = (
                config.beta * state.persistence[slot] + (1.0 - config.beta) * evidence
            )
            persistence = state.persistence[slot]
        elif config.persistence_mode == "two_trace":
            positive_evidence = evidence if direction > 0 else 0.0
            negative_evidence = evidence if direction < 0 else 0.0
            state.positive_persistence[slot] = (
                config.beta * state.positive_persistence[slot]
                + (1.0 - config.beta) * positive_evidence
            )
            state.negative_persistence[slot] = (
                config.beta * state.negative_persistence[slot]
                + (1.0 - config.beta) * negative_evidence
            )
            persistence = (
                state.positive_persistence[slot]
                if direction > 0
                else state.negative_persistence[slot]
            )
        elif config.persistence_mode == "signed_hysteresis":
            signed_evidence = direction * evidence
            state.persistence[slot] = (
                config.beta * state.persistence[slot]
                + (1.0 - config.beta) * signed_evidence
            )
            persistence = abs(state.persistence[slot])
            if state.persistence[slot] > 0.0:
                contextual_gate = 1.0
            elif state.persistence[slot] < 0.0:
                contextual_gate = 0.0
            difference = contextual_gate - state.gates[slot]
        else:
            raise ValueError("unknown persistence mode: %s" % config.persistence_mode)
        update_gate = torch.sigmoid(
            torch.tensor(config.sharpness * (persistence - config.threshold))
        ).item()
        eta = config.eta_n
    elif method == "always_update":
        update_gate, eta = 1.0, config.eta_n
    elif method == "fixed_slow":
        update_gate, eta = 1.0, config.eta_fixed
    elif method == "context_only":
        update_gate, eta = 0.0, 0.0
    else:
        raise ValueError("unknown method: %s" % method)
    state.gates[slot] += eta * update_gate * difference
    state.gates[slot] = min(1.0, max(0.0, state.gates[slot]))
    return update_gate


def phase_control(phase: str) -> Optional[int]:
    if phase in ("short_revoke", "long_revoke"):
        return REVOKE_OP
    if phase == "long_restore":
        return RESTORE_OP
    return None


def expected_target_gate(phase: str) -> float:
    return 0.0 if phase in ("short_revoke", "long_revoke", "revoke_probe") else 1.0


@torch.no_grad()
def run_cycle(
    method: str,
    model: ChevronMemoryNetwork,
    task: RecallAndControlTask,
    cycle: CycleConfig,
    idl: IDLConfig,
    seed: int,
    device: torch.device,
    signal_noise: SignalNoise = SignalNoise(),
) -> List[CycleRecord]:
    if method not in METHODS:
        raise ValueError("unknown method: %s" % method)
    rng = random.Random(seed + 20_000)
    c = task.config
    fact_keys = rng.sample(range(c.num_keys), c.num_facts)
    fact_values = rng.sample(range(c.num_values), c.num_facts)
    target_slot = 0
    target_key = fact_keys[target_slot]
    state = RetainedState(
        gates=[1.0 for _ in fact_keys],
        persistence=[0.0 for _ in fact_keys],
        directions=[0 for _ in fact_keys],
        positive_persistence=[0.0 for _ in fact_keys],
        negative_persistence=[0.0 for _ in fact_keys],
    )
    records: List[CycleRecord] = []

    for phase, duration in cycle.phases:
        op = phase_control(phase)
        for _ in range(duration):
            query_is_target = rng.random() < cycle.target_query_probability
            query_key = target_key if query_is_target else rng.choice(fact_keys[1:])
            control = None if op is None else (target_key, op)
            batch = task.runtime_batch(
                fact_keys, fact_values, query_key, state.gates, control, device
            )
            outputs = model(batch)
            predicted = int(outputs["answer_logits"].argmax(-1).item())
            predicted_slot = int(outputs["alpha"].argmax(-1).item())
            expected_slot = int(batch.target_slots.item())
            context_predictions = outputs["context_logits"].argmax(-1)[0]

            context_gate: Optional[float] = None
            context_correct = True
            if op is not None:
                predicted_class = int(context_predictions[target_slot].item())
                expected_class = REVOKED if op == REVOKE_OP else RESTORED
                context_correct = predicted_class == expected_class
                probabilities = outputs["context_probabilities"][0, target_slot]
                context_mass = float((probabilities[REVOKED] + probabilities[RESTORED]).item())
                if predicted_class != NO_CONTEXT and context_mass >= 0.5:
                    context_gate = float((probabilities[RESTORED] / context_mass).item())

            # This deliberately perturbs only the learned signal passed to the
            # retention rule. Current contextual behavior still comes from the
            # network, so probe failures isolate consolidation robustness.
            if context_gate is not None and signal_noise.dropout_probability:
                if rng.random() < signal_noise.dropout_probability:
                    context_gate = None
            if context_gate is not None and signal_noise.flip_probability:
                if rng.random() < signal_noise.flip_probability:
                    context_gate = 1.0 - context_gate
            if context_gate is not None and signal_noise.gaussian_std:
                context_gate += rng.gauss(0.0, signal_noise.gaussian_std)
                context_gate = min(1.0, max(0.0, context_gate))

            update_gate = update_retained(
                method, state, target_slot, context_gate, idl
            )
            expected_answer = batch.answers.item()
            if query_is_target:
                desired_gate = expected_target_gate(phase)
                expected_answer = (
                    fact_values[target_slot] if desired_gate >= 0.5 else c.idk_class
                )
            records.append(
                CycleRecord(
                    phase=phase,
                    query_is_target=query_is_target,
                    answer_correct=predicted == expected_answer,
                    retrieval_correct=predicted_slot == expected_slot,
                    context_correct=context_correct,
                    target_alpha=float(outputs["alpha"][0, expected_slot].item()),
                    retained_target=state.gates[target_slot],
                    update_gate=update_gate,
                )
            )
    return records


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def cycle_metrics(records: Sequence[CycleRecord]) -> Dict[str, float]:
    def phase_target(name: str) -> List[CycleRecord]:
        return [r for r in records if r.phase == name and r.query_is_target]

    controlled = [r for r in records if r.phase in ("short_revoke", "long_revoke", "long_restore")]
    phase_end = {
        name: [r for r in records if r.phase == name][-1].retained_target
        for name in ("short_revoke", "long_revoke", "long_restore")
    }
    short_accuracy = mean(float(r.answer_correct) for r in phase_target("short_probe"))
    revoke_accuracy = mean(float(r.answer_correct) for r in phase_target("revoke_probe"))
    final_active_accuracy = mean(float(r.answer_correct) for r in phase_target("final_probe"))
    return {
        "answer_accuracy": mean(float(r.answer_correct) for r in records),
        "retrieval_accuracy": mean(float(r.retrieval_correct) for r in records),
        "context_accuracy": mean(float(r.context_correct) for r in controlled),
        "target_alpha": mean(r.target_alpha for r in records),
        "short_probe_preserve": short_accuracy,
        "long_probe_consolidate": revoke_accuracy,
        # Kept for compatibility. This measures final active behavior; by
        # itself it does not prove restoration if revocation never consolidated.
        "restore_probe_consolidate": final_active_accuracy,
        "restore_probe_active": final_active_accuracy,
        "full_revoke_restore_cycle": float(
            revoke_accuracy > 0.5 and final_active_accuracy > 0.5
        ),
        "retained_after_short": phase_end["short_revoke"],
        "retained_after_long_revoke": phase_end["long_revoke"],
        "retained_after_long_restore": phase_end["long_restore"],
        "short_update_gate": mean(r.update_gate for r in records if r.phase == "short_revoke"),
        "long_update_gate": mean(r.update_gate for r in records if r.phase == "long_revoke"),
    }


def summarize(results: Sequence[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    summary = {}
    for key in results[0]:
        values = [result[key] for result in results]
        summary[key] = (
            statistics.mean(values),
            statistics.stdev(values) if len(values) > 1 else 0.0,
        )
    return summary


def print_demo(metrics_by_method: Dict[str, Dict[str, float]]) -> None:
    print("\nretained N permission (1=use fact, 0=abstain)")
    print("method          after short revoke   after long revoke   after long restore")
    for method in METHODS:
        m = metrics_by_method[method]
        print(
            "%-16s %18.3f %19.3f %20.3f"
            % (
                method,
                m["retained_after_short"],
                m["retained_after_long_revoke"],
                m["retained_after_long_restore"],
            )
        )
    print("\nprobe accuracy (short must preserve; long must consolidate)")
    print("method               short       revoke  final active  full cycle")
    for method in METHODS:
        m = metrics_by_method[method]
        print(
            "%-16s %11.3f %12.3f %13.3f %11.3f"
            % (
                method,
                m["short_probe_preserve"],
                m["long_probe_consolidate"],
                m["restore_probe_active"],
                m["full_revoke_restore_cycle"],
            )
        )


def select_device(name: str) -> torch.device:
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but unavailable")
        return torch.device("mps")
    if name == "auto" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    task = RecallAndControlTask(TaskConfig())
    train_config = TrainConfig(steps=args.steps)
    per_method: Dict[str, List[Dict[str, float]]] = {method: [] for method in METHODS}
    for seed in args.seeds:
        model = train_model(task, train_config, seed, device)
        held_out = evaluate(model, task, random.Random(seed + 90_000), device)
        print("seed=%d held_out %s" % (seed, " ".join("%s=%.4f" % x for x in held_out.items())))
        seed_metrics = {}
        for method in METHODS:
            metrics = cycle_metrics(
                run_cycle(method, model, task, CycleConfig(), IDLConfig(), seed, device)
            )
            per_method[method].append(metrics)
            seed_metrics[method] = metrics
        print_demo(seed_metrics)
    if len(args.seeds) > 1:
        print("\nsummary mean+/-sd")
        for method in METHODS:
            print(method)
            for name, (average, spread) in summarize(per_method[method]).items():
                print("  %s=%.4f+/-%.4f" % (name, average, spread))


if __name__ == "__main__":
    main()
