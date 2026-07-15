"""Phase 6: category search, veto, and persistence-gated creation.

The experiment is an online, open-world classification stream.  Three stable
categories are learned first.  A sequence of coherent but brief categories is
then presented, followed by one genuinely persistent new category.  Every
learner predicts before seeing the label and may update only on learning
phases.

The Chevron learner uses A keys for attention, N templates for a vigilance
match, ART-like reset/search, and a fast candidate trace.  A candidate becomes
a retained category only after its mismatch has remained coherent and
persistent.  The principal comparison is a standard Q/K/V prototype-attention
memory with immediate writes and approximately the same vector-state budget.
A plain online MLP is included as a deliberately generous fixed-vocabulary
control.
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


UNKNOWN = -1
METHODS = (
    "chevron_art",
    "persistent_attention",
    "standard_attention",
    "online_mlp",
)


@dataclass(frozen=True)
class TaskConfig:
    dimension: int = 16
    base_categories: int = 3
    transient_categories: int = 8
    warmup_per_category: int = 24
    mixed_warmup: int = 72
    transient_length: int = 5
    recovery_probe: int = 96
    persistent_length: int = 40
    final_probe: int = 160
    noise_std: float = 0.035
    min_code_distance: int = 5

    @property
    def persistent_label(self) -> int:
        return self.base_categories + self.transient_categories

    @property
    def num_categories(self) -> int:
        return self.persistent_label + 1


@dataclass(frozen=True)
class MemoryConfig:
    retained_slots: int = 4
    vigilance: float = 0.105
    min_complementarity: float = 0.94
    candidate_vigilance: float = 0.075
    candidate_beta: float = 0.80
    creation_threshold: float = 0.75
    eta_a: float = 0.25
    eta_n: float = 0.04
    attention_temperature: float = 0.04

    @property
    def standard_slots(self) -> int:
        # Four Chevron slots hold an A key and an N template.  The additional
        # fast candidate holds one A vector: 2*4+1 = nine feature vectors.
        return 2 * self.retained_slots + 1


@dataclass(frozen=True)
class MLPConfig:
    hidden: int = 32
    learning_rate: float = 0.035
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class StreamItem:
    phase: str
    x: Tensor
    label: int
    learn: bool


@dataclass(frozen=True)
class Prediction:
    label: int
    slot: Optional[int] = None
    resets: int = 0
    mismatch: float = math.nan
    complementarity: float = math.nan
    used_candidate: bool = False


@dataclass(frozen=True)
class Record:
    phase: str
    target: int
    prediction: int
    correct: bool
    learn: bool
    resets: int
    used_candidate: bool


@dataclass
class CategorySlot:
    a: Tensor
    n: Tensor
    label: int
    last_used: int
    observations: int = 1


@dataclass
class FastCandidate:
    a: Tensor
    label: int
    persistence: float
    observations: int = 1


@dataclass
class RunResult:
    records: List[Record]
    snapshots: Dict[str, Tuple[int, ...]]
    creations: int
    evictions: int
    resets: int


def complement_code(x: Tensor) -> Tensor:
    """Non-negative coding makes the A/N contrast well defined."""

    return torch.cat([x.clamp(0.0, 1.0), 1.0 - x.clamp(0.0, 1.0)])


def contrast_components(a: Tensor, n: Tensor, epsilon: float = 1e-8) -> Tuple[Tensor, Tensor]:
    """Return signed contrast C and normalized complementarity G per component.

    Symmetric smoothing preserves C = 2p - 1 exactly, where
    p = (A + epsilon/2) / (A + N + epsilon).
    """

    engagement = a + n + epsilon
    p = (a + epsilon / 2.0) / engagement
    contrast = 2.0 * p - 1.0
    complementarity = 2.0 * torch.sqrt((p * (1.0 - p)).clamp_min(0.0))
    return contrast, complementarity


def contrast_match(a: Tensor, n: Tensor, epsilon: float = 1e-8) -> Tuple[float, float]:
    """Aggregate direction-free mismatch and complementarity using engagement."""

    contrast, gate = contrast_components(a, n, epsilon)
    weights = a + n + epsilon
    weights = weights / weights.sum()
    mismatch = float((weights * contrast.abs()).sum().item())
    complementarity = float((weights * gate).sum().item())
    return mismatch, complementarity


def cosine_scores(query: Tensor, keys: Sequence[Tensor]) -> Tensor:
    matrix = torch.stack(list(keys))
    return F.cosine_similarity(query.unsqueeze(0), matrix, dim=-1)


class CategoryStream:
    """Deterministic phase structure with seed-dependent prototypes and noise."""

    def __init__(self, config: TaskConfig, seed: int):
        self.config = config
        self.seed = seed
        self._rng = random.Random(seed + 10_000)
        self._torch_generator = torch.Generator().manual_seed(seed + 20_000)
        self.prototypes = self._make_prototypes()

    def _make_prototypes(self) -> Tensor:
        codes: List[Tensor] = []
        attempts = 0
        while len(codes) < self.config.num_categories:
            attempts += 1
            if attempts > 20_000:
                raise RuntimeError("could not construct separated category codes")
            candidate = torch.randint(
                0,
                2,
                (self.config.dimension,),
                generator=self._torch_generator,
                dtype=torch.float32,
            )
            if all(
                int(candidate.ne(code).sum().item()) >= self.config.min_code_distance
                for code in codes
            ):
                codes.append(candidate)
        binary = torch.stack(codes)
        return 0.16 + 0.68 * binary

    def _sample(self, label: int) -> Tensor:
        noise = torch.randn(
            self.config.dimension, generator=self._torch_generator
        ) * self.config.noise_std
        return (self.prototypes[label] + noise).clamp(0.0, 1.0)

    def _items(self, phase: str, labels: Iterable[int], learn: bool) -> List[StreamItem]:
        return [StreamItem(phase, self._sample(label), label, learn) for label in labels]

    def build(self) -> List[StreamItem]:
        c = self.config
        stream: List[StreamItem] = []
        for label in range(c.base_categories):
            stream += self._items(
                "warmup", [label] * c.warmup_per_category, learn=True
            )
        mixed = [i % c.base_categories for i in range(c.mixed_warmup)]
        self._rng.shuffle(mixed)
        stream += self._items("mixed_warmup", mixed, learn=True)

        transient_labels = range(c.base_categories, c.persistent_label)
        for label in transient_labels:
            stream += self._items(
                "transient", [label] * c.transient_length, learn=True
            )

        recovery = [i % c.base_categories for i in range(c.recovery_probe)]
        self._rng.shuffle(recovery)
        stream += self._items("recovery_probe", recovery, learn=False)

        stream += self._items(
            "persistent_new", [c.persistent_label] * c.persistent_length, learn=True
        )
        final_labels = list(range(c.base_categories)) + [c.persistent_label]
        final = [final_labels[i % len(final_labels)] for i in range(c.final_probe)]
        self._rng.shuffle(final)
        stream += self._items("final_probe", final, learn=False)
        return stream


class ChevronARTMemory:
    """A-key attention followed by N-template vigilance and reset/search."""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.slots: List[CategorySlot] = []
        self.candidate: Optional[FastCandidate] = None
        self.step = 0
        self.creations = 0
        self.evictions = 0
        self.total_resets = 0

    @property
    def retained_labels(self) -> Tuple[int, ...]:
        return tuple(sorted(slot.label for slot in self.slots))

    def _ranked_slots(self, a: Tensor) -> List[int]:
        if not self.slots:
            return []
        scores = cosine_scores(a, [slot.a for slot in self.slots])
        return scores.argsort(descending=True).tolist()

    def _resonant(self, a: Tensor, slot: CategorySlot) -> Tuple[bool, float, float]:
        mismatch, gate = contrast_match(a, slot.n)
        return (
            mismatch <= self.config.vigilance
            and gate >= self.config.min_complementarity,
            mismatch,
            gate,
        )

    def predict(self, x: Tensor) -> Prediction:
        a = complement_code(x)
        resets = 0
        last_mismatch = math.nan
        last_gate = math.nan
        for index in self._ranked_slots(a):
            resonates, mismatch, gate = self._resonant(a, self.slots[index])
            last_mismatch, last_gate = mismatch, gate
            if resonates:
                self.total_resets += resets
                return Prediction(
                    self.slots[index].label, index, resets, mismatch, gate, False
                )
            resets += 1

        self.total_resets += resets
        if self.candidate is not None:
            mismatch, gate = contrast_match(a, self.candidate.a)
            if mismatch <= self.config.candidate_vigilance:
                return Prediction(
                    self.candidate.label,
                    None,
                    resets,
                    mismatch,
                    gate,
                    True,
                )
        return Prediction(UNKNOWN, None, resets, last_mismatch, last_gate, False)

    def _update_slot(self, index: int, a: Tensor) -> None:
        slot = self.slots[index]
        slot.a = (1.0 - self.config.eta_a) * slot.a + self.config.eta_a * a
        slot.n = (1.0 - self.config.eta_n) * slot.n + self.config.eta_n * a
        slot.last_used = self.step
        slot.observations += 1

    def _find_labeled_resonance(self, a: Tensor, label: int) -> Optional[int]:
        for index in self._ranked_slots(a):
            slot = self.slots[index]
            if slot.label == label and self._resonant(a, slot)[0]:
                return index
        return None

    def _create(self, a: Tensor, label: int) -> None:
        if len(self.slots) >= self.config.retained_slots:
            victim = min(range(len(self.slots)), key=lambda i: self.slots[i].last_used)
            self.slots.pop(victim)
            self.evictions += 1
        self.slots.append(CategorySlot(a.clone(), a.clone(), label, self.step))
        self.creations += 1

    def observe(self, x: Tensor, label: int, prediction: Prediction) -> None:
        self.step += 1
        a = complement_code(x)
        index = self._find_labeled_resonance(a, label)
        if index is not None:
            self._update_slot(index, a)
            if self.candidate is not None:
                self.candidate.persistence *= self.config.candidate_beta
            return

        coherent = False
        if self.candidate is not None and self.candidate.label == label:
            mismatch, _gate = contrast_match(a, self.candidate.a)
            coherent = mismatch <= self.config.candidate_vigilance

        if coherent and self.candidate is not None:
            self.candidate.a = (
                (1.0 - self.config.eta_a) * self.candidate.a
                + self.config.eta_a * a
            )
            self.candidate.persistence = (
                self.config.candidate_beta * self.candidate.persistence
                + (1.0 - self.config.candidate_beta)
            )
            self.candidate.observations += 1
        else:
            self.candidate = FastCandidate(
                a.clone(), label, 1.0 - self.config.candidate_beta
            )

        if self.candidate.persistence >= self.config.creation_threshold:
            self._create(self.candidate.a, self.candidate.label)
            self.candidate = None


class StandardAttentionMemory:
    """Ordinary softmax Q/K/V category memory with immediate supervised writes."""

    def __init__(self, config: MemoryConfig, num_categories: int):
        self.config = config
        self.num_categories = num_categories
        self.slots: List[CategorySlot] = []
        self.step = 0
        self.creations = 0
        self.evictions = 0

    @property
    def retained_labels(self) -> Tuple[int, ...]:
        return tuple(sorted(slot.label for slot in self.slots))

    def predict(self, x: Tensor) -> Prediction:
        if not self.slots:
            return Prediction(UNKNOWN)
        q = complement_code(x)
        scores = cosine_scores(q, [slot.a for slot in self.slots])
        weights = (scores / self.config.attention_temperature).softmax(dim=0)
        values = F.one_hot(
            torch.tensor([slot.label for slot in self.slots]),
            num_classes=self.num_categories,
        ).float()
        label = int((weights.unsqueeze(1) * values).sum(0).argmax().item())
        slot = int(weights.argmax().item())
        mismatch, gate = contrast_match(q, self.slots[slot].a)
        return Prediction(label, slot, 0, mismatch, gate, False)

    def observe(self, x: Tensor, label: int, prediction: Prediction) -> None:
        self.step += 1
        a = complement_code(x)
        same_label = [i for i, slot in enumerate(self.slots) if slot.label == label]
        if same_label:
            scores = cosine_scores(a, [self.slots[i].a for i in same_label])
            index = same_label[int(scores.argmax().item())]
            slot = self.slots[index]
            slot.a = (1.0 - self.config.eta_a) * slot.a + self.config.eta_a * a
            slot.n = slot.a
            slot.last_used = self.step
            slot.observations += 1
            return

        if len(self.slots) >= self.config.standard_slots:
            victim = min(range(len(self.slots)), key=lambda i: self.slots[i].last_used)
            self.slots.pop(victim)
            self.evictions += 1
        self.slots.append(CategorySlot(a.clone(), a.clone(), label, self.step))
        self.creations += 1


class PersistentAttentionMemory(ChevronARTMemory):
    """Single-template attention with the same persistent creation rule.

    This is the decisive ablation. It keeps vigilance, search, and a fast
    candidate, but removes the distinct A key / N template roles. It also gets
    eight retained prototypes plus one candidate, matching Chevron's nine
    feature-vector state budget.
    """

    @property
    def retained_capacity(self) -> int:
        return self.config.standard_slots - 1

    def _resonant(self, a: Tensor, slot: CategorySlot) -> Tuple[bool, float, float]:
        mismatch, gate = contrast_match(a, slot.a)
        return (
            mismatch <= self.config.vigilance
            and gate >= self.config.min_complementarity,
            mismatch,
            gate,
        )

    def _update_slot(self, index: int, a: Tensor) -> None:
        slot = self.slots[index]
        slot.a = (1.0 - self.config.eta_a) * slot.a + self.config.eta_a * a
        slot.n = slot.a
        slot.last_used = self.step
        slot.observations += 1

    def _create(self, a: Tensor, label: int) -> None:
        if len(self.slots) >= self.retained_capacity:
            victim = min(range(len(self.slots)), key=lambda i: self.slots[i].last_used)
            self.slots.pop(victim)
            self.evictions += 1
        self.slots.append(CategorySlot(a.clone(), a.clone(), label, self.step))
        self.creations += 1


class OnlineMLP:
    """Plain supervised online MLP with the full label vocabulary in advance."""

    def __init__(self, dimension: int, num_categories: int, config: MLPConfig, seed: int):
        torch.manual_seed(seed + 30_000)
        self.model = nn.Sequential(
            nn.Linear(dimension, config.hidden),
            nn.GELU(),
            nn.Linear(config.hidden, num_categories),
        )
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=0.8,
            weight_decay=config.weight_decay,
        )

    @property
    def retained_labels(self) -> Tuple[int, ...]:
        return ()

    def predict(self, x: Tensor) -> Prediction:
        self.model.eval()
        with torch.no_grad():
            label = int(self.model(x.unsqueeze(0)).argmax(-1).item())
        return Prediction(label)

    def observe(self, x: Tensor, label: int, prediction: Prediction) -> None:
        self.model.train()
        logits = self.model(x.unsqueeze(0))
        loss = F.cross_entropy(logits, torch.tensor([label]))
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()


def build_method(
    method: str,
    task: TaskConfig,
    memory: MemoryConfig,
    mlp: MLPConfig,
    seed: int,
):
    if method == "chevron_art":
        return ChevronARTMemory(memory)
    if method == "persistent_attention":
        return PersistentAttentionMemory(memory)
    if method == "standard_attention":
        return StandardAttentionMemory(memory, task.num_categories)
    if method == "online_mlp":
        return OnlineMLP(task.dimension, task.num_categories, mlp, seed)
    raise ValueError("unknown method: %s" % method)


def run_method(
    method: str,
    stream: Sequence[StreamItem],
    task: TaskConfig,
    memory: MemoryConfig,
    mlp: MLPConfig,
    seed: int,
) -> RunResult:
    learner = build_method(method, task, memory, mlp, seed)
    records: List[Record] = []
    snapshots: Dict[str, Tuple[int, ...]] = {}
    previous_phase: Optional[str] = None
    for item in stream:
        if previous_phase is not None and item.phase != previous_phase:
            snapshots["after_" + previous_phase] = learner.retained_labels
        prediction = learner.predict(item.x)
        records.append(
            Record(
                item.phase,
                item.label,
                prediction.label,
                prediction.label == item.label,
                item.learn,
                prediction.resets,
                prediction.used_candidate,
            )
        )
        if item.learn:
            learner.observe(item.x, item.label, prediction)
        previous_phase = item.phase
    if previous_phase is not None:
        snapshots["after_" + previous_phase] = learner.retained_labels
    return RunResult(
        records,
        snapshots,
        getattr(learner, "creations", 0),
        getattr(learner, "evictions", 0),
        getattr(learner, "total_resets", 0),
    )


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def accuracy(records: Sequence[Record], phase: str, labels: Optional[Sequence[int]] = None) -> float:
    allowed = set(labels) if labels is not None else None
    selected = [
        record
        for record in records
        if record.phase == phase and (allowed is None or record.target in allowed)
    ]
    return mean(float(record.correct) for record in selected)


def metrics(result: RunResult, task: TaskConfig) -> Dict[str, float]:
    after_transients = result.snapshots.get("after_transient", ())
    final_labels = result.snapshots.get("after_final_probe", ())
    base = list(range(task.base_categories))
    return {
        "transient_online_accuracy": accuracy(result.records, "transient"),
        "old_retention_after_transients": accuracy(result.records, "recovery_probe"),
        "persistent_online_accuracy": accuracy(result.records, "persistent_new"),
        "final_old_accuracy": accuracy(result.records, "final_probe", base),
        "final_new_accuracy": accuracy(
            result.records, "final_probe", [task.persistent_label]
        ),
        "final_accuracy": accuracy(result.records, "final_probe"),
        "base_categories_retained": float(sum(label in after_transients for label in base)),
        "transient_categories_retained": float(
            sum(task.base_categories <= label < task.persistent_label for label in after_transients)
        ),
        "persistent_category_retained": float(task.persistent_label in final_labels),
        "category_creations": float(result.creations),
        "evictions": float(result.evictions),
        "vigilance_resets": float(result.resets),
    }


def summarize(rows: Sequence[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    return {
        key: (
            statistics.mean(row[key] for row in rows),
            statistics.stdev(row[key] for row in rows) if len(rows) > 1 else 0.0,
        )
        for key in rows[0]
    }


def print_summary(per_method: Dict[str, List[Dict[str, float]]]) -> None:
    keys = (
        "transient_online_accuracy",
        "old_retention_after_transients",
        "persistent_online_accuracy",
        "final_old_accuracy",
        "final_new_accuracy",
        "final_accuracy",
        "base_categories_retained",
        "transient_categories_retained",
        "persistent_category_retained",
        "category_creations",
        "evictions",
        "vigilance_resets",
    )
    for method in METHODS:
        print("\n" + method)
        summary = summarize(per_method[method])
        for key in keys:
            average, spread = summary[key]
            print("  %-34s %.4f +/- %.4f" % (key, average, spread))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task = TaskConfig()
    memory = MemoryConfig()
    mlp = MLPConfig()
    per_method: Dict[str, List[Dict[str, float]]] = {method: [] for method in METHODS}
    for seed in args.seeds:
        stream = CategoryStream(task, seed).build()
        for method in METHODS:
            result = run_method(method, stream, task, memory, mlp, seed)
            per_method[method].append(metrics(result, task))
    print_summary(per_method)


if __name__ == "__main__":
    main()
