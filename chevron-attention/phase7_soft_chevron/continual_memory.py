"""Phase 7.5: learned admission controls writes to persistent category memory.

Six categories begin in a stable nine-slot memory. Three related categories
then arrive online, followed by a noisy continual-learning shift. Models must
use their learned novelty signal to allocate a category or update retained
slots. Wrong decisions alter persistent keys and templates before final probes.
"""

from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor

from phase7_soft_chevron.experiment import (
    JOINT_ATTENTION,
    SOFT_CHEVRON,
    Batch,
    CategoryMatchingTask,
    JointAttention,
    SoftChevronAttention,
    TaskConfig,
    TrainConfig,
    parameter_count,
    train_model,
)


CHEVRON_WRITE = "chevron_write"
JOINT_NULL_WRITE = "joint_null_write"
JOINT_OUTPUT_WRITE = "joint_output_write"
POLICIES = (CHEVRON_WRITE, JOINT_NULL_WRITE, JOINT_OUTPUT_WRITE)


@dataclass(frozen=True)
class ContinualConfig:
    base_adaptation: int = 180
    novel_learning: int = 180
    shifted_learning: int = 270
    final_per_category: int = 40
    observation_noise: float = 0.035
    shifted_noise: float = 0.080
    key_write_rate: float = 0.08
    template_write_rate: float = 0.06
    calibration_batches: int = 12
    calibration_batch_size: int = 128
    memory_capacity: int = 9
    novel_flips: int = 2
    blocked_novel: bool = False


@dataclass
class Slot:
    key_a: Tensor
    template_n: Tensor
    value_id: int
    last_used: int


@dataclass
class PersistentMemory:
    task: TaskConfig
    slots: List[Slot]
    step: int = 0
    evictions: int = 0
    capacity: int = 0

    def __post_init__(self) -> None:
        if self.capacity == 0:
            self.capacity = self.task.num_slots

    @property
    def labels(self) -> Tuple[int, ...]:
        return tuple(slot.value_id for slot in self.slots)

    def batch(self, query_a: Tensor, match_a: Tensor) -> Tuple[Batch, Tensor]:
        count = len(self.slots)
        keys = torch.zeros(self.task.num_slots, self.task.a_dimension)
        templates = torch.zeros(self.task.num_slots, self.task.n_dimension)
        values = torch.zeros(self.task.num_slots, dtype=torch.long)
        active = torch.zeros(self.task.num_slots, dtype=torch.bool)
        for index, slot in enumerate(self.slots):
            keys[index] = slot.key_a
            templates[index] = slot.template_n
            values[index] = slot.value_id
            active[index] = True
        batch = Batch(
            query_a=query_a.unsqueeze(0),
            match_a=match_a.unsqueeze(0),
            keys_a=keys.unsqueeze(0),
            templates_n=templates.unsqueeze(0),
            value_ids=values.unsqueeze(0),
            target_groups=torch.zeros(1, dtype=torch.long),
            target_slots=torch.zeros(1, dtype=torch.long),
            matched=torch.zeros(1, dtype=torch.bool),
            answers=torch.zeros(1, dtype=torch.long),
        )
        if count == 0:
            active.zero_()
        return batch, active

    def write(self, query_a: Tensor, match_a: Tensor, weights: Tensor, config: ContinualConfig) -> None:
        self.step += 1
        active_weights = weights[: len(self.slots)].detach().cpu()
        for index, slot in enumerate(self.slots):
            weight = float(active_weights[index].item())
            slot.key_a = (
                (1.0 - config.key_write_rate * weight) * slot.key_a
                + config.key_write_rate * weight * query_a
            )
            slot.template_n = (
                (1.0 - config.template_write_rate * weight) * slot.template_n
                + config.template_write_rate * weight * match_a
            )
        if self.slots and active_weights.numel() > 0:
            winner = int(active_weights.argmax().item())
            if float(active_weights[winner].item()) > 0.01:
                self.slots[winner].last_used = self.step

    def allocate(self, query_a: Tensor, match_a: Tensor, value_id: int) -> None:
        if len(self.slots) >= self.capacity:
            victim = min(range(len(self.slots)), key=lambda index: self.slots[index].last_used)
            self.slots.pop(victim)
            self.evictions += 1
        self.slots.append(Slot(query_a.clone(), match_a.clone(), value_id, self.step))


class ContinualCategories:
    """Three ambiguous A groups, each with two old and one new N category."""

    def __init__(self, task: TaskConfig, seed: int, novel_flips: int = 2):
        if not 1 <= novel_flips <= task.n_dimension:
            raise ValueError("novel_flips must be between one and n_dimension")
        self.task = task
        self.generator = torch.Generator().manual_seed(seed + 120_000)
        self.python = random.Random(seed + 130_000)
        self.group_a = 0.15 + 0.70 * torch.rand(
            task.num_groups, task.a_dimension, generator=self.generator
        )
        base = 0.10 + 0.80 * torch.randint(
            0,
            2,
            (task.num_groups, task.n_dimension),
            generator=self.generator,
            dtype=torch.float32,
        )
        codes = torch.empty(task.num_slots, task.n_dimension)
        for group in range(task.num_groups):
            for member in range(task.group_size):
                label = group * task.group_size + member
                code = base[group].clone()
                if member > 0:
                    offset = (2 * group + 2 * (member - 1)) % task.n_dimension
                    flips = 2 if member < task.group_size - 1 else novel_flips
                    for index in range(flips):
                        coordinate = (offset + index) % task.n_dimension
                        code[coordinate] = 1.0 - code[coordinate]
                codes[label] = code
        self.codes_n = codes
        self.base_labels = tuple(
            group * task.group_size + member
            for group in range(task.num_groups)
            for member in range(task.group_size - 1)
        )
        self.novel_labels = tuple(
            group * task.group_size + task.group_size - 1
            for group in range(task.num_groups)
        )
        self.all_labels = tuple(range(task.num_slots))

    def sample(self, label: int, noise: float) -> Tuple[Tensor, Tensor]:
        group = label // self.task.group_size
        query = self.group_a[group] + noise * torch.randn(
            self.task.a_dimension, generator=self.generator
        )
        match = self.codes_n[label] + noise * torch.randn(
            self.task.n_dimension, generator=self.generator
        )
        return query.clamp(0.0, 1.0), match.clamp(0.0, 1.0)

    def shuffled_labels(self, labels: Sequence[int], count: int) -> List[int]:
        result = [labels[index % len(labels)] for index in range(count)]
        self.python.shuffle(result)
        return result

    def initial_memory(self) -> PersistentMemory:
        slots = [
            Slot(
                self.group_a[label // self.task.group_size].clone(),
                self.codes_n[label].clone(),
                label,
                0,
            )
            for label in self.base_labels
        ]
        return PersistentMemory(self.task, slots)

    def oracle_memory(self) -> PersistentMemory:
        slots = [
            Slot(
                self.group_a[label // self.task.group_size].clone(),
                self.codes_n[label].clone(),
                label,
                0,
            )
            for label in self.all_labels
        ]
        return PersistentMemory(self.task, slots)


def forward_memory(
    method: str,
    model: torch.nn.Module,
    batch: Batch,
    active: Tensor,
) -> Dict[str, Tensor]:
    """Run a trained model with an explicit active-slot mask."""

    if method == SOFT_CHEVRON:
        assert isinstance(model, SoftChevronAttention)
        q = model.q_a(batch.query_a).unsqueeze(1)
        keys = model.k_a(batch.keys_a)
        logits = (q * keys).sum(-1) / model.scale
        logits = logits.masked_fill(~active.unsqueeze(0), -1e4)
        alpha = logits.softmax(-1)
        alpha = alpha * active.unsqueeze(0)
        alpha = alpha / alpha.sum(-1, keepdim=True).clamp_min(1e-8)
        current = torch.sigmoid(model.match_a(batch.match_a)).unsqueeze(1)
        retained = torch.sigmoid(model.match_n(batch.templates_n))
        mismatch = (current - retained).abs().sum(-1) / (current + retained + 1e-6).sum(-1)
        theta = torch.sigmoid(model.theta_logit)
        sharpness = torch.nn.functional.softplus(model.k_raw)
        admission = torch.sigmoid(sharpness * (theta - mismatch))
        slot_mass = alpha * admission * active.unsqueeze(0)
        null_mass = (1.0 - slot_mass.sum(-1)).clamp(0.0, 1.0)
    else:
        assert isinstance(model, JointAttention)
        q_a = model.q_a(batch.query_a)
        k_a = model.k_a(batch.keys_a)
        q_n = model.q_n(batch.match_a - 0.5)
        k_n = model.k_n(batch.templates_n - 0.5)
        logits = (
            torch.einsum("bd,bsd->bs", q_a, k_a)
            + torch.einsum("bd,bsd->bs", q_n, k_n)
        ) / model.scale
        logits = logits.masked_fill(~active.unsqueeze(0), -1e4)
        null_logits = model.null_logit.expand(batch.query_a.size(0), 1)
        attention = torch.cat([logits, null_logits], -1).softmax(-1)
        slot_mass = attention[:, :-1] * active.unsqueeze(0)
        null_mass = attention[:, -1]

    values = model.output.values(batch.value_ids)
    answer_logits = model.output.answer(slot_mass, values, null_mass)
    return {
        "answer_logits": answer_logits,
        "slot_mass": slot_mass,
        "null_mass": null_mass,
    }


def novelty_and_write_mass(policy: str, outputs: Dict[str, Tensor]) -> Tuple[float, Tensor]:
    slot_mass = outputs["slot_mass"][0]
    if policy == JOINT_OUTPUT_WRITE:
        probabilities = outputs["answer_logits"].softmax(-1)
        novelty = float(probabilities[0, -1].item())
        admitted = 1.0 - novelty
        write_mass = admitted * slot_mass / slot_mass.sum().clamp_min(1e-8)
    else:
        novelty = float(outputs["null_mass"][0].item())
        write_mass = slot_mass
    return novelty, write_mass


@torch.no_grad()
def calibrate_threshold(
    policy: str,
    method: str,
    model: torch.nn.Module,
    task: CategoryMatchingTask,
    config: ContinualConfig,
    seed: int,
) -> Tuple[float, float]:
    generator = torch.Generator().manual_seed(seed + 140_000)
    device = next(model.parameters()).device
    scores: List[float] = []
    targets: List[bool] = []
    for _ in range(config.calibration_batches):
        batch = task.batch(config.calibration_batch_size, generator, device)
        outputs = model(batch)
        if policy == JOINT_OUTPUT_WRITE:
            values = outputs["answer_logits"].softmax(-1)[:, task.config.idk_class]
        else:
            values = outputs["null_mass"]
        scores.extend(float(value) for value in values.tolist())
        targets.extend(bool(value) for value in (~batch.matched).tolist())
    low, high = min(scores), max(scores)
    candidates = [low - 1e-6, high + 1e-6]
    candidates.extend(
        low + (high - low) * index / 400.0 for index in range(401)
    )
    best_threshold, best_balanced = candidates[0], -1.0
    for threshold in candidates:
        true_positive = sum(score >= threshold and target for score, target in zip(scores, targets))
        true_negative = sum(score < threshold and not target for score, target in zip(scores, targets))
        positives = sum(targets)
        negatives = len(targets) - positives
        balanced = 0.5 * (true_positive / positives + true_negative / negatives)
        if balanced > best_balanced:
            best_threshold, best_balanced = threshold, balanced
    return best_threshold, best_balanced


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


@torch.no_grad()
def probe_accuracy(
    policy: str,
    model: torch.nn.Module,
    memory: PersistentMemory,
    stream: ContinualCategories,
    labels: Sequence[int],
    noise: float,
    count_per_label: int,
) -> float:
    method = SOFT_CHEVRON if policy == CHEVRON_WRITE else JOINT_ATTENTION
    device = next(model.parameters()).device
    correct: List[float] = []
    for label in labels:
        for _ in range(count_per_label):
            query, match = stream.sample(label, noise)
            batch, active = memory.batch(query, match)
            outputs = forward_memory(
                method, model, batch.to(device), active.to(device)
            )
            prediction = int(outputs["answer_logits"].argmax(-1).item())
            correct.append(float(prediction == label))
    return mean(correct)


def memory_purity(memory: PersistentMemory, stream: ContinualCategories) -> Tuple[float, float]:
    key_errors: List[float] = []
    template_errors: List[float] = []
    for slot in memory.slots:
        group = slot.value_id // memory.task.group_size
        key_errors.append(float((slot.key_a - stream.group_a[group]).pow(2).mean().item()))
        template_errors.append(
            float((slot.template_n - stream.codes_n[slot.value_id]).pow(2).mean().item())
        )
    return mean(key_errors), mean(template_errors)


@torch.no_grad()
def run_policy(
    policy: str,
    model: torch.nn.Module,
    threshold: float,
    task: TaskConfig,
    continual: ContinualConfig,
    seed: int,
) -> Dict[str, float]:
    method = SOFT_CHEVRON if policy == CHEVRON_WRITE else JOINT_ATTENTION
    device = next(model.parameters()).device
    stream = ContinualCategories(task, seed, continual.novel_flips)
    memory = stream.initial_memory()
    memory.capacity = continual.memory_capacity
    phase_accuracy: Dict[str, List[float]] = {
        "base": [],
        "novel": [],
        "shift": [],
    }
    false_merges = 0
    novel_opportunities = 0
    false_splits = 0
    familiar_opportunities = 0
    cross_write_mass = 0.0
    write_steps = 0

    if continual.blocked_novel:
        base_count = continual.novel_learning // 2
        novel_count = continual.novel_learning - base_count
        novel_phase = stream.shuffled_labels(stream.base_labels, base_count)
        per_novel = novel_count // len(stream.novel_labels)
        for label in stream.novel_labels:
            novel_phase.extend([label] * per_novel)
        novel_phase.extend(
            stream.novel_labels[: novel_count - per_novel * len(stream.novel_labels)]
        )
    else:
        novel_phase = stream.shuffled_labels(
            stream.all_labels, continual.novel_learning
        )

    phases = (
        (
            "base",
            stream.shuffled_labels(stream.base_labels, continual.base_adaptation),
            continual.observation_noise,
        ),
        (
            "novel",
            novel_phase,
            continual.observation_noise,
        ),
        (
            "shift",
            stream.shuffled_labels(stream.all_labels, continual.shifted_learning),
            continual.shifted_noise,
        ),
    )

    for phase, labels, noise in phases:
        for label in labels:
            query, match = stream.sample(label, noise)
            batch, active = memory.batch(query, match)
            outputs = forward_memory(
                method, model, batch.to(device), active.to(device)
            )
            prediction = int(outputs["answer_logits"].argmax(-1).item())
            phase_accuracy[phase].append(float(prediction == label))
            novelty, write_mass = novelty_and_write_mass(policy, outputs)
            existing = label in memory.labels
            allocate = novelty >= threshold
            if existing:
                familiar_opportunities += 1
                false_splits += int(allocate)
            else:
                novel_opportunities += 1
                false_merges += int(not allocate)

            labels_before = memory.labels
            cross_write_mass += sum(
                float(write_mass[index].item())
                for index, stored_label in enumerate(labels_before)
                if stored_label != label
            )
            write_steps += 1
            memory.write(query, match, write_mass, continual)
            if allocate:
                memory.allocate(query, match, label)

    key_error, template_error = memory_purity(memory, stream)
    final_old = probe_accuracy(
        policy,
        model,
        memory,
        stream,
        stream.base_labels,
        continual.observation_noise,
        continual.final_per_category,
    )
    final_new = probe_accuracy(
        policy,
        model,
        memory,
        stream,
        stream.novel_labels,
        continual.observation_noise,
        continual.final_per_category,
    )
    final_shifted = probe_accuracy(
        policy,
        model,
        memory,
        stream,
        stream.all_labels,
        continual.shifted_noise,
        continual.final_per_category,
    )
    oracle = stream.oracle_memory()
    oracle_clean = probe_accuracy(
        policy,
        model,
        oracle,
        stream,
        stream.all_labels,
        continual.observation_noise,
        continual.final_per_category,
    )
    oracle_shifted = probe_accuracy(
        policy,
        model,
        oracle,
        stream,
        stream.all_labels,
        continual.shifted_noise,
        continual.final_per_category,
    )
    retained = len(set(memory.labels).intersection(stream.all_labels))
    return {
        "base_online_accuracy": mean(phase_accuracy["base"]),
        "novel_online_accuracy": mean(phase_accuracy["novel"]),
        "shift_online_accuracy": mean(phase_accuracy["shift"]),
        "final_old_accuracy": final_old,
        "final_new_accuracy": final_new,
        "final_clean_accuracy": (
            len(stream.base_labels) * final_old + len(stream.novel_labels) * final_new
        ) / len(stream.all_labels),
        "final_shifted_accuracy": final_shifted,
        "oracle_clean_accuracy": oracle_clean,
        "oracle_shifted_accuracy": oracle_shifted,
        "false_merge_rate": false_merges / max(novel_opportunities, 1),
        "false_split_rate": false_splits / max(familiar_opportunities, 1),
        "cross_write_mass": cross_write_mass / max(write_steps, 1),
        "key_mse": key_error,
        "template_mse": template_error,
        "categories_retained": float(retained),
        "evictions": float(memory.evictions),
        "threshold": threshold,
    }


def summarize(rows: Sequence[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    return {
        key: (
            statistics.mean(row[key] for row in rows),
            statistics.stdev(row[key] for row in rows) if len(rows) > 1 else 0.0,
        )
        for key in rows[0]
    }


def run_experiment(
    seeds: Sequence[int],
    steps: int,
    device: torch.device,
    continual: ContinualConfig,
) -> Dict[str, List[Dict[str, float]]]:
    task_config = TaskConfig()
    episodic_task = CategoryMatchingTask(task_config)
    train_config = replace(
        TrainConfig(steps=steps),
        retrieval_weight=0.0,
        gate_weight=0.0,
        match_init="independent_random",
    )
    rows: Dict[str, List[Dict[str, float]]] = {policy: [] for policy in POLICIES}
    for seed in seeds:
        chevron = train_model(SOFT_CHEVRON, episodic_task, train_config, seed, device)
        joint = train_model(JOINT_ATTENTION, episodic_task, train_config, seed, device)
        models = {
            CHEVRON_WRITE: (SOFT_CHEVRON, chevron),
            JOINT_NULL_WRITE: (JOINT_ATTENTION, joint),
            JOINT_OUTPUT_WRITE: (JOINT_ATTENTION, joint),
        }
        for policy, (method, model) in models.items():
            threshold, calibration = calibrate_threshold(
                policy, method, model, episodic_task, continual, seed
            )
            result = run_policy(
                policy, model, threshold, task_config, continual, seed
            )
            result["calibration_balanced_accuracy"] = calibration
            result["parameters"] = float(parameter_count(model))
            rows[policy].append(result)
    return rows


def print_results(rows: Dict[str, List[Dict[str, float]]]) -> None:
    for policy in POLICIES:
        print("\n" + policy)
        for key, (average, spread) in summarize(rows[policy]).items():
            print("  %-31s %.4f +/- %.4f" % (key, average, spread))
    for baseline in (JOINT_NULL_WRITE, JOINT_OUTPUT_WRITE):
        print("\npaired chevron - %s" % baseline)
        for metric in ("final_clean_accuracy", "final_shifted_accuracy"):
            differences = [
                chevron[metric] - joint[metric]
                for chevron, joint in zip(rows[CHEVRON_WRITE], rows[baseline])
            ]
            average = statistics.mean(differences)
            spread = statistics.stdev(differences) if len(differences) > 1 else 0.0
            wins = sum(difference > 0.0 for difference in differences)
            ties = sum(difference == 0.0 for difference in differences)
            print(
                "  %-31s %+.4f +/- %.4f wins=%d ties=%d"
                % (metric, average, spread, wins, ties)
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7, 17, 27, 37, 47, 57, 67, 77, 87, 97],
    )
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = run_experiment(
        args.seeds,
        args.steps,
        torch.device(args.device),
        ContinualConfig(),
    )
    print_results(rows)


if __name__ == "__main__":
    main()
