"""Phase 7.6: learned write baseline, causal ablations, and robustness matrix."""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from phase7_soft_chevron.continual_memory import (
    ContinualCategories,
    ContinualConfig,
    PersistentMemory,
    forward_memory,
    mean,
    memory_purity,
)
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


CHEVRON_LEARNED = "chevron_learned"
CHEVRON_ALPHA = "chevron_alpha_only"
CHEVRON_FIXED = "chevron_fixed_gate"
CHEVRON_MISALIGNED = "chevron_misaligned_gate"
CHEVRON_ORACLE = "chevron_oracle_write"
JOINT_CONTROLLER = "joint_learned_controller"
ABLATIONS = (
    CHEVRON_LEARNED,
    CHEVRON_ALPHA,
    CHEVRON_FIXED,
    CHEVRON_MISALIGNED,
    CHEVRON_ORACLE,
    JOINT_CONTROLLER,
)
PRIMARY = (CHEVRON_LEARNED, JOINT_CONTROLLER)


@dataclass(frozen=True)
class ClosureConfig:
    controller_steps: int = 350
    controller_batch_size: int = 128
    controller_learning_rate: float = 3e-3
    calibration_batches: int = 12
    calibration_batch_size: int = 128
    minimum_active: int = 6


@dataclass
class Prepared:
    soft: SoftChevronAttention
    joint: JointAttention
    controller: "JointWriteController"
    thresholds: Dict[str, float]
    calibration: Dict[str, float]


class JointWriteController(nn.Module):
    """Parameter-matched veto head over permutation-invariant attention statistics."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(9, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.network(features).squeeze(-1)


def active_matrix(active: Tensor, batch_size: int) -> Tensor:
    if active.dim() == 1:
        return active.unsqueeze(0).expand(batch_size, -1)
    return active


def random_active_mask(
    batch: Batch,
    minimum_active: int,
    generator: torch.Generator,
) -> Tensor:
    batch_size, slots = batch.keys_a.shape[:2]
    counts = torch.randint(
        minimum_active, slots + 1, (batch_size,), generator=generator
    )
    order = torch.rand(batch_size, slots, generator=generator).argsort(-1)
    ranks = order.argsort(-1)
    active = ranks < counts.unsqueeze(1)
    for row in range(batch_size):
        if bool(batch.matched[row]) and not bool(active[row, batch.target_slots[row]]):
            victim = int(active[row].nonzero()[0].item())
            active[row, victim] = False
            active[row, batch.target_slots[row]] = True
    return active


def soft_components(
    model: SoftChevronAttention,
    batch: Batch,
    active: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    mask = active_matrix(active, batch.query_a.size(0)).to(batch.query_a.device)
    query = model.q_a(batch.query_a).unsqueeze(1)
    keys = model.k_a(batch.keys_a)
    logits = (query * keys).sum(-1) / model.scale
    logits = logits.masked_fill(~mask, -1e4)
    alpha = logits.softmax(-1) * mask
    alpha = alpha / alpha.sum(-1, keepdim=True).clamp_min(1e-8)

    current = torch.sigmoid(model.match_a(batch.match_a)).unsqueeze(1)
    retained = torch.sigmoid(model.match_n(batch.templates_n))
    mismatch = (current - retained).abs().sum(-1) / (
        current + retained + 1e-6
    ).sum(-1)
    theta = torch.sigmoid(model.theta_logit)
    sharpness = F.softplus(model.k_raw)
    learned = torch.sigmoid(sharpness * (theta - mismatch)) * mask

    raw_current = batch.match_a.unsqueeze(1)
    raw_mismatch = (raw_current - batch.templates_n).abs().sum(-1) / (
        raw_current + batch.templates_n + 1e-6
    ).sum(-1)
    fixed = torch.sigmoid(30.0 * (0.10 - raw_mismatch)) * mask
    return alpha, learned, fixed


def misalign(values: Tensor, active: Tensor) -> Tensor:
    mask = active_matrix(active, values.size(0)).to(values.device)
    result = values.clone()
    for row in range(values.size(0)):
        indices = mask[row].nonzero().flatten()
        if indices.numel() > 1:
            result[row, indices] = torch.roll(values[row, indices], shifts=1)
    return result


def joint_features(
    model: JointAttention,
    batch: Batch,
    active: Tensor,
) -> Tensor:
    mask = active_matrix(active, batch.query_a.size(0)).to(batch.query_a.device)
    q_a = model.q_a(batch.query_a)
    k_a = model.k_a(batch.keys_a)
    q_n = model.q_n(batch.match_a - 0.5)
    k_n = model.k_n(batch.templates_n - 0.5)
    a_logits = torch.einsum("bd,bsd->bs", q_a, k_a) / model.scale
    n_logits = torch.einsum("bd,bsd->bs", q_n, k_n) / model.scale
    joint = a_logits + n_logits
    masked_joint = joint.masked_fill(~mask, -1e4)
    count = mask.sum(-1).clamp_min(1)
    joint_mean = (joint * mask).sum(-1) / count
    joint_variance = ((joint - joint_mean.unsqueeze(1)).pow(2) * mask).sum(-1) / count
    top = masked_joint.topk(k=2, dim=-1).values
    weights = masked_joint.softmax(-1) * mask
    entropy = -(weights * weights.clamp_min(1e-8).log()).sum(-1)
    entropy = entropy / count.float().log().clamp_min(1.0)
    log_mean_exp = torch.logsumexp(masked_joint, -1) - count.float().log()
    return torch.stack(
        [
            a_logits.masked_fill(~mask, -1e4).max(-1).values,
            n_logits.masked_fill(~mask, -1e4).max(-1).values,
            masked_joint.max(-1).values,
            joint_mean,
            joint_variance.sqrt(),
            top[:, 0] - top[:, 1],
            log_mean_exp,
            entropy,
            count.float() / mask.size(1),
        ],
        -1,
    )


def best_threshold(scores: Sequence[float], targets: Sequence[bool]) -> Tuple[float, float]:
    low, high = min(scores), max(scores)
    candidates = [low - 1e-6, high + 1e-6]
    candidates.extend(low + (high - low) * index / 400.0 for index in range(401))
    positives = sum(targets)
    negatives = len(targets) - positives
    best_value, best_score = candidates[0], -1.0
    for threshold in candidates:
        true_positive = sum(score >= threshold and target for score, target in zip(scores, targets))
        true_negative = sum(score < threshold and not target for score, target in zip(scores, targets))
        balanced = 0.5 * (true_positive / positives + true_negative / negatives)
        if balanced > best_score:
            best_value, best_score = threshold, balanced
    return best_value, best_score


def train_controller(
    model: JointAttention,
    task: CategoryMatchingTask,
    config: ClosureConfig,
    seed: int,
    device: torch.device,
) -> JointWriteController:
    torch.manual_seed(seed + 150_000)
    generator = torch.Generator().manual_seed(seed + 151_000)
    controller = JointWriteController().to(device)
    optimizer = torch.optim.AdamW(
        controller.parameters(), lr=config.controller_learning_rate
    )
    model.eval()
    controller.train()
    for _ in range(config.controller_steps):
        cpu_batch = task.batch(config.controller_batch_size, generator)
        active = random_active_mask(cpu_batch, config.minimum_active, generator)
        batch = cpu_batch.to(device)
        with torch.no_grad():
            features = joint_features(model, batch, active.to(device))
        target = (~batch.matched).float()
        loss = F.binary_cross_entropy_with_logits(controller(features), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    controller.eval()
    return controller


@torch.no_grad()
def calibrate_variant(
    variant: str,
    prepared: Prepared,
    task: CategoryMatchingTask,
    config: ClosureConfig,
    seed: int,
) -> Tuple[float, float]:
    generator = torch.Generator().manual_seed(seed + 152_000)
    device = next(prepared.soft.parameters()).device
    scores: List[float] = []
    targets: List[bool] = []
    for _ in range(config.calibration_batches):
        cpu_batch = task.batch(config.calibration_batch_size, generator)
        active = random_active_mask(cpu_batch, config.minimum_active, generator)
        batch = cpu_batch.to(device)
        active_device = active.to(device)
        if variant == JOINT_CONTROLLER:
            values = prepared.controller(
                joint_features(prepared.joint, batch, active_device)
            ).sigmoid()
        else:
            alpha, learned, fixed = soft_components(
                prepared.soft, batch, active_device
            )
            if variant in (CHEVRON_LEARNED, CHEVRON_ALPHA):
                admitted = alpha * learned
            elif variant == CHEVRON_FIXED:
                admitted = alpha * fixed
            elif variant == CHEVRON_MISALIGNED:
                admitted = alpha * misalign(learned, active_device)
            else:
                raise ValueError("cannot calibrate variant: %s" % variant)
            values = 1.0 - admitted.sum(-1)
        scores.extend(float(value) for value in values.cpu().tolist())
        targets.extend(bool(value) for value in (~cpu_batch.matched).tolist())
    return best_threshold(scores, targets)


def phase_schedule(
    stream: ContinualCategories,
    config: ContinualConfig,
) -> Tuple[Tuple[str, List[int], float], ...]:
    if config.blocked_novel:
        base_count = config.novel_learning // 2
        novel_count = config.novel_learning - base_count
        novel_phase = stream.shuffled_labels(stream.base_labels, base_count)
        per_novel = novel_count // len(stream.novel_labels)
        for label in stream.novel_labels:
            novel_phase.extend([label] * per_novel)
        novel_phase.extend(
            stream.novel_labels[: novel_count - per_novel * len(stream.novel_labels)]
        )
    else:
        novel_phase = stream.shuffled_labels(stream.all_labels, config.novel_learning)
    return (
        (
            "base",
            stream.shuffled_labels(stream.base_labels, config.base_adaptation),
            config.observation_noise,
        ),
        ("novel", novel_phase, config.observation_noise),
        (
            "shift",
            stream.shuffled_labels(stream.all_labels, config.shifted_learning),
            config.shifted_noise,
        ),
    )


def decision(
    variant: str,
    prepared: Prepared,
    batch: Batch,
    active: Tensor,
    outputs: Dict[str, Tensor],
) -> Tuple[float, Tensor]:
    if variant == JOINT_CONTROLLER:
        novelty = float(
            prepared.controller(joint_features(prepared.joint, batch, active)).sigmoid().item()
        )
        slot_mass = outputs["slot_mass"][0]
        write = (1.0 - novelty) * slot_mass / slot_mass.sum().clamp_min(1e-8)
        return novelty, write
    alpha, learned, fixed = soft_components(prepared.soft, batch, active)
    if variant == CHEVRON_ALPHA:
        write = alpha[0]
        novelty = float((1.0 - (alpha * learned).sum(-1)).item())
    elif variant == CHEVRON_FIXED:
        write = (alpha * fixed)[0]
        novelty = float((1.0 - write.sum()).item())
    elif variant == CHEVRON_MISALIGNED:
        write = (alpha * misalign(learned, active))[0]
        novelty = float((1.0 - write.sum()).item())
    else:
        write = (alpha * learned)[0]
        novelty = float((1.0 - write.sum()).item())
    return novelty, write


@torch.no_grad()
def probe(
    method: str,
    model: nn.Module,
    memory: PersistentMemory,
    stream: ContinualCategories,
    labels: Sequence[int],
    noise: float,
    count_per_label: int,
) -> float:
    device = next(model.parameters()).device
    correct: List[float] = []
    for label in labels:
        for _ in range(count_per_label):
            query, match = stream.sample(label, noise)
            batch, active = memory.batch(query, match)
            outputs = forward_memory(method, model, batch.to(device), active.to(device))
            prediction = int(outputs["answer_logits"].argmax(-1).item())
            correct.append(float(prediction == label))
    return mean(correct)


@torch.no_grad()
def run_variant(
    variant: str,
    prepared: Prepared,
    task: TaskConfig,
    config: ContinualConfig,
    seed: int,
) -> Dict[str, float]:
    joint = variant == JOINT_CONTROLLER
    method = JOINT_ATTENTION if joint else SOFT_CHEVRON
    model = prepared.joint if joint else prepared.soft
    device = next(model.parameters()).device
    stream = ContinualCategories(task, seed, config.novel_flips)
    memory = stream.initial_memory()
    memory.capacity = config.memory_capacity
    threshold = prepared.thresholds.get(variant, 0.5)
    accuracy: Dict[str, List[float]] = {"base": [], "novel": [], "shift": []}
    false_merges = false_splits = novel_cases = familiar_cases = 0
    cross_mass = 0.0
    steps = 0

    for phase, labels, noise in phase_schedule(stream, config):
        for label in labels:
            query, match = stream.sample(label, noise)
            cpu_batch, cpu_active = memory.batch(query, match)
            batch, active = cpu_batch.to(device), cpu_active.to(device)
            outputs = forward_memory(method, model, batch, active)
            prediction = int(outputs["answer_logits"].argmax(-1).item())
            accuracy[phase].append(float(prediction == label))
            existing = label in memory.labels

            if variant == CHEVRON_ORACLE:
                allocate = not existing
                write_mass = torch.zeros(task.num_slots, device=device)
                matching = [
                    index for index, stored in enumerate(memory.labels) if stored == label
                ]
                for index in matching:
                    write_mass[index] = 1.0 / len(matching)
            else:
                novelty, write_mass = decision(
                    variant, prepared, batch, active, outputs
                )
                allocate = novelty >= threshold

            if existing:
                familiar_cases += 1
                false_splits += int(allocate)
            else:
                novel_cases += 1
                false_merges += int(not allocate)
            cross_mass += sum(
                float(write_mass[index].item())
                for index, stored in enumerate(memory.labels)
                if stored != label
            )
            steps += 1
            memory.write(query, match, write_mass, config)
            if allocate:
                memory.allocate(query, match, label)

    key_mse, template_mse = memory_purity(memory, stream)
    final_old = probe(
        method,
        model,
        memory,
        stream,
        stream.base_labels,
        config.observation_noise,
        config.final_per_category,
    )
    final_new = probe(
        method,
        model,
        memory,
        stream,
        stream.novel_labels,
        config.observation_noise,
        config.final_per_category,
    )
    final_shifted = probe(
        method,
        model,
        memory,
        stream,
        stream.all_labels,
        config.shifted_noise,
        config.final_per_category,
    )
    return {
        "base_online_accuracy": mean(accuracy["base"]),
        "novel_online_accuracy": mean(accuracy["novel"]),
        "shift_online_accuracy": mean(accuracy["shift"]),
        "final_old_accuracy": final_old,
        "final_new_accuracy": final_new,
        "final_clean_accuracy": (
            len(stream.base_labels) * final_old + len(stream.novel_labels) * final_new
        ) / len(stream.all_labels),
        "final_shifted_accuracy": final_shifted,
        "false_merge_rate": false_merges / max(novel_cases, 1),
        "false_split_rate": false_splits / max(familiar_cases, 1),
        "cross_write_mass": cross_mass / max(steps, 1),
        "key_mse": key_mse,
        "template_mse": template_mse,
        "categories_retained": float(len(set(memory.labels))),
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


def prepare_models(
    seeds: Sequence[int],
    task: CategoryMatchingTask,
    steps: int,
    closure: ClosureConfig,
    device: torch.device,
) -> Dict[int, Prepared]:
    prepared: Dict[int, Prepared] = {}
    soft_config = replace(
        TrainConfig(steps=steps, d_model=40),
        retrieval_weight=0.0,
        gate_weight=0.0,
        match_init="independent_random",
    )
    joint_config = replace(
        TrainConfig(steps=steps, d_model=35),
        retrieval_weight=0.0,
        gate_weight=0.0,
    )
    for seed in seeds:
        soft = train_model(SOFT_CHEVRON, task, soft_config, seed, device)
        joint = train_model(JOINT_ATTENTION, task, joint_config, seed, device)
        controller = train_controller(joint, task, closure, seed, device)
        item = Prepared(soft, joint, controller, {}, {})
        for variant in (
            CHEVRON_LEARNED,
            CHEVRON_ALPHA,
            CHEVRON_FIXED,
            CHEVRON_MISALIGNED,
            JOINT_CONTROLLER,
        ):
            threshold, score = calibrate_variant(
                variant, item, task, closure, seed
            )
            item.thresholds[variant] = threshold
            item.calibration[variant] = score
        item.thresholds[CHEVRON_ORACLE] = 0.5
        item.calibration[CHEVRON_ORACLE] = 1.0
        prepared[seed] = item
    return prepared


MATRIX: Dict[str, ContinualConfig] = {
    "default": ContinualConfig(),
    "shift_noise06": replace(ContinualConfig(), shifted_noise=0.060),
    "shift_noise10": replace(ContinualConfig(), shifted_noise=0.100),
    "near_1flip": replace(ContinualConfig(), novel_flips=1),
    "far_3flip": replace(ContinualConfig(), novel_flips=3),
    "capacity8": replace(ContinualConfig(), memory_capacity=8),
    "blocked_novel": replace(ContinualConfig(), blocked_novel=True),
    "long_stream": replace(ContinualConfig(), shifted_learning=540),
}


def run_ablation(
    prepared: Dict[int, Prepared], seeds: Sequence[int], task: TaskConfig
) -> Dict[str, List[Dict[str, float]]]:
    rows = {variant: [] for variant in ABLATIONS}
    for seed in seeds:
        for variant in ABLATIONS:
            result = run_variant(
                variant, prepared[seed], task, ContinualConfig(), seed
            )
            result["calibration_balanced_accuracy"] = prepared[seed].calibration[variant]
            if variant == JOINT_CONTROLLER:
                parameters = parameter_count(prepared[seed].joint) + parameter_count(
                    prepared[seed].controller
                )
            else:
                parameters = parameter_count(prepared[seed].soft)
            result["parameters"] = float(parameters)
            rows[variant].append(result)
    return rows


def run_matrix(
    prepared: Dict[int, Prepared], seeds: Sequence[int], task: TaskConfig
) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    rows: Dict[str, Dict[str, List[Dict[str, float]]]] = {}
    for condition, config in MATRIX.items():
        rows[condition] = {variant: [] for variant in PRIMARY}
        for seed in seeds:
            for variant in PRIMARY:
                rows[condition][variant].append(
                    run_variant(variant, prepared[seed], task, config, seed)
                )
    return rows


def print_ablation(rows: Dict[str, List[Dict[str, float]]]) -> None:
    print("\nABLATIONS")
    for variant in ABLATIONS:
        summary = summarize(rows[variant])
        print("\n" + variant)
        for key in (
            "final_clean_accuracy",
            "final_shifted_accuracy",
            "false_merge_rate",
            "false_split_rate",
            "cross_write_mass",
            "template_mse",
            "categories_retained",
            "evictions",
            "calibration_balanced_accuracy",
            "parameters",
        ):
            average, spread = summary[key]
            print("  %-31s %.4f +/- %.4f" % (key, average, spread))
    comparisons = (
        (CHEVRON_LEARNED, CHEVRON_ALPHA),
        (CHEVRON_LEARNED, CHEVRON_FIXED),
        (CHEVRON_LEARNED, CHEVRON_MISALIGNED),
        (CHEVRON_LEARNED, JOINT_CONTROLLER),
    )
    for variant, baseline_name in comparisons:
        differences = [
            chevron["final_clean_accuracy"] - joint["final_clean_accuracy"]
            for chevron, joint in zip(rows[variant], rows[baseline_name])
        ]
        print(
            "\npaired %-30s %+0.4f +/- %.4f wins=%d ties=%d"
            % (
                variant + " - " + baseline_name,
                statistics.mean(differences),
                statistics.stdev(differences) if len(differences) > 1 else 0.0,
                sum(value > 0.0 for value in differences),
                sum(value == 0.0 for value in differences),
            )
        )


def print_matrix(rows: Dict[str, Dict[str, List[Dict[str, float]]]]) -> None:
    print("\nROBUSTNESS MATRIX")
    for condition in MATRIX:
        chevron = summarize(rows[condition][CHEVRON_LEARNED])
        joint = summarize(rows[condition][JOINT_CONTROLLER])
        differences = [
            left["final_clean_accuracy"] - right["final_clean_accuracy"]
            for left, right in zip(
                rows[condition][CHEVRON_LEARNED], rows[condition][JOINT_CONTROLLER]
            )
        ]
        print(
            "%-16s chevron=%.4f+/-%.4f joint=%.4f+/-%.4f delta=%+.4f wins=%d"
            % (
                condition,
                chevron["final_clean_accuracy"][0],
                chevron["final_clean_accuracy"][1],
                joint["final_clean_accuracy"][0],
                joint["final_clean_accuracy"][1],
                statistics.mean(differences),
                sum(value > 0.0 for value in differences),
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[7, 17, 27, 37, 47, 57, 67, 77, 87, 97],
    )
    parser.add_argument("--matrix-seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--controller-steps", type=int, default=350)
    parser.add_argument("--section", choices=("all", "ablation", "matrix"), default="all")
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_config = TaskConfig()
    task = CategoryMatchingTask(task_config)
    closure = ClosureConfig(controller_steps=args.controller_steps)
    prepared = prepare_models(
        args.seeds, task, args.steps, closure, torch.device(args.device)
    )
    if args.section in ("all", "ablation"):
        print_ablation(run_ablation(prepared, args.seeds, task_config))
    if args.section in ("all", "matrix"):
        matrix_seeds = args.seeds[: args.matrix_seeds]
        print_matrix(run_matrix(prepared, matrix_seeds, task_config))


if __name__ == "__main__":
    main()
