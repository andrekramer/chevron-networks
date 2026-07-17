"""Phase 7: learned soft Chevron Attention with normalized mismatch.

Fresh episodic memories contain several bottom-up-plausible category members.
Q_A/K_A must retrieve the relevant group; an A/N mismatch gate must admit the
one retained template that matches current evidence, or route all remaining
mass to a learned null value when no template matches.
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


SOFT_CHEVRON = "soft_chevron"
JOINT_ATTENTION = "joint_attention"
A_ONLY_ATTENTION = "a_only_attention"
METHODS = (SOFT_CHEVRON, JOINT_ATTENTION, A_ONLY_ATTENTION)


@dataclass(frozen=True)
class TaskConfig:
    a_dimension: int = 12
    n_dimension: int = 8
    num_groups: int = 3
    group_size: int = 3
    num_values: int = 12
    matched_probability: float = 0.75
    key_noise: float = 0.045
    query_noise: float = 0.035
    template_noise: float = 0.015
    match_noise: float = 0.0

    @property
    def num_slots(self) -> int:
        return self.num_groups * self.group_size

    @property
    def idk_class(self) -> int:
        return self.num_values


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 700
    batch_size: int = 128
    d_model: int = 40
    learning_rate: float = 3e-3
    retrieval_weight: float = 0.5
    gate_weight: float = 0.5
    grad_clip: float = 1.0
    theta_init: float = 0.10
    sharpness_init: float = 30.0
    match_init: str = "identity"


@dataclass
class Batch:
    query_a: Tensor
    match_a: Tensor
    keys_a: Tensor
    templates_n: Tensor
    value_ids: Tensor
    target_groups: Tensor
    target_slots: Tensor
    matched: Tensor
    answers: Tensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(**{name: value.to(device) for name, value in vars(self).items()})


class CategoryMatchingTask:
    """Fresh groups with ambiguous A addresses and discriminating N templates."""

    def __init__(self, config: TaskConfig):
        if config.num_values < config.num_slots:
            raise ValueError("num_values must be at least num_slots")
        if config.n_dimension < 3:
            raise ValueError("n_dimension must be at least three")
        self.config = config

    def batch(
        self,
        batch_size: int,
        generator: torch.Generator,
        device: Optional[torch.device] = None,
    ) -> Batch:
        c = self.config
        row = torch.arange(batch_size)
        prototypes = 0.15 + 0.70 * torch.rand(
            batch_size, c.num_groups, c.a_dimension, generator=generator
        )
        keys = prototypes.unsqueeze(2).expand(-1, -1, c.group_size, -1).clone()
        keys += c.key_noise * torch.randn(keys.shape, generator=generator)
        keys = keys.clamp(0.0, 1.0).reshape(batch_size, c.num_slots, c.a_dimension)

        target_groups = torch.randint(
            c.num_groups, (batch_size,), generator=generator
        )
        target_within = torch.randint(
            c.group_size, (batch_size,), generator=generator
        )
        target_slots = target_groups * c.group_size + target_within
        query_a = prototypes[row, target_groups]
        query_a = (
            query_a
            + c.query_noise
            * torch.randn(batch_size, c.a_dimension, generator=generator)
        ).clamp(0.0, 1.0)

        match_base = 0.10 + 0.80 * torch.randint(
            0,
            2,
            (batch_size, c.n_dimension),
            generator=generator,
            dtype=torch.float32,
        )
        match_a = (
            match_base
            + c.match_noise
            * torch.randn(batch_size, c.n_dimension, generator=generator)
        ).clamp(0.0, 1.0)
        templates = match_base.unsqueeze(1).expand(-1, c.num_slots, -1).clone()
        # Every non-target template differs in two deterministic coordinates.
        # Slot-dependent flips prevent a single feature from becoming a global
        # shortcut for mismatch.
        for slot in range(c.num_slots):
            first = slot % c.n_dimension
            second = (3 * slot + 1) % c.n_dimension
            templates[:, slot, first] = 1.0 - templates[:, slot, first]
            templates[:, slot, second] = 1.0 - templates[:, slot, second]

        matched = torch.rand(batch_size, generator=generator) < c.matched_probability
        matched_rows = row[matched]
        matched_slots = target_slots[matched]
        templates[matched_rows, matched_slots] = match_base[matched]
        templates += c.template_noise * torch.randn(templates.shape, generator=generator)
        templates = templates.clamp(0.0, 1.0)

        value_ids = torch.rand(batch_size, c.num_values, generator=generator).argsort(1)
        value_ids = value_ids[:, : c.num_slots]
        answers = torch.full((batch_size,), c.idk_class, dtype=torch.long)
        answers[matched] = value_ids[matched_rows, matched_slots]

        batch = Batch(
            query_a=query_a,
            match_a=match_a,
            keys_a=keys,
            templates_n=templates,
            value_ids=value_ids,
            target_groups=target_groups,
            target_slots=target_slots,
            matched=matched,
            answers=answers,
        )
        return batch.to(device) if device is not None else batch


def copied_pair(input_dimension: int, output_dimension: int) -> Tuple[nn.Linear, nn.Linear]:
    query = nn.Linear(input_dimension, output_dimension, bias=False)
    key = nn.Linear(input_dimension, output_dimension, bias=False)
    with torch.no_grad():
        key.weight.copy_(query.weight)
    return query, key


def inverse_softplus(value: float) -> float:
    return math.log(math.exp(value) - 1.0)


class ValueOutput(nn.Module):
    def __init__(self, task: TaskConfig, d_model: int):
        super().__init__()
        self.n_values = nn.Embedding(task.num_values, d_model)
        self.v_n = nn.Linear(d_model, d_model, bias=False)
        self.v_null = nn.Parameter(torch.zeros(d_model))
        self.answer_head = nn.Linear(d_model, task.num_values + 1)

    def values(self, value_ids: Tensor) -> Tensor:
        return self.v_n(self.n_values(value_ids))

    def answer(self, slot_mass: Tensor, values: Tensor, null_mass: Tensor) -> Tensor:
        output = (slot_mass.unsqueeze(-1) * values).sum(1)
        output = output + null_mass.unsqueeze(-1) * self.v_null
        return self.answer_head(output)


class SoftChevronAttention(nn.Module):
    """alpha=softmax(Q_A K_A^T), then soft A/N admission of V_N."""

    def __init__(
        self,
        task: TaskConfig,
        d_model: int,
        theta_init: float = 0.10,
        sharpness_init: float = 30.0,
        match_init: str = "identity",
    ):
        super().__init__()
        if not 0.0 < theta_init < 1.0:
            raise ValueError("theta_init must be between zero and one")
        if sharpness_init <= 0.0:
            raise ValueError("sharpness_init must be positive")
        if match_init not in ("identity", "shared_random", "independent_random"):
            raise ValueError("unknown match_init: %s" % match_init)
        self.task = task
        self.scale = math.sqrt(d_model)
        self.q_a, self.k_a = copied_pair(task.a_dimension, d_model)
        self.match_a = nn.Linear(task.n_dimension, task.n_dimension)
        self.match_n = nn.Linear(task.n_dimension, task.n_dimension)
        with torch.no_grad():
            if match_init == "identity":
                self.match_a.weight.zero_()
                self.match_n.weight.zero_()
                self.match_a.weight.add_(4.0 * torch.eye(task.n_dimension))
                self.match_n.weight.add_(4.0 * torch.eye(task.n_dimension))
                self.match_a.bias.fill_(-2.0)
                self.match_n.bias.fill_(-2.0)
            elif match_init == "shared_random":
                self.match_n.weight.copy_(self.match_a.weight)
                self.match_n.bias.copy_(self.match_a.bias)
        self.theta_logit = nn.Parameter(
            torch.tensor(math.log(theta_init / (1.0 - theta_init)))
        )
        self.k_raw = nn.Parameter(torch.tensor(inverse_softplus(sharpness_init)))
        self.output = ValueOutput(task, d_model)

    def forward(self, batch: Batch) -> Dict[str, Tensor]:
        q = self.q_a(batch.query_a).unsqueeze(1)
        keys = self.k_a(batch.keys_a)
        retrieval_logits = (q * keys).sum(-1) / self.scale
        alpha = retrieval_logits.softmax(-1)

        current_a = torch.sigmoid(self.match_a(batch.match_a)).unsqueeze(1)
        retained_n = torch.sigmoid(self.match_n(batch.templates_n))
        engagement = current_a + retained_n + 1e-6
        mismatch = (current_a - retained_n).abs().sum(-1) / engagement.sum(-1)
        theta = torch.sigmoid(self.theta_logit)
        sharpness = F.softplus(self.k_raw)
        r = torch.sigmoid(sharpness * (theta - mismatch))

        admitted = alpha * r
        admitted_mass = admitted.sum(-1)
        remaining_mass = (1.0 - admitted_mass).clamp(0.0, 1.0)
        values = self.output.values(batch.value_ids)
        answer_logits = self.output.answer(admitted, values, remaining_mass)
        return {
            "answer_logits": answer_logits,
            "retrieval_logits": retrieval_logits,
            "alpha": alpha,
            "r": r,
            "mismatch": mismatch,
            "slot_mass": admitted,
            "null_mass": remaining_mass,
            "theta": theta,
            "sharpness": sharpness,
        }


class AOnlyAttention(nn.Module):
    """Standard attention over A addresses with a learned null slot."""

    def __init__(self, task: TaskConfig, d_model: int):
        super().__init__()
        self.task = task
        self.scale = math.sqrt(d_model)
        self.q_a, self.k_a = copied_pair(task.a_dimension, d_model)
        self.null_logit = nn.Parameter(torch.tensor(0.0))
        self.output = ValueOutput(task, d_model)

    def forward(self, batch: Batch) -> Dict[str, Tensor]:
        q = self.q_a(batch.query_a).unsqueeze(1)
        keys = self.k_a(batch.keys_a)
        slot_logits = (q * keys).sum(-1) / self.scale
        null_logits = self.null_logit.expand(batch.query_a.size(0), 1)
        attention = torch.cat([slot_logits, null_logits], -1).softmax(-1)
        slot_mass, null_mass = attention[:, :-1], attention[:, -1]
        values = self.output.values(batch.value_ids)
        return {
            "answer_logits": self.output.answer(slot_mass, values, null_mass),
            "retrieval_logits": torch.cat([slot_logits, null_logits], -1),
            "alpha": slot_mass,
            "slot_mass": slot_mass,
            "null_mass": null_mass,
        }


class JointAttention(nn.Module):
    """Strong standard baseline: A and N jointly determine one softmax."""

    def __init__(self, task: TaskConfig, d_model: int):
        super().__init__()
        self.task = task
        self.scale = math.sqrt(d_model)
        self.q_a, self.k_a = copied_pair(task.a_dimension, d_model)
        self.q_n, self.k_n = copied_pair(task.n_dimension, d_model)
        self.null_logit = nn.Parameter(torch.tensor(0.0))
        self.output = ValueOutput(task, d_model)

    def forward(self, batch: Batch) -> Dict[str, Tensor]:
        q_a = self.q_a(batch.query_a)
        k_a = self.k_a(batch.keys_a)
        # Center bounded template features so dot-product attention represents
        # agreement rather than merely rewarding jointly positive activity.
        q_n = self.q_n(batch.match_a - 0.5)
        k_n = self.k_n(batch.templates_n - 0.5)
        # Concatenated joint attention is equivalent to adding the two
        # within-channel dot products. Keeping them separate avoids unintended
        # A-query/N-key and N-query/A-key cross terms.
        slot_logits = (
            torch.einsum("bd,bsd->bs", q_a, k_a)
            + torch.einsum("bd,bsd->bs", q_n, k_n)
        ) / self.scale
        null_logits = self.null_logit.expand(batch.query_a.size(0), 1)
        attention = torch.cat([slot_logits, null_logits], -1).softmax(-1)
        slot_mass, null_mass = attention[:, :-1], attention[:, -1]
        values = self.output.values(batch.value_ids)
        return {
            "answer_logits": self.output.answer(slot_mass, values, null_mass),
            "retrieval_logits": torch.cat([slot_logits, null_logits], -1),
            "alpha": slot_mass,
            "slot_mass": slot_mass,
            "null_mass": null_mass,
        }


def build_model(method: str, task: TaskConfig, config: TrainConfig) -> nn.Module:
    if method == SOFT_CHEVRON:
        return SoftChevronAttention(
            task,
            config.d_model,
            theta_init=config.theta_init,
            sharpness_init=config.sharpness_init,
            match_init=config.match_init,
        )
    if method == JOINT_ATTENTION:
        return JointAttention(task, config.d_model)
    if method == A_ONLY_ATTENTION:
        return AOnlyAttention(task, config.d_model)
    raise ValueError("unknown method: %s" % method)


def group_mass(alpha: Tensor, target_groups: Tensor, task: TaskConfig) -> Tensor:
    slot_groups = (
        torch.arange(task.num_slots, device=alpha.device) // task.group_size
    )
    mask = slot_groups.unsqueeze(0).eq(target_groups.unsqueeze(1))
    return (alpha * mask).sum(-1)


def losses(
    method: str,
    outputs: Dict[str, Tensor],
    batch: Batch,
    task: TaskConfig,
    config: TrainConfig,
) -> Dict[str, Tensor]:
    answer = F.cross_entropy(outputs["answer_logits"], batch.answers)
    if method == SOFT_CHEVRON:
        retrieval = -torch.log(group_mass(outputs["alpha"], batch.target_groups, task) + 1e-8).mean()
        gate_targets = torch.zeros_like(outputs["r"])
        row = torch.arange(batch.answers.size(0), device=batch.answers.device)
        gate_targets[row[batch.matched], batch.target_slots[batch.matched]] = 1.0
        gate = F.binary_cross_entropy(outputs["r"], gate_targets)
    else:
        retrieval_targets = batch.target_slots.clone()
        retrieval_targets[~batch.matched] = task.num_slots
        retrieval = F.cross_entropy(outputs["retrieval_logits"], retrieval_targets)
        gate = torch.zeros((), device=answer.device)
    total = answer + config.retrieval_weight * retrieval + config.gate_weight * gate
    return {"total": total, "answer": answer, "retrieval": retrieval, "gate": gate}


def train_model(
    method: str,
    task: CategoryMatchingTask,
    config: TrainConfig,
    seed: int,
    device: torch.device,
) -> nn.Module:
    random.seed(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 10_000)
    model = build_model(method, task.config, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    model.train()
    for _ in range(config.steps):
        batch = task.batch(config.batch_size, generator, device)
        objective = losses(method, model(batch), batch, task.config, config)["total"]
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()
    model.eval()
    return model


@torch.no_grad()
def evaluate(
    method: str,
    model: nn.Module,
    task: CategoryMatchingTask,
    seed: int,
    device: torch.device,
    batches: int = 20,
    batch_size: int = 128,
) -> Dict[str, float]:
    generator = torch.Generator().manual_seed(seed + 90_000)
    totals = {
        "count": 0,
        "matched_count": 0,
        "no_match_count": 0,
        "correct": 0,
        "matched_correct": 0,
        "no_match_correct": 0,
        "group_correct": 0,
        "target_r": 0.0,
        "decoy_r": 0.0,
        "null_matched": 0.0,
        "null_no_match": 0.0,
        "theta": 0.0,
        "sharpness": 0.0,
        "target_mismatch": 0.0,
        "decoy_mismatch": 0.0,
    }
    c = task.config
    slot_groups = torch.arange(c.num_slots, device=device) // c.group_size
    for _ in range(batches):
        batch = task.batch(batch_size, generator, device)
        outputs = model(batch)
        prediction = outputs["answer_logits"].argmax(-1)
        matched = batch.matched
        no_match = ~matched
        totals["count"] += batch_size
        totals["matched_count"] += int(matched.sum().item())
        totals["no_match_count"] += int(no_match.sum().item())
        totals["correct"] += int(prediction.eq(batch.answers).sum().item())
        totals["matched_correct"] += int(
            prediction[matched].eq(batch.answers[matched]).sum().item()
        )
        totals["no_match_correct"] += int(
            prediction[no_match].eq(batch.answers[no_match]).sum().item()
        )
        per_group = []
        for group in range(c.num_groups):
            per_group.append(outputs["alpha"][:, slot_groups.eq(group)].sum(-1))
        predicted_group = torch.stack(per_group, -1).argmax(-1)
        totals["group_correct"] += int(
            predicted_group.eq(batch.target_groups).sum().item()
        )
        totals["null_matched"] += float(outputs["null_mass"][matched].sum().item())
        totals["null_no_match"] += float(outputs["null_mass"][no_match].sum().item())
        if method == SOFT_CHEVRON:
            row = torch.arange(batch_size, device=device)
            target_r = outputs["r"][row, batch.target_slots]
            totals["target_r"] += float(target_r[matched].sum().item())
            target_mismatch = outputs["mismatch"][row, batch.target_slots]
            totals["target_mismatch"] += float(
                target_mismatch[matched].sum().item()
            )
            decoy_mask = torch.ones_like(outputs["r"], dtype=torch.bool)
            decoy_mask[row, batch.target_slots] = False
            totals["decoy_r"] += float(outputs["r"][decoy_mask].sum().item())
            totals["decoy_mismatch"] += float(
                outputs["mismatch"][decoy_mask].sum().item()
            )
            totals["theta"] += float(outputs["theta"].item())
            totals["sharpness"] += float(outputs["sharpness"].item())
    result = {
        "answer_accuracy": totals["correct"] / totals["count"],
        "matched_accuracy": totals["matched_correct"] / totals["matched_count"],
        "no_match_accuracy": totals["no_match_correct"] / totals["no_match_count"],
        "group_accuracy": totals["group_correct"] / totals["count"],
        "null_mass_matched": totals["null_matched"] / totals["matched_count"],
        "null_mass_no_match": totals["null_no_match"] / totals["no_match_count"],
    }
    if method == SOFT_CHEVRON:
        result.update(
            {
                "target_r": totals["target_r"] / totals["matched_count"],
                "decoy_r": totals["decoy_r"] / (totals["count"] * (c.num_slots - 1)),
                "theta": totals["theta"] / batches,
                "sharpness": totals["sharpness"] / batches,
                "target_mismatch": totals["target_mismatch"]
                / totals["matched_count"],
                "decoy_mismatch": totals["decoy_mismatch"]
                / (totals["count"] * (c.num_slots - 1)),
            }
        )
    return result


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def summarize(rows: Sequence[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    keys = rows[0].keys()
    return {
        key: (
            statistics.mean(row[key] for row in rows),
            statistics.stdev(row[key] for row in rows) if len(rows) > 1 else 0.0,
        )
        for key in keys
    }


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
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--methods", choices=METHODS, nargs="+", default=list(METHODS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    task = CategoryMatchingTask(TaskConfig())
    train = TrainConfig(steps=args.steps)
    results: Dict[str, List[Dict[str, float]]] = {method: [] for method in args.methods}
    parameters: Dict[str, int] = {}
    for seed in args.seeds:
        for method in args.methods:
            model = train_model(method, task, train, seed, device)
            parameters[method] = parameter_count(model)
            results[method].append(evaluate(method, model, task, seed, device))
    for method in args.methods:
        print("\n%s parameters=%d" % (method, parameters[method]))
        for key, (average, spread) in summarize(results[method]).items():
            print("  %-24s %.4f +/- %.4f" % (key, average, spread))


if __name__ == "__main__":
    main()
