"""Phase-one revocable associative recall experiment.

The A stream receives facts and the query, but control instructions are masked.
The N stream receives the complete sequence and produces query-dependent gates.
This makes retrieval and permission separately measurable by construction.
"""

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


ACTIVE = 0
REVOKED = 1
RESTORED = 2
MODE_NAMES = {ACTIVE: "active", REVOKED: "revoked", RESTORED: "restored"}


@dataclass
class Batch:
    a_tokens: Tensor
    n_tokens: Tensor
    value_positions: Tensor
    target_slot: Tensor
    permission: Tensor
    answer: Tensor
    mode: Tensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(**{name: value.to(device) for name, value in vars(self).items()})


class RecallTask:
    """Generates fresh random key-value memories and balanced control states."""

    PAD = 0
    BOS = 1
    FACT = 2
    REVOKE = 3
    RESTORE = 4
    QUERY = 5

    def __init__(self, num_keys: int = 16, num_values: int = 16, num_facts: int = 4):
        if num_facts < 2 or num_facts > num_keys:
            raise ValueError("num_facts must be between 2 and num_keys")
        if num_values < num_facts:
            raise ValueError("num_values must be at least num_facts")
        self.num_keys = num_keys
        self.num_values = num_values
        self.num_facts = num_facts
        self.key_offset = 6
        self.value_offset = self.key_offset + num_keys
        self.vocab_size = self.value_offset + num_values
        self.idk_class = num_values

    def key_token(self, key: int) -> int:
        return self.key_offset + key

    def value_token(self, value: int) -> int:
        return self.value_offset + value

    def _example(self, rng: random.Random, mode: int) -> Tuple[List[int], ...]:
        keys = rng.sample(range(self.num_keys), self.num_facts)
        values = rng.sample(range(self.num_values), self.num_facts)
        target_slot = rng.randrange(self.num_facts)
        target_key = keys[target_slot]

        full = [self.BOS]
        value_positions = []
        for key, value in zip(keys, values):
            full.extend([self.FACT, self.key_token(key), self.value_token(value)])
            value_positions.append(len(full) - 1)

        # Two distractor controls force N to track permissions by key.
        decoy_slots = [slot for slot in range(self.num_facts) if slot != target_slot]
        permissions = [1.0] * self.num_facts
        for _ in range(2):
            slot = rng.choice(decoy_slots)
            op = rng.choice([self.REVOKE, self.RESTORE])
            full.extend([op, self.key_token(keys[slot])])
            permissions[slot] = float(op == self.RESTORE)

        # Every mode has two target controls, preventing sequence-length shortcuts.
        if mode == ACTIVE:
            target_ops = [self.RESTORE, self.RESTORE]
        elif mode == REVOKED:
            target_ops = [self.RESTORE, self.REVOKE]
        elif mode == RESTORED:
            target_ops = [self.REVOKE, self.RESTORE]
        else:
            raise ValueError("invalid mode")
        for op in target_ops:
            full.extend([op, self.key_token(target_key)])
            permissions[target_slot] = float(op == self.RESTORE)

        full.extend([self.QUERY, self.key_token(target_key)])

        # A cannot infer permission: mask operation and controlled-key tokens.
        a_tokens = list(full)
        fact_end = 1 + 3 * self.num_facts
        query_start = len(full) - 2
        for index in range(fact_end, query_start):
            a_tokens[index] = self.PAD

        answer = values[target_slot] if permissions[target_slot] else self.idk_class
        return (
            a_tokens,
            full,
            value_positions,
            target_slot,
            permissions,
            answer,
            mode,
        )

    def batch(
        self,
        batch_size: int,
        rng: random.Random,
        device: Optional[torch.device] = None,
        balanced: bool = True,
    ) -> Batch:
        rows = []
        for index in range(batch_size):
            mode = index % 3 if balanced else rng.randrange(3)
            rows.append(self._example(rng, mode))
        rng.shuffle(rows)
        fields = list(zip(*rows))
        batch = Batch(
            a_tokens=torch.tensor(fields[0], dtype=torch.long),
            n_tokens=torch.tensor(fields[1], dtype=torch.long),
            value_positions=torch.tensor(fields[2], dtype=torch.long),
            target_slot=torch.tensor(fields[3], dtype=torch.long),
            permission=torch.tensor(fields[4], dtype=torch.float32),
            answer=torch.tensor(fields[5], dtype=torch.long),
            mode=torch.tensor(fields[6], dtype=torch.long),
        )
        return batch.to(device) if device is not None else batch


class CausalEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model, padding_idx=RecallTask.PAD)
        self.position = nn.Embedding(max_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor) -> Tensor:
        length = tokens.size(1)
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token(tokens) + self.position(positions)[None, :, :]
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1
        )
        hidden = self.encoder(
            hidden,
            mask=causal_mask,
            src_key_padding_mask=tokens.eq(RecallTask.PAD),
        )
        return self.norm(hidden)


def gather_slots(hidden: Tensor, positions: Tensor) -> Tensor:
    index = positions.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
    return hidden.gather(1, index)


class ChevronAttention(nn.Module):
    def __init__(
        self,
        task: RecallTask,
        max_length: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.a_encoder = CausalEncoder(
            task.vocab_size, max_length, d_model, nhead, num_layers, dropout
        )
        self.n_encoder = CausalEncoder(
            task.vocab_size, max_length, d_model, nhead, num_layers, dropout
        )
        self.query = nn.Linear(d_model, d_model, bias=False)
        self.key = nn.Linear(d_model, d_model, bias=False)
        self.value = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        self.null_value = nn.Parameter(torch.zeros(d_model))
        self.output = nn.Linear(d_model, task.num_values + 1)
        self.scale = math.sqrt(d_model)

    def forward(self, batch: Batch) -> Dict[str, Tensor]:
        a_hidden = self.a_encoder(batch.a_tokens)
        n_hidden = self.n_encoder(batch.n_tokens)
        a_memory = gather_slots(a_hidden, batch.value_positions)
        n_memory = gather_slots(n_hidden, batch.value_positions)
        a_query = a_hidden[:, -1]
        n_query = n_hidden[:, -1]

        q = self.query(a_query).unsqueeze(1)
        k = self.key(a_memory)
        values = self.value(a_memory)
        retrieval_logits = (q * k).sum(-1) / self.scale
        alpha = retrieval_logits.softmax(dim=-1)

        expanded_query = n_query.unsqueeze(1).expand_as(n_memory)
        gate_input = torch.cat(
            [expanded_query, n_memory, expanded_query * n_memory], dim=-1
        )
        gate_logits = self.gate(gate_input).squeeze(-1)
        gates = gate_logits.sigmoid()
        admitted = alpha * gates
        admitted_mass = admitted.sum(-1, keepdim=True)
        output_value = (admitted.unsqueeze(-1) * values).sum(1)
        output_value = output_value + (1.0 - admitted_mass) * self.null_value
        return {
            "answer_logits": self.output(output_value),
            "retrieval_logits": retrieval_logits,
            "alpha": alpha,
            "gate_logits": gate_logits,
            "gates": gates,
            "admitted_mass": admitted_mass.squeeze(-1),
        }


class TransformerBaseline(nn.Module):
    def __init__(
        self,
        task: RecallTask,
        max_length: int,
        d_model: int = 96,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = CausalEncoder(
            task.vocab_size, max_length, d_model, nhead, num_layers, dropout
        )
        self.output = nn.Linear(d_model, task.num_values + 1)

    def forward(self, batch: Batch) -> Dict[str, Tensor]:
        hidden = self.encoder(batch.n_tokens)
        return {"answer_logits": self.output(hidden[:, -1])}


def losses(
    outputs: Dict[str, Tensor],
    batch: Batch,
    retrieval_weight: float,
    permission_weight: float,
) -> Dict[str, Tensor]:
    answer = F.cross_entropy(outputs["answer_logits"], batch.answer)
    result = {"answer": answer}
    total = answer
    if "retrieval_logits" in outputs:
        retrieval = F.cross_entropy(outputs["retrieval_logits"], batch.target_slot)
        permission = F.binary_cross_entropy_with_logits(
            outputs["gate_logits"], batch.permission
        )
        total = total + retrieval_weight * retrieval + permission_weight * permission
        result.update(retrieval=retrieval, permission=permission)
    result["total"] = total
    return result


@torch.no_grad()
def evaluate(
    model: nn.Module,
    task: RecallTask,
    rng: random.Random,
    device: torch.device,
    batches: int,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {"correct": 0.0, "count": 0.0}
    for mode in MODE_NAMES:
        totals["correct_%d" % mode] = 0.0
        totals["count_%d" % mode] = 0.0
        totals["target_gate_%d" % mode] = 0.0
        totals["gate_count_%d" % mode] = 0.0
    totals.update(retrieval_correct=0.0, target_alpha=0.0, permission_correct=0.0)

    for _ in range(batches):
        batch = task.batch(batch_size, rng, device=device, balanced=True)
        outputs = model(batch)
        predictions = outputs["answer_logits"].argmax(-1)
        correct = predictions.eq(batch.answer)
        totals["correct"] += correct.sum().item()
        totals["count"] += batch_size
        for mode in MODE_NAMES:
            mask = batch.mode.eq(mode)
            totals["correct_%d" % mode] += correct[mask].sum().item()
            totals["count_%d" % mode] += mask.sum().item()

        if "alpha" in outputs:
            row = torch.arange(batch_size, device=device)
            target_alpha = outputs["alpha"][row, batch.target_slot]
            target_gate = outputs["gates"][row, batch.target_slot]
            totals["retrieval_correct"] += outputs["alpha"].argmax(-1).eq(
                batch.target_slot
            ).sum().item()
            totals["target_alpha"] += target_alpha.sum().item()
            totals["permission_correct"] += outputs["gates"].gt(0.5).eq(
                batch.permission.bool()
            ).sum().item()
            for mode in MODE_NAMES:
                mask = batch.mode.eq(mode)
                totals["target_gate_%d" % mode] += target_gate[mask].sum().item()
                totals["gate_count_%d" % mode] += mask.sum().item()

    metrics = {"answer_accuracy": totals["correct"] / totals["count"]}
    for mode, name in MODE_NAMES.items():
        metrics["answer_%s" % name] = totals["correct_%d" % mode] / totals[
            "count_%d" % mode
        ]
    if totals["target_alpha"]:
        metrics.update(
            retrieval_accuracy=totals["retrieval_correct"] / totals["count"],
            target_alpha=totals["target_alpha"] / totals["count"],
            permission_accuracy=totals["permission_correct"]
            / (totals["count"] * task.num_facts),
        )
        for mode, name in MODE_NAMES.items():
            metrics["gate_%s" % name] = totals["target_gate_%d" % mode] / totals[
                "gate_count_%d" % mode
            ]
    model.train()
    return metrics


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def format_metrics(metrics: Dict[str, float]) -> str:
    return " ".join("%s=%.3f" % item for item in metrics.items())


def train_one(
    name: str,
    model: nn.Module,
    task: RecallTask,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, float]:
    train_rng = random.Random(args.seed)
    eval_rng = random.Random(args.seed + 10_000)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print("%s parameters=%d device=%s" % (name, parameter_count(model), device))

    model.train()
    for step in range(1, args.steps + 1):
        batch = task.batch(args.batch_size, train_rng, device=device, balanced=True)
        outputs = model(batch)
        current_losses = losses(
            outputs, batch, args.retrieval_weight, args.permission_weight
        )
        optimizer.zero_grad(set_to_none=True)
        current_losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            metrics = evaluate(
                model,
                task,
                eval_rng,
                device,
                args.eval_batches,
                args.batch_size,
            )
            print(
                "%s step=%d loss=%.4f %s"
                % (name, step, current_losses["total"].item(), format_metrics(metrics))
            )
    return evaluate(
        model, task, eval_rng, device, args.eval_batches * 2, args.batch_size
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["chevron", "baseline", "all"], default="all")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--retrieval-weight", type=float, default=1.0)
    parser.add_argument("--permission-weight", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-keys", type=int, default=16)
    parser.add_argument("--num-values", type=int, default=16)
    parser.add_argument("--num-facts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")
        return torch.device("mps")
    if name == "auto" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    task = RecallTask(args.num_keys, args.num_values, args.num_facts)
    sample = task.batch(1, random.Random(args.seed))
    max_length = sample.n_tokens.size(1)
    results = {}
    if args.model in ("chevron", "all"):
        torch.manual_seed(args.seed)
        model = ChevronAttention(
            task, max_length, args.d_model, args.heads, args.layers
        )
        results["chevron"] = train_one("chevron", model, task, args, device)
    if args.model in ("baseline", "all"):
        torch.manual_seed(args.seed)
        # sqrt(2) width roughly compensates for Chevron's two encoder streams.
        baseline_width = int(round(args.d_model * math.sqrt(2) / args.heads)) * args.heads
        model = TransformerBaseline(
            task, max_length, baseline_width, args.heads, args.layers
        )
        results["baseline"] = train_one("baseline", model, task, args, device)
    print("final")
    for name, metrics in results.items():
        print("%s %s" % (name, format_metrics(metrics)))


if __name__ == "__main__":
    main()
