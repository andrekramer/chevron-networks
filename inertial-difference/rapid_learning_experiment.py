#!/usr/bin/env python3
"""PyTorch experiment: rapid provisional learning and slow consolidation.

This fifth test asks whether A can learn a small new rule quickly while N only
consolidates rules that recur.

The stream has:
  - a stable background task learned by N,
  - one-shot local exception rules,
  - recurring local exception rules.

For chevron models, each episode starts with A copied from N. A then adapts
quickly to a tiny support set. N consolidates toward A after the episode:
  - ChevronSlow consolidates every episode.
  - IDLGated consolidates only when the same rule has persisted.

Run:
    .venv-torch/bin/python rapid_learning_experiment.py

Outputs:
    runs_rapid_learning/episodes.csv
    runs_rapid_learning/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Rule:
    rule_id: str
    kind: str
    center_x: float
    center_y: float
    radius: float


class SmallMLP(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def copy_from(self, other: "SmallMLP") -> None:
        self.load_state_dict(other.state_dict())


def background_labels(x: torch.Tensor) -> torch.Tensor:
    score = x[:, 1] + 0.85 * x[:, 0] - 0.55 * x[:, 0].square() + 0.12
    return (score > 0.0).float()


def exception_labels(x: torch.Tensor) -> torch.Tensor:
    return torch.ones((x.shape[0],), dtype=torch.float32)


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) >= 0.5).float()
    return (pred == y).float().mean().item()


def sample_background(batch_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand((batch_size, 2), generator=generator) * 2.0 - 1.0
    return x, background_labels(x)


def sample_rule(rule: Rule, n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    # Rejection sample points inside the local exception region.
    xs: list[torch.Tensor] = []
    needed = n
    center = torch.tensor([rule.center_x, rule.center_y])
    while needed > 0:
        candidates = torch.rand((max(needed * 4, 32), 2), generator=generator) * 2.0 - 1.0
        keep = (candidates - center).norm(dim=1) <= rule.radius
        kept = candidates[keep][:needed]
        if len(kept) > 0:
            xs.append(kept)
            needed -= len(kept)
    x = torch.cat(xs, dim=0)
    return x, exception_labels(x)


def fixed_rule_probe(rule: Rule, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return sample_rule(rule, n, generator)


def param_distance(a: nn.Module, b: nn.Module) -> float:
    total = torch.tensor(0.0)
    count = 0
    with torch.no_grad():
        for pa, pb in zip(a.parameters(), b.parameters()):
            total += (pa - pb).square().sum().cpu()
            count += pa.numel()
    return math.sqrt(float(total) / max(count, 1))


def train_steps(
    model: SmallMLP,
    x: torch.Tensor,
    y: torch.Tensor,
    lr: float,
    steps: int,
    weight_decay: float,
) -> None:
    opt = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        logits = model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()


class OnlineMLP:
    def __init__(self, hidden_dim: int, lr: float, weight_decay: float):
        self.model = SmallMLP(hidden_dim)
        self.lr = lr
        self.weight_decay = weight_decay

    def pretrain(self, steps: int, batch_size: int, generator: torch.Generator) -> None:
        opt = torch.optim.AdamW(self.model.parameters(), lr=0.003, weight_decay=1e-4)
        for _ in range(steps):
            x, y = sample_background(batch_size, generator)
            loss = F.binary_cross_entropy_with_logits(self.model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    def episode(
        self,
        rule: Rule,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        args: argparse.Namespace,
    ) -> dict[str, float]:
        before = accuracy(self.model(query_x), query_y)
        train_steps(
            self.model,
            support_x,
            support_y,
            self.lr,
            args.fast_steps,
            self.weight_decay,
        )
        after = accuracy(self.model(query_x), query_y)
        return {
            "provisional_acc": after,
            "before_acc": before,
            "n_query_acc": 0.0,
            "rho": 1.0,
            "a_n_distance": 0.0,
        }

    def background_acc(self, x: torch.Tensor, y: torch.Tensor) -> float:
        return accuracy(self.model(x), y)

    def rule_acc(self, x: torch.Tensor, y: torch.Tensor) -> float:
        return accuracy(self.model(x), y)


class ChevronEpisode:
    def __init__(
        self,
        hidden_dim: int,
        fast_lr: float,
        weight_decay: float,
        gated: bool,
        consolidate_rate: float,
        gate_threshold: float,
        gate_sharpness: float,
    ):
        self.n = SmallMLP(hidden_dim)
        self.a = SmallMLP(hidden_dim)
        self.fast_lr = fast_lr
        self.weight_decay = weight_decay
        self.gated = gated
        self.consolidate_rate = consolidate_rate
        self.gate_threshold = gate_threshold
        self.gate_sharpness = gate_sharpness
        self.rule_persistence: dict[str, float] = {}

    def pretrain(self, steps: int, batch_size: int, generator: torch.Generator) -> None:
        opt = torch.optim.AdamW(self.n.parameters(), lr=0.003, weight_decay=1e-4)
        for _ in range(steps):
            x, y = sample_background(batch_size, generator)
            loss = F.binary_cross_entropy_with_logits(self.n(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.a.copy_from(self.n)

    def _rho(self, rule: Rule) -> float:
        if not self.gated:
            return 1.0
        old = self.rule_persistence.get(rule.rule_id, 0.0)
        new = old + 1.0
        self.rule_persistence[rule.rule_id] = new
        return torch.sigmoid(
            torch.tensor(self.gate_sharpness * (new - self.gate_threshold))
        ).item()

    def episode(
        self,
        rule: Rule,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        args: argparse.Namespace,
    ) -> dict[str, float]:
        self.a.copy_from(self.n)
        before = accuracy(self.a(query_x), query_y)
        train_steps(
            self.a,
            support_x,
            support_y,
            self.fast_lr,
            args.fast_steps,
            self.weight_decay,
        )
        provisional = accuracy(self.a(query_x), query_y)
        distance = param_distance(self.a, self.n)
        rho = self._rho(rule)
        with torch.no_grad():
            amount = self.consolidate_rate * rho
            for pn, pa in zip(self.n.parameters(), self.a.parameters()):
                pn.add_(amount * (pa - pn))
        n_query = accuracy(self.n(query_x), query_y)
        return {
            "provisional_acc": provisional,
            "before_acc": before,
            "n_query_acc": n_query,
            "rho": rho,
            "a_n_distance": distance,
        }

    def background_acc(self, x: torch.Tensor, y: torch.Tensor) -> float:
        return accuracy(self.n(x), y)

    def rule_acc(self, x: torch.Tensor, y: torch.Tensor) -> float:
        return accuracy(self.n(x), y)


def make_rules(args: argparse.Namespace, rng: random.Random) -> tuple[list[Rule], list[Rule]]:
    recurring: list[Rule] = []
    one_shot: list[Rule] = []
    used: list[tuple[float, float]] = []

    def new_center() -> tuple[float, float]:
        for _ in range(10_000):
            cx = rng.uniform(-0.75, 0.75)
            cy = rng.uniform(-0.75, 0.75)
            center = torch.tensor([[cx, cy]], dtype=torch.float32)
            if background_labels(center).item() > 0.5:
                continue
            if all((cx - x) ** 2 + (cy - y) ** 2 > 0.035 for x, y in used):
                used.append((cx, cy))
                return cx, cy
        raise RuntimeError("could not place non-overlapping rule center")

    for i in range(args.recurring_rules):
        cx, cy = new_center()
        recurring.append(Rule(f"recurring_{i}", "recurring", cx, cy, args.radius))
    for i in range(args.one_shot_rules):
        cx, cy = new_center()
        one_shot.append(Rule(f"one_shot_{i}", "one_shot", cx, cy, args.radius))
    return recurring, one_shot


def make_episode_order(
    recurring: list[Rule], one_shot: list[Rule], args: argparse.Namespace, rng: random.Random
) -> list[Rule]:
    episodes: list[Rule] = []
    for rule in recurring:
        episodes.extend([rule] * args.recurring_repeats)
    episodes.extend(one_shot)
    rng.shuffle(episodes)
    return episodes


def evaluate_rule_set(
    model: OnlineMLP | ChevronEpisode,
    rules: list[Rule],
    probe_cache: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> float:
    vals = []
    for rule in rules:
        x, y = probe_cache[rule.rule_id]
        vals.append(model.rule_acc(x, y))
    return sum(vals) / len(vals)


def run_seed(args: argparse.Namespace, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1000)
    recurring, one_shot = make_rules(args, rng)
    episodes = make_episode_order(recurring, one_shot, args, rng)

    bg_x, bg_y = sample_background(args.probe_size, torch.Generator().manual_seed(seed + 2000))
    probe_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for idx, rule in enumerate(recurring + one_shot):
        probe_cache[rule.rule_id] = fixed_rule_probe(rule, args.probe_size, seed + 3000 + idx)

    models = {
        "MLP": OnlineMLP(args.hidden_dim, args.mlp_fast_lr, args.weight_decay),
        "ChevronSlow": ChevronEpisode(
            args.hidden_dim,
            args.fast_lr,
            args.weight_decay,
            gated=False,
            consolidate_rate=args.consolidate_rate,
            gate_threshold=args.gate_threshold,
            gate_sharpness=args.gate_sharpness,
        ),
        "IDLGated": ChevronEpisode(
            args.hidden_dim,
            args.fast_lr,
            args.weight_decay,
            gated=True,
            consolidate_rate=args.consolidate_rate,
            gate_threshold=args.gate_threshold,
            gate_sharpness=args.gate_sharpness,
        ),
    }
    for model in models.values():
        model.pretrain(args.pretrain_steps, args.batch_size, generator)

    episode_rows: list[dict[str, object]] = []
    for episode_idx, rule in enumerate(episodes):
        exception_x, exception_y = sample_rule(rule, args.support_size, generator)
        anchor_x, anchor_y = sample_background(args.anchor_size, generator)
        support_x = torch.cat([exception_x.repeat((args.exception_weight, 1)), anchor_x], dim=0)
        support_y = torch.cat([exception_y.repeat(args.exception_weight), anchor_y], dim=0)
        query_x, query_y = sample_rule(rule, args.query_size, generator)
        occurrence = sum(1 for prior in episodes[: episode_idx + 1] if prior.rule_id == rule.rule_id)
        for name, model in models.items():
            metrics = model.episode(rule, support_x, support_y, query_x, query_y, args)
            episode_rows.append(
                {
                    "seed": seed,
                    "episode": episode_idx,
                    "model": name,
                    "rule_id": rule.rule_id,
                    "rule_kind": rule.kind,
                    "occurrence": occurrence,
                    **metrics,
                    "background_acc": model.background_acc(bg_x, bg_y),
                }
            )

    summary_rows: list[dict[str, object]] = []
    for name, model in models.items():
        recurrent_n = evaluate_rule_set(model, recurring, probe_cache)
        one_shot_n = evaluate_rule_set(model, one_shot, probe_cache)
        background = model.background_acc(bg_x, bg_y)
        rows = [r for r in episode_rows if r["model"] == name]
        recurrent_rows = [r for r in rows if r["rule_kind"] == "recurring"]
        one_shot_rows = [r for r in rows if r["rule_kind"] == "one_shot"]
        late_recurrent_rows = [
            r
            for r in recurrent_rows
            if int(r["occurrence"]) >= max(2, args.recurring_repeats - 1)
        ]
        summary_rows.append(
            {
                "seed": seed,
                "model": name,
                "provisional_acc_all": mean(rows, "provisional_acc"),
                "provisional_acc_one_shot": mean(one_shot_rows, "provisional_acc"),
                "provisional_acc_recurring_late": mean(
                    late_recurrent_rows, "provisional_acc"
                ),
                "n_acc_recurring_final": recurrent_n,
                "n_acc_one_shot_final": one_shot_n,
                "n_consolidation_selectivity": recurrent_n - one_shot_n,
                "false_consolidation_gap": one_shot_n,
                "background_acc_final": background,
                "rho_one_shot": mean(one_shot_rows, "rho"),
                "rho_recurring_late": mean(late_recurrent_rows, "rho"),
            }
        )
    return episode_rows, summary_rows


def mean(rows: list[dict[str, object]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(float(r[key]) for r in rows) / len(rows)


def aggregate(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for model in ["MLP", "ChevronSlow", "IDLGated"]:
        rows = [r for r in summary_rows if r["model"] == model]
        row: dict[str, object] = {"model": model, "n": len(rows)}
        for key in [
            "provisional_acc_all",
            "provisional_acc_one_shot",
            "provisional_acc_recurring_late",
            "n_acc_recurring_final",
            "n_acc_one_shot_final",
            "n_consolidation_selectivity",
            "false_consolidation_gap",
            "background_acc_final",
            "rho_one_shot",
            "rho_recurring_late",
        ]:
            row[key] = mean(rows, key)
        out.append(row)
    return out


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, object]]) -> None:
    print("Rapid provisional learning experiment")
    print("model          prov_all prov_1shot rec_N  one_N  sel_N bg    rho_1 rho_rec")
    for row in rows:
        print(
            f"{str(row['model']):<14} "
            f"{float(row['provisional_acc_all']):.3f}    "
            f"{float(row['provisional_acc_one_shot']):.3f}      "
            f"{float(row['n_acc_recurring_final']):.3f}  "
            f"{float(row['n_acc_one_shot_final']):.3f}  "
            f"{float(row['n_consolidation_selectivity']):.3f} "
            f"{float(row['background_acc_final']):.3f} "
            f"{float(row['rho_one_shot']):.3f} "
            f"{float(row['rho_recurring_late']):.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--pretrain-steps", type=int, default=1800)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--support-size", type=int, default=6)
    parser.add_argument("--exception-weight", type=int, default=6)
    parser.add_argument("--anchor-size", type=int, default=32)
    parser.add_argument("--query-size", type=int, default=96)
    parser.add_argument("--probe-size", type=int, default=512)
    parser.add_argument("--recurring-rules", type=int, default=4)
    parser.add_argument("--recurring-repeats", type=int, default=10)
    parser.add_argument("--one-shot-rules", type=int, default=16)
    parser.add_argument("--radius", type=float, default=0.28)
    parser.add_argument("--fast-steps", type=int, default=24)
    parser.add_argument("--fast-lr", type=float, default=0.10)
    parser.add_argument("--mlp-fast-lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--consolidate-rate", type=float, default=0.50)
    parser.add_argument("--gate-threshold", type=float, default=2.5)
    parser.add_argument("--gate-sharpness", type=float, default=3.0)
    parser.add_argument("--out-dir", default="runs_rapid_learning")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_episodes: list[dict[str, object]] = []
    all_summaries: list[dict[str, object]] = []
    for seed in range(args.seeds):
        episode_rows, summary_rows = run_seed(args, seed)
        all_episodes.extend(episode_rows)
        all_summaries.extend(summary_rows)
    aggregate_rows = aggregate(all_summaries)
    write_csv(os.path.join(args.out_dir, "episodes.csv"), all_episodes)
    write_csv(os.path.join(args.out_dir, "summary_by_seed.csv"), all_summaries)
    write_csv(os.path.join(args.out_dir, "summary.csv"), aggregate_rows)
    print_summary(aggregate_rows)
    print(
        f"\nWrote {args.out_dir}/episodes.csv, "
        f"{args.out_dir}/summary_by_seed.csv, and {args.out_dir}/summary.csv"
    )


if __name__ == "__main__":
    main()
