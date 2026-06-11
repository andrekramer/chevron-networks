#!/usr/bin/env python3
"""PyTorch experiment: temporary difference vs retained difference.

This is the second IDL test. The first experiment asked whether a usable
A-N signal exists. This one asks whether that signal can help a model avoid
retaining temporary differences while still adapting to persistent change.

Run:
    .venv-torch/bin/python temporary_difference_experiment.py

Outputs:
    runs_temporary_difference/trace.csv
    runs_temporary_difference/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Schedule:
    temp_start: int = 1500
    temp_end: int = 1760
    shift_start: int = 3600
    steps: int = 5200


def phase_name(t: int, schedule: Schedule) -> str:
    if schedule.temp_start <= t < schedule.temp_end:
        return "temporary_difference"
    if t >= schedule.shift_start:
        return "persistent_shift"
    return "stable"


def window_name(t: int, schedule: Schedule) -> str:
    if schedule.temp_start - 300 <= t < schedule.temp_start:
        return "pre_temp_stable_300"
    if schedule.temp_start <= t < schedule.temp_end:
        return "temporary_difference"
    if schedule.temp_end <= t < schedule.temp_end + 350:
        return "recovery_after_temp_350"
    if schedule.shift_start - 300 <= t < schedule.shift_start:
        return "pre_shift_stable_300"
    if schedule.shift_start <= t < schedule.shift_start + 350:
        return "shift_adaptation_350"
    if schedule.shift_start + 350 <= t < schedule.steps:
        return "post_shift_retained"
    return "other"


def make_batch(
    t: int,
    batch_size: int,
    schedule: Schedule,
    generator: torch.Generator,
    label_noise: float,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    x = torch.rand((batch_size, 2), generator=generator) * 2.0 - 1.0
    phase = phase_name(t, schedule)

    if phase == "temporary_difference" or phase == "persistent_shift":
        # Alternative concept: a real changed rule if it persists, noise if brief.
        score = -0.85 * x[:, 0] + x[:, 1].square() - 0.20 * x[:, 1] - 0.08
    else:
        # Stable concept.
        score = x[:, 1] + 0.90 * x[:, 0] - 0.65 * x[:, 0].square() + 0.15

    y = (score > 0.0).float()
    if label_noise > 0.0:
        flips = torch.rand((batch_size,), generator=generator) < label_noise
        y = torch.where(flips, 1.0 - y, y)
    return x, y, phase


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

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in list(self.net.children())[:-1]:
            h = layer(h)
        return h


def mean_param_distance(a: nn.Module, b: nn.Module) -> float:
    total = torch.tensor(0.0)
    count = 0
    with torch.no_grad():
        for pa, pb in zip(a.parameters(), b.parameters()):
            total += (pa - pb).square().sum().cpu()
            count += pa.numel()
    return math.sqrt(float(total) / max(count, 1))


def mean_param_move(model: nn.Module, checkpoint: list[torch.Tensor] | None) -> float:
    if checkpoint is None:
        return 0.0
    total = torch.tensor(0.0)
    count = 0
    with torch.no_grad():
        for p, p0 in zip(model.parameters(), checkpoint):
            total += (p.detach().cpu() - p0).square().sum()
            count += p.numel()
    return math.sqrt(float(total) / max(count, 1))


def checkpoint(model: nn.Module) -> list[torch.Tensor]:
    return [p.detach().cpu().clone() for p in model.parameters()]


class OnlineMLP:
    def __init__(self, hidden_dim: int, lr: float):
        self.model = SmallMLP(hidden_dim)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        logit = self.model(x)
        loss = F.binary_cross_entropy_with_logits(logit, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        with torch.no_grad():
            pred = (torch.sigmoid(logit) >= 0.5).float()
            acc = (pred == y).float().mean().item()
        return {
            "accuracy": acc,
            "loss": loss.item(),
            "rho": 1.0,
            "diff": 0.0,
            "persistence": 0.0,
            "fast_mismatch": 0.0,
            "baseline_mismatch": 0.0,
            "a_n_distance": 0.0,
            "n_move_from_temp_start": 0.0,
            "n_move_from_shift_start": 0.0,
        }


class Chevron:
    def __init__(
        self,
        hidden_dim: int,
        lr_a: float,
        lr_n: float,
        gated: bool,
        blend_n: float,
        fast_beta: float,
        baseline_beta: float,
        persistence_beta: float,
        theta: float,
        sharpness: float,
    ):
        self.a = SmallMLP(hidden_dim)
        self.n = SmallMLP(hidden_dim)
        self.n.load_state_dict(self.a.state_dict())
        self.opt_a = torch.optim.AdamW(self.a.parameters(), lr=lr_a, weight_decay=1e-4)
        self.opt_n = torch.optim.AdamW(self.n.parameters(), lr=lr_n, weight_decay=1e-4)
        self.gated = gated
        self.blend_n = blend_n
        self.fast_beta = fast_beta
        self.baseline_beta = baseline_beta
        self.persistence_beta = persistence_beta
        self.theta = theta
        self.sharpness = sharpness
        self.fast_mismatch = 0.0
        self.baseline_mismatch = 0.0
        self.persistence = 0.0
        self.n_at_temp_start: list[torch.Tensor] | None = None
        self.n_at_shift_start: list[torch.Tensor] | None = None

    def maybe_mark(self, t: int, schedule: Schedule) -> None:
        if t == schedule.temp_start:
            self.n_at_temp_start = checkpoint(self.n)
        if t == schedule.shift_start:
            self.n_at_shift_start = checkpoint(self.n)

    def step(self, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
        logit_a = self.a(x)
        logit_n = self.n(x)
        logit = (1.0 - self.blend_n) * logit_a + self.blend_n * logit_n
        loss = F.binary_cross_entropy_with_logits(logit, y)

        with torch.no_grad():
            hidden_diff = (self.a.hidden(x) - self.n.hidden(x)).abs().mean().item()
            output_diff = (logit_a - logit_n).abs().mean().item()
            param_diff = mean_param_distance(self.a, self.n)
            diff = 0.45 * hidden_diff + 0.45 * output_diff + 0.10 * param_diff
            self.fast_mismatch = (
                self.fast_beta * self.fast_mismatch + (1.0 - self.fast_beta) * diff
            )
            self.baseline_mismatch = (
                self.baseline_beta * self.baseline_mismatch
                + (1.0 - self.baseline_beta) * diff
            )
            excess_mismatch = max(0.0, self.fast_mismatch - self.baseline_mismatch)
            self.persistence = (
                self.persistence_beta * self.persistence
                + (1.0 - self.persistence_beta) * excess_mismatch
            )
            if self.gated:
                rho = torch.sigmoid(
                    torch.tensor(self.sharpness * (self.persistence - self.theta))
                ).item()
            else:
                rho = 1.0

        self.opt_a.zero_grad()
        self.opt_n.zero_grad()
        loss.backward()

        if self.gated:
            for p in self.n.parameters():
                if p.grad is not None:
                    p.grad.mul_(rho)

        self.opt_a.step()
        self.opt_n.step()

        if self.gated:
            # Slow structural retention: N moves toward A only when difference persists.
            with torch.no_grad():
                for pn, pa in zip(self.n.parameters(), self.a.parameters()):
                    pn.add_(0.010 * rho * (pa - pn))
        else:
            with torch.no_grad():
                for pn, pa in zip(self.n.parameters(), self.a.parameters()):
                    pn.add_(0.004 * (pa - pn))

        with torch.no_grad():
            pred = (torch.sigmoid(logit) >= 0.5).float()
            acc = (pred == y).float().mean().item()

        return {
            "accuracy": acc,
            "loss": loss.item(),
            "rho": rho,
            "diff": diff,
            "persistence": self.persistence,
            "fast_mismatch": self.fast_mismatch,
            "baseline_mismatch": self.baseline_mismatch,
            "a_n_distance": param_diff,
            "n_move_from_temp_start": mean_param_move(self.n, self.n_at_temp_start),
            "n_move_from_shift_start": mean_param_move(self.n, self.n_at_shift_start),
        }


def run_seed(args: argparse.Namespace, seed: int) -> list[dict[str, object]]:
    torch.manual_seed(seed)
    schedule = Schedule(
        temp_start=args.temp_start,
        temp_end=args.temp_end,
        shift_start=args.shift_start,
        steps=args.steps,
    )
    generator = torch.Generator().manual_seed(seed + 1000)
    models = {
        "MLP": OnlineMLP(args.hidden_dim, args.lr_mlp),
        "ChevronSlow": Chevron(
            args.hidden_dim,
            args.lr_a,
            args.lr_n,
            gated=False,
            blend_n=args.blend_n,
            fast_beta=args.fast_beta,
            baseline_beta=args.baseline_beta,
            persistence_beta=args.persistence_beta,
            theta=args.theta,
            sharpness=args.sharpness,
        ),
        "IDLGated": Chevron(
            args.hidden_dim,
            args.lr_a,
            args.lr_n,
            gated=True,
            blend_n=args.blend_n,
            fast_beta=args.fast_beta,
            baseline_beta=args.baseline_beta,
            persistence_beta=args.persistence_beta,
            theta=args.theta,
            sharpness=args.sharpness,
        ),
    }

    rows: list[dict[str, object]] = []
    for t in range(args.steps):
        x, y, phase = make_batch(t, args.batch_size, schedule, generator, args.label_noise)
        for model in models.values():
            if isinstance(model, Chevron):
                model.maybe_mark(t, schedule)
        for name, model in models.items():
            metrics = model.step(x, y)
            rows.append(
                {
                    "seed": seed,
                    "t": t,
                    "model": name,
                    "phase": phase,
                    "window": window_name(t, schedule),
                    **metrics,
                }
            )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        if row["window"] == "other":
            continue
        grouped.setdefault((str(row["model"]), str(row["window"])), []).append(row)

    summary: list[dict[str, object]] = []
    for (model, window), group in sorted(grouped.items()):
        n = len(group)
        summary.append(
            {
                "model": model,
                "window": window,
                "n": n,
                "accuracy": sum(float(r["accuracy"]) for r in group) / n,
                "loss": sum(float(r["loss"]) for r in group) / n,
                "rho": sum(float(r["rho"]) for r in group) / n,
                "diff": sum(float(r["diff"]) for r in group) / n,
                "persistence": sum(float(r["persistence"]) for r in group) / n,
                "fast_mismatch": sum(float(r["fast_mismatch"]) for r in group) / n,
                "baseline_mismatch": sum(float(r["baseline_mismatch"]) for r in group)
                / n,
                "a_n_distance": sum(float(r["a_n_distance"]) for r in group) / n,
                "n_move_from_temp_start": sum(
                    float(r["n_move_from_temp_start"]) for r in group
                )
                / n,
                "n_move_from_shift_start": sum(
                    float(r["n_move_from_shift_start"]) for r in group
                )
                / n,
            }
        )
    return summary


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: list[dict[str, object]]) -> None:
    windows = [
        "pre_temp_stable_300",
        "temporary_difference",
        "recovery_after_temp_350",
        "pre_shift_stable_300",
        "shift_adaptation_350",
        "post_shift_retained",
    ]
    models = ["MLP", "ChevronSlow", "IDLGated"]
    print("Temporary difference experiment")
    print("model          window                    acc    loss   rho    P      NmoveT  NmoveS")
    for model in models:
        for window in windows:
            row = next(r for r in summary if r["model"] == model and r["window"] == window)
            print(
                f"{model:<14} {window:<25} "
                f"{row['accuracy']:.3f}  {row['loss']:.3f}  {row['rho']:.3f}  "
                f"{row['persistence']:.3f}  {row['n_move_from_temp_start']:.4f}  "
                f"{row['n_move_from_shift_start']:.4f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5200)
    parser.add_argument("--temp-start", type=int, default=1500)
    parser.add_argument("--temp-end", type=int, default=1550)
    parser.add_argument("--shift-start", type=int, default=3600)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--label-noise", type=float, default=0.02)
    parser.add_argument("--lr-mlp", type=float, default=0.003)
    parser.add_argument("--lr-a", type=float, default=0.004)
    parser.add_argument("--lr-n", type=float, default=0.0014)
    parser.add_argument("--blend-n", type=float, default=0.38)
    parser.add_argument("--fast-beta", type=float, default=0.96)
    parser.add_argument("--baseline-beta", type=float, default=0.999)
    parser.add_argument("--persistence-beta", type=float, default=0.992)
    parser.add_argument("--theta", type=float, default=0.23)
    parser.add_argument("--sharpness", type=float, default=30.0)
    parser.add_argument("--out-dir", default="runs_temporary_difference")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.temp_start < args.temp_end < args.shift_start < args.steps):
        raise ValueError("Expected temp_start < temp_end < shift_start < steps.")

    all_rows: list[dict[str, object]] = []
    for seed in range(args.seeds):
        all_rows.extend(run_seed(args, seed))

    summary = summarize(all_rows)
    write_csv(os.path.join(args.out_dir, "trace.csv"), all_rows)
    write_csv(os.path.join(args.out_dir, "summary.csv"), summary)
    print_summary(summary)
    print(f"\nWrote {args.out_dir}/trace.csv and {args.out_dir}/summary.csv")


if __name__ == "__main__":
    main()
