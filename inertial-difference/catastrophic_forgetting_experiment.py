#!/usr/bin/env python3
"""PyTorch experiment: IDL and catastrophic forgetting.

This third test uses sequential learning:
  1. Train on Task 1 until it is stable.
  2. Train on Task 2 with no replay of Task 1.
  3. Probe both tasks throughout.

The diagnostic is whether the IDL retained channel N preserves Task 1 better
while the adaptive channel A learns Task 2. The default Task 2 phase is long
enough for Task 2 to be learned, but short enough to inspect the catastrophic
forgetting window before very long-run consolidation overwrites N.

Run:
    .venv-torch/bin/python catastrophic_forgetting_experiment.py

Outputs:
    runs_catastrophic_forgetting/probes.csv
    runs_catastrophic_forgetting/summary.csv
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
class ProbeResult:
    task1: float
    task2: float


def task_labels(x: torch.Tensor, task: str) -> torch.Tensor:
    if task == "task1":
        score = x[:, 1] + 0.85 * x[:, 0] - 0.55 * x[:, 0].square() + 0.12
    elif task == "task2":
        score = -0.75 * x[:, 0] + x[:, 1].square() - 0.35 * x[:, 1] - 0.10
    else:
        raise ValueError(f"unknown task: {task}")
    return (score > 0.0).float()


def sample_batch(
    task: str,
    batch_size: int,
    generator: torch.Generator,
    label_noise: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.rand((batch_size, 2), generator=generator) * 2.0 - 1.0
    y = task_labels(x, task)
    if label_noise > 0.0:
        flips = torch.rand((batch_size,), generator=generator) < label_noise
        y = torch.where(flips, 1.0 - y, y)
    return x, y


def make_probe(task: str, n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand((n, 2), generator=generator) * 2.0 - 1.0
    return x, task_labels(x, task)


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


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) >= 0.5).float()
    return (pred == y).float().mean().item()


def param_distance(a: nn.Module, b: nn.Module) -> float:
    total = torch.tensor(0.0)
    count = 0
    with torch.no_grad():
        for pa, pb in zip(a.parameters(), b.parameters()):
            total += (pa - pb).square().sum().cpu()
            count += pa.numel()
    return math.sqrt(float(total) / max(count, 1))


class OnlineMLP:
    def __init__(self, hidden_dim: int, lr: float):
        self.model = SmallMLP(hidden_dim)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def step(self, x: torch.Tensor, y: torch.Tensor, phase: str = "") -> float:
        logits = self.model(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def probe(
        self,
        x_task1: torch.Tensor,
        y_task1: torch.Tensor,
        x_task2: torch.Tensor,
        y_task2: torch.Tensor,
    ) -> dict[str, float]:
        with torch.no_grad():
            return {
                "combined_task1": accuracy_from_logits(self.model(x_task1), y_task1),
                "combined_task2": accuracy_from_logits(self.model(x_task2), y_task2),
                "a_channel_task1": 0.0,
                "a_channel_task2": 0.0,
                "n_channel_task1": 0.0,
                "n_channel_task2": 0.0,
                "rho": 1.0,
                "persistence": 0.0,
                "a_n_distance": 0.0,
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
        retain_pull: float,
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
        self.retain_pull = retain_pull
        self.fast_mismatch = 0.0
        self.baseline_mismatch = 0.0
        self.persistence = 0.0
        self.rho = 1.0

    def _combined_logits(self, x: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.blend_n) * self.a(x) + self.blend_n * self.n(x)

    def _update_gate(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            hidden_diff = (self.a.hidden(x) - self.n.hidden(x)).abs().mean().item()
            output_diff = (self.a(x) - self.n(x)).abs().mean().item()
            distance = param_distance(self.a, self.n)
            diff = 0.45 * hidden_diff + 0.45 * output_diff + 0.10 * distance
            self.fast_mismatch = (
                self.fast_beta * self.fast_mismatch + (1.0 - self.fast_beta) * diff
            )
            self.baseline_mismatch = (
                self.baseline_beta * self.baseline_mismatch
                + (1.0 - self.baseline_beta) * diff
            )
            excess = max(0.0, self.fast_mismatch - self.baseline_mismatch)
            self.persistence = (
                self.persistence_beta * self.persistence + (1.0 - self.persistence_beta) * excess
            )
            if self.gated:
                self.rho = torch.sigmoid(
                    torch.tensor(self.sharpness * (self.persistence - self.theta))
                ).item()
            else:
                self.rho = 1.0

    def step(self, x: torch.Tensor, y: torch.Tensor, phase: str) -> float:
        self._update_gate(x)
        logits = self._combined_logits(x)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        self.opt_a.zero_grad()
        self.opt_n.zero_grad()
        loss.backward()

        n_scale = 1.0
        if self.gated and phase == "task2":
            n_scale = self.rho
        for p in self.n.parameters():
            if p.grad is not None:
                p.grad.mul_(n_scale)

        self.opt_a.step()
        self.opt_n.step()

        with torch.no_grad():
            pull = (
                self.retain_pull
                if not self.gated or phase == "task1"
                else self.retain_pull * self.rho
            )
            for pn, pa in zip(self.n.parameters(), self.a.parameters()):
                pn.add_(pull * (pa - pn))
        return loss.item()

    def probe(
        self,
        x_task1: torch.Tensor,
        y_task1: torch.Tensor,
        x_task2: torch.Tensor,
        y_task2: torch.Tensor,
    ) -> dict[str, float]:
        with torch.no_grad():
            return {
                "combined_task1": accuracy_from_logits(
                    self._combined_logits(x_task1), y_task1
                ),
                "combined_task2": accuracy_from_logits(
                    self._combined_logits(x_task2), y_task2
                ),
                "a_channel_task1": accuracy_from_logits(self.a(x_task1), y_task1),
                "a_channel_task2": accuracy_from_logits(self.a(x_task2), y_task2),
                "n_channel_task1": accuracy_from_logits(self.n(x_task1), y_task1),
                "n_channel_task2": accuracy_from_logits(self.n(x_task2), y_task2),
                "rho": self.rho,
                "persistence": self.persistence,
                "a_n_distance": param_distance(self.a, self.n),
            }


def model_rows(
    seed: int,
    step: int,
    phase: str,
    model_name: str,
    metrics: dict[str, float],
) -> dict[str, object]:
    return {
        "seed": seed,
        "step": step,
        "phase": phase,
        "model": model_name,
        **metrics,
    }


def run_seed(args: argparse.Namespace, seed: int) -> list[dict[str, object]]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1000)
    x_task1_probe, y_task1_probe = make_probe("task1", args.probe_size, seed + 2000)
    x_task2_probe, y_task2_probe = make_probe("task2", args.probe_size, seed + 3000)

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
            retain_pull=args.retain_pull,
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
            retain_pull=args.retain_pull,
        ),
    }

    rows: list[dict[str, object]] = []
    total_steps = args.task1_steps + args.task2_steps
    for step in range(total_steps + 1):
        if step % args.probe_every == 0 or step in (args.task1_steps, total_steps):
            phase = "task1" if step <= args.task1_steps else "task2"
            if step == args.task1_steps:
                phase = "end_task1"
            if step == total_steps:
                phase = "end_task2"
            for name, model in models.items():
                rows.append(
                    model_rows(
                        seed,
                        step,
                        phase,
                        name,
                        model.probe(
                            x_task1_probe,
                            y_task1_probe,
                            x_task2_probe,
                            y_task2_probe,
                        ),
                    )
                )
        if step == total_steps:
            break

        task = "task1" if step < args.task1_steps else "task2"
        x, y = sample_batch(task, args.batch_size, generator, args.label_noise)
        for model in models.values():
            model.step(x, y, task)
    return rows


def summarize(
    rows: list[dict[str, object]], task1_steps: int, task2_steps: int
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    total_steps = task1_steps + task2_steps
    for model in ["MLP", "ChevronSlow", "IDLGated"]:
        end_task1 = [
            r for r in rows if r["model"] == model and int(r["step"]) == task1_steps
        ]
        end_task2 = [
            r for r in rows if r["model"] == model and int(r["step"]) == total_steps
        ]
        for label, group in [("end_task1", end_task1), ("end_task2", end_task2)]:
            n = len(group)
            row: dict[str, object] = {"model": model, "checkpoint": label, "n": n}
            for key in [
                "combined_task1",
                "combined_task2",
                "a_channel_task1",
                "a_channel_task2",
                "n_channel_task1",
                "n_channel_task2",
                "rho",
                "persistence",
                "a_n_distance",
            ]:
                row[key] = sum(float(r[key]) for r in group) / n
            summary.append(row)

        before = summary[-2]
        after = summary[-1]
        retained_key = "combined_task1" if model == "MLP" else "n_channel_task1"
        summary.append(
            {
                "model": model,
                "checkpoint": "forgetting",
                "n": len(end_task2),
                "combined_task1": float(before["combined_task1"])
                - float(after["combined_task1"]),
                "combined_task2": float(after["combined_task2"]),
                "a_channel_task1": float(before["a_channel_task1"])
                - float(after["a_channel_task1"]),
                "a_channel_task2": float(after["a_channel_task2"]),
                "n_channel_task1": float(before["n_channel_task1"])
                - float(after["n_channel_task1"]),
                "n_channel_task2": float(after["n_channel_task2"]),
                "rho": float(after["rho"]),
                "persistence": float(after["persistence"]),
                "a_n_distance": float(after["a_n_distance"]),
                "retained_task1_accuracy": float(after[retained_key]),
            }
        )
    return summary


def write_csv(path: str, rows: list[dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary: list[dict[str, object]]) -> None:
    print("Catastrophic forgetting experiment")
    print("model          checkpoint  comb_1 comb_2 A_1   A_2   N_1   N_2   rho   P")
    for row in summary:
        if row["checkpoint"] == "forgetting":
            continue
        print(
            f"{str(row['model']):<14} {str(row['checkpoint']):<10} "
            f"{float(row['combined_task1']):.3f}  {float(row['combined_task2']):.3f}  "
            f"{float(row['a_channel_task1']):.3f} {float(row['a_channel_task2']):.3f} "
            f"{float(row['n_channel_task1']):.3f} {float(row['n_channel_task2']):.3f} "
            f"{float(row['rho']):.3f} {float(row['persistence']):.3f}"
        )
    print("\nForgetting summary")
    print("model          retained_1  combined_1_drop  N_1_drop  after_2")
    for row in summary:
        if row["checkpoint"] != "forgetting":
            continue
        retained = row.get("retained_task1_accuracy", row["combined_task1"])
        print(
            f"{str(row['model']):<14} {float(retained):.3f}       "
            f"{float(row['combined_task1']):.3f}            "
            f"{float(row['n_channel_task1']):.3f}     "
            f"{float(row['combined_task2']):.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task1-steps", type=int, default=2200)
    parser.add_argument("--task2-steps", type=int, default=220)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-size", type=int, default=2048)
    parser.add_argument("--probe-every", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=20)
    parser.add_argument("--label-noise", type=float, default=0.02)
    parser.add_argument("--lr-mlp", type=float, default=0.003)
    parser.add_argument("--lr-a", type=float, default=0.004)
    parser.add_argument("--lr-n", type=float, default=0.0012)
    parser.add_argument("--blend-n", type=float, default=0.25)
    parser.add_argument("--fast-beta", type=float, default=0.96)
    parser.add_argument("--baseline-beta", type=float, default=0.999)
    parser.add_argument("--persistence-beta", type=float, default=0.994)
    parser.add_argument("--theta", type=float, default=0.36)
    parser.add_argument("--sharpness", type=float, default=22.0)
    parser.add_argument("--retain-pull", type=float, default=0.003)
    parser.add_argument("--out-dir", default="runs_catastrophic_forgetting")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, object]] = []
    for seed in range(args.seeds):
        all_rows.extend(run_seed(args, seed))
    summary = summarize(all_rows, args.task1_steps, args.task2_steps)
    write_csv(os.path.join(args.out_dir, "probes.csv"), all_rows)
    write_csv(os.path.join(args.out_dir, "summary.csv"), summary)
    print_summary(summary)
    print(f"\nWrote {args.out_dir}/probes.csv and {args.out_dir}/summary.csv")


if __name__ == "__main__":
    main()
