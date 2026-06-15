#!/usr/bin/env python3
"""PyTorch experiment: recovery from retained N after Task 2.

This fourth test asks whether the retained Task 1 structure in N has external
value. The sequence is:
  1. Train on Task 1.
  2. Train on Task 2 without replay.
  3. Return to Task 1 and measure recovery speed.

The diagnostic is whether IDLGated regains Task 1 performance faster because N
retained more Task 1 structure during Task 2.

This file includes an explicit N-to-A recovery constraint during Task 1 phases.
That tests whether retained structure can steer fast adaptation when an old
regime returns.

Run:
    .venv-torch/bin/python task1_recovery_experiment.py

Outputs:
    runs_task1_recovery/probes.csv
    runs_task1_recovery/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os

import torch
from torch import nn
from torch.nn import functional as F


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
        constraint_pull: float,
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
        self.constraint_pull = constraint_pull
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
                self.persistence_beta * self.persistence
                + (1.0 - self.persistence_beta) * excess
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
            if phase == "task1":
                for pa, pn in zip(self.a.parameters(), self.n.parameters()):
                    pa.add_(self.constraint_pull * (pn - pa))
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


def phase_for_step(args: argparse.Namespace, step: int) -> str:
    if step < args.task1_steps:
        return "task1_initial"
    if step < args.task1_steps + args.task2_steps:
        return "task2"
    return "task1_return"


def task_for_phase(phase: str) -> str:
    return "task2" if phase == "task2" else "task1"


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
            constraint_pull=args.constraint_pull,
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
            constraint_pull=args.constraint_pull,
        ),
    }

    rows: list[dict[str, object]] = []
    total_steps = args.task1_steps + args.task2_steps + args.return_steps
    checkpoints = {
        args.task1_steps: "end_task1",
        args.task1_steps + args.task2_steps: "end_task2",
        total_steps: "end_return",
    }

    for step in range(total_steps + 1):
        phase = checkpoints.get(step, phase_for_step(args, step))
        if step % args.probe_every == 0 or step in checkpoints:
            return_step = max(0, step - args.task1_steps - args.task2_steps)
            for name, model in models.items():
                rows.append(
                    {
                        "seed": seed,
                        "step": step,
                        "return_step": return_step,
                        "phase": phase,
                        "model": name,
                        **model.probe(
                            x_task1_probe,
                            y_task1_probe,
                            x_task2_probe,
                            y_task2_probe,
                        ),
                    }
                )
        if step == total_steps:
            break

        train_phase = phase_for_step(args, step)
        train_task = task_for_phase(train_phase)
        x, y = sample_batch(train_task, args.batch_size, generator, args.label_noise)
        for model in models.values():
            model.step(x, y, train_task)
    return rows


def mean(rows: list[dict[str, object]], key: str) -> float:
    return sum(float(r[key]) for r in rows) / len(rows)


def first_recovery_step(
    rows: list[dict[str, object]],
    model: str,
    seed: int,
    target: float,
) -> int | None:
    candidates = [
        r
        for r in rows
        if r["model"] == model
        and int(r["seed"]) == seed
        and str(r["phase"]) in {"task1_return", "end_return"}
        and int(r["return_step"]) >= 0
    ]
    candidates.sort(key=lambda r: int(r["return_step"]))
    for row in candidates:
        if float(row["combined_task1"]) >= target:
            return int(row["return_step"])
    return None


def summarize(
    rows: list[dict[str, object]], args: argparse.Namespace
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    seeds = sorted({int(r["seed"]) for r in rows})
    end_task1_step = args.task1_steps
    end_task2_step = args.task1_steps + args.task2_steps
    end_return_step = end_task2_step + args.return_steps

    for model in ["MLP", "ChevronSlow", "IDLGated"]:
        end_task1 = [
            r for r in rows if r["model"] == model and int(r["step"]) == end_task1_step
        ]
        end_task2 = [
            r for r in rows if r["model"] == model and int(r["step"]) == end_task2_step
        ]
        end_return = [
            r for r in rows if r["model"] == model and int(r["step"]) == end_return_step
        ]
        baseline_task1 = mean(end_task1, "combined_task1")
        target_95 = 0.95 * baseline_task1
        target_98 = 0.98 * baseline_task1
        target_abs_95 = 0.95

        recovery_95 = []
        recovery_98 = []
        recovery_abs_95 = []
        for seed in seeds:
            step_95 = first_recovery_step(rows, model, seed, target_95)
            step_98 = first_recovery_step(rows, model, seed, target_98)
            step_abs_95 = first_recovery_step(rows, model, seed, target_abs_95)
            recovery_95.append(args.return_steps if step_95 is None else step_95)
            recovery_98.append(args.return_steps if step_98 is None else step_98)
            recovery_abs_95.append(
                args.return_steps if step_abs_95 is None else step_abs_95
            )

        summary.append(
            {
                "model": model,
                "baseline_task1": baseline_task1,
                "after_task2_combined_task1": mean(end_task2, "combined_task1"),
                "after_task2_combined_task2": mean(end_task2, "combined_task2"),
                "after_task2_n_task1": mean(end_task2, "n_channel_task1"),
                "after_return_combined_task1": mean(end_return, "combined_task1"),
                "after_return_combined_task2": mean(end_return, "combined_task2"),
                "mean_steps_to_95pct_baseline": sum(recovery_95) / len(recovery_95),
                "mean_steps_to_98pct_baseline": sum(recovery_98) / len(recovery_98),
                "mean_steps_to_abs_0_95": sum(recovery_abs_95)
                / len(recovery_abs_95),
                "rho_after_task2": mean(end_task2, "rho"),
                "rho_after_return": mean(end_return, "rho"),
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
    print("Task 1 recovery experiment")
    print("model          base1 after2_1 after2_2 N_after2_1 return1 steps95 steps98")
    for row in summary:
        print(
            f"{str(row['model']):<14} "
            f"{float(row['baseline_task1']):.3f}  "
            f"{float(row['after_task2_combined_task1']):.3f}    "
            f"{float(row['after_task2_combined_task2']):.3f}    "
            f"{float(row['after_task2_n_task1']):.3f}      "
            f"{float(row['after_return_combined_task1']):.3f}   "
            f"{float(row['mean_steps_to_95pct_baseline']):.1f}   "
            f"{float(row['mean_steps_to_98pct_baseline']):.1f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task1-steps", type=int, default=2200)
    parser.add_argument("--task2-steps", type=int, default=220)
    parser.add_argument("--return-steps", type=int, default=260)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--probe-size", type=int, default=2048)
    parser.add_argument("--probe-every", type=int, default=10)
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
    parser.add_argument("--constraint-pull", type=float, default=0.05)
    parser.add_argument("--out-dir", default="runs_task1_recovery")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, object]] = []
    for seed in range(args.seeds):
        all_rows.extend(run_seed(args, seed))
    summary = summarize(all_rows, args)
    write_csv(os.path.join(args.out_dir, "probes.csv"), all_rows)
    write_csv(os.path.join(args.out_dir, "summary.csv"), summary)
    print_summary(summary)
    print(f"\nWrote {args.out_dir}/probes.csv and {args.out_dir}/summary.csv")


if __name__ == "__main__":
    main()
