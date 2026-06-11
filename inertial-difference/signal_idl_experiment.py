#!/usr/bin/env python3
"""Small pure-Python signal detect experiment for Inertial Difference Learning.

The task is an online binary classification stream with:
  - a long stable concept,
  - short contradictory bursts,
  - a persistent concept shift.

The comparison is:
  - MLP: one ordinary online MLP.
  - ChevronSlow: fast A network plus slow N network updated every example.
  - IDLGated: fast A network plus N network retained only when A-N difference
    persists.

No third-party packages are required.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass
from typing import Iterable


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def tanh(x: float) -> float:
    return math.tanh(x)


def bce(pred: float, y: int) -> float:
    pred = min(max(pred, 1e-7), 1.0 - 1e-7)
    return -(y * math.log(pred) + (1 - y) * math.log(1.0 - pred))


@dataclass
class Example:
    t: int
    x: list[float]
    y: int
    phase: str


def concept_label(x0: float, x1: float, concept: int) -> int:
    if concept == 0:
        score = x1 + 0.9 * x0 - 0.65 * x0 * x0 + 0.15
    elif concept == 1:
        score = -0.85 * x0 + x1 * x1 - 0.2 * x1 - 0.08
    else:
        score = 0.55 * x0 - 0.85 * x1 + 0.55 * x0 * x1
    return 1 if score > 0.0 else 0


def stream(
    steps: int,
    rng: random.Random,
    shift_at: int,
    burst_period: int,
    burst_len: int,
    label_noise: float,
) -> Iterable[Example]:
    for t in range(steps):
        x0 = rng.uniform(-1.0, 1.0)
        x1 = rng.uniform(-1.0, 1.0)
        in_burst = t < shift_at and (t % burst_period) < burst_len and t > 0
        if t >= shift_at:
            concept = 1
            phase = "post_shift"
        elif in_burst:
            concept = 2
            phase = "transient_burst"
        else:
            concept = 0
            phase = "pre_shift"
        y = concept_label(x0, x1, concept)
        if rng.random() < label_noise:
            y = 1 - y
        yield Example(t=t, x=[x0, x1], y=y, phase=phase)


class MLPParams:
    def __init__(self, input_dim: int, hidden_dim: int, rng: random.Random):
        scale = 0.45
        self.w1 = [
            [rng.uniform(-scale, scale) for _ in range(input_dim)]
            for _ in range(hidden_dim)
        ]
        self.b1 = [0.0 for _ in range(hidden_dim)]
        self.w2 = [rng.uniform(-scale, scale) for _ in range(hidden_dim)]
        self.b2 = 0.0

    def forward(self, x: list[float]) -> tuple[list[float], float, float]:
        h = []
        for row, bias in zip(self.w1, self.b1):
            h.append(tanh(sum(w * xi for w, xi in zip(row, x)) + bias))
        logit = sum(w * hi for w, hi in zip(self.w2, h)) + self.b2
        return h, logit, sigmoid(logit)

    def apply_grad(
        self,
        x: list[float],
        h: list[float],
        dlogit: float,
        lr: float,
        weight_decay: float,
    ) -> None:
        old_w2 = self.w2[:]
        for j, hj in enumerate(h):
            self.w2[j] -= lr * (dlogit * hj + weight_decay * self.w2[j])
        self.b2 -= lr * dlogit
        for j, hj in enumerate(h):
            dh_pre = dlogit * old_w2[j] * (1.0 - hj * hj)
            for i, xi in enumerate(x):
                self.w1[j][i] -= lr * (dh_pre * xi + weight_decay * self.w1[j][i])
            self.b1[j] -= lr * dh_pre

    def move_toward(self, other: "MLPParams", amount: float) -> None:
        for j in range(len(self.w1)):
            for i in range(len(self.w1[j])):
                self.w1[j][i] += amount * (other.w1[j][i] - self.w1[j][i])
            self.b1[j] += amount * (other.b1[j] - self.b1[j])
            self.w2[j] += amount * (other.w2[j] - self.w2[j])
        self.b2 += amount * (other.b2 - self.b2)

    def param_rms_difference(self, other: "MLPParams") -> float:
        total = 0.0
        count = 0
        for j in range(len(self.w1)):
            for i in range(len(self.w1[j])):
                total += (self.w1[j][i] - other.w1[j][i]) ** 2
                count += 1
            total += (self.b1[j] - other.b1[j]) ** 2
            total += (self.w2[j] - other.w2[j]) ** 2
            count += 2
        total += (self.b2 - other.b2) ** 2
        count += 1
        return math.sqrt(total / count)


class OnlineMLP:
    def __init__(self, input_dim: int, hidden_dim: int, rng: random.Random):
        self.params = MLPParams(input_dim, hidden_dim, rng)

    def step(self, x: list[float], y: int) -> dict[str, float]:
        h, _logit, pred = self.params.forward(x)
        loss = bce(pred, y)
        self.params.apply_grad(x, h, pred - y, lr=0.035, weight_decay=1e-5)
        return {"pred": pred, "loss": loss, "rho": 1.0, "diff": 0.0, "persistence": 0.0}


class ChevronModel:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        rng: random.Random,
        gated: bool,
        blend_n: float = 0.36,
    ):
        self.a = MLPParams(input_dim, hidden_dim, rng)
        self.n = MLPParams(input_dim, hidden_dim, rng)
        self.n.move_toward(self.a, 1.0)
        self.gated = gated
        self.blend_n = blend_n
        self.persistence = 0.0

    def step(self, x: list[float], y: int) -> dict[str, float]:
        h_a, logit_a, _pred_a = self.a.forward(x)
        h_n, logit_n, _pred_n = self.n.forward(x)
        logit = (1.0 - self.blend_n) * logit_a + self.blend_n * logit_n
        pred = sigmoid(logit)
        loss = bce(pred, y)
        dlogit = pred - y

        activation_diff = abs(logit_a - logit_n) + sum(
            abs(a - n) for a, n in zip(h_a, h_n)
        ) / len(h_a)
        param_diff = self.a.param_rms_difference(self.n)
        diff = 0.5 * activation_diff + 0.5 * param_diff
        self.persistence = 0.975 * self.persistence + 0.025 * diff

        self.a.apply_grad(
            x,
            h_a,
            dlogit * (1.0 - self.blend_n),
            lr=0.055,
            weight_decay=1e-5,
        )

        if self.gated:
            rho = sigmoid(22.0 * (self.persistence - 0.115))
            self.n.apply_grad(
                x,
                h_n,
                dlogit * self.blend_n,
                lr=0.012 * rho,
                weight_decay=1e-5,
            )
            self.n.move_toward(self.a, 0.018 * rho)
        else:
            rho = 1.0
            self.n.apply_grad(
                x,
                h_n,
                dlogit * self.blend_n,
                lr=0.009,
                weight_decay=1e-5,
            )
            self.n.move_toward(self.a, 0.006)

        return {
            "pred": pred,
            "loss": loss,
            "rho": rho,
            "diff": diff,
            "persistence": self.persistence,
        }


def window_name(t: int, shift_at: int, burst_len: int, burst_period: int) -> str:
    if shift_at - 400 <= t < shift_at:
        return "pre_shift_last_400"
    if shift_at <= t < shift_at + 200:
        return "shift_adaptation_200"
    if shift_at + 200 <= t < shift_at + 800:
        return "post_shift_200_800"
    if t < shift_at and (t % burst_period) < burst_len and t > 0:
        return "transient_bursts"
    return "other"


def run_seed(args: argparse.Namespace, seed: int) -> list[dict[str, object]]:
    rng_data = random.Random(seed)
    models = {
        "MLP": OnlineMLP(args.input_dim, args.hidden_dim, random.Random(seed + 10_000)),
        "ChevronSlow": ChevronModel(
            args.input_dim, args.hidden_dim, random.Random(seed + 20_000), gated=False
        ),
        "IDLGated": ChevronModel(
            args.input_dim, args.hidden_dim, random.Random(seed + 30_000), gated=True
        ),
    }
    rows: list[dict[str, object]] = []
    examples = list(
        stream(
            args.steps,
            rng_data,
            args.shift_at,
            args.burst_period,
            args.burst_len,
            args.label_noise,
        )
    )
    for ex in examples:
        for name, model in models.items():
            metrics = model.step(ex.x, ex.y)
            pred_label = 1 if metrics["pred"] >= 0.5 else 0
            rows.append(
                {
                    "seed": seed,
                    "t": ex.t,
                    "model": name,
                    "phase": ex.phase,
                    "window": window_name(
                        ex.t, args.shift_at, args.burst_len, args.burst_period
                    ),
                    "y": ex.y,
                    "pred": metrics["pred"],
                    "correct": 1 if pred_label == ex.y else 0,
                    "loss": metrics["loss"],
                    "rho": metrics["rho"],
                    "diff": metrics["diff"],
                    "persistence": metrics["persistence"],
                }
            )
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        window = str(row["window"])
        if window == "other":
            continue
        grouped.setdefault((str(row["model"]), window), []).append(row)

    summary = []
    for (model, window), group in sorted(grouped.items()):
        n = len(group)
        summary.append(
            {
                "model": model,
                "window": window,
                "n": n,
                "accuracy": sum(int(r["correct"]) for r in group) / n,
                "loss": sum(float(r["loss"]) for r in group) / n,
                "rho": sum(float(r["rho"]) for r in group) / n,
                "diff": sum(float(r["diff"]) for r in group) / n,
                "persistence": sum(float(r["persistence"]) for r in group) / n,
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
        "transient_bursts",
        "pre_shift_last_400",
        "shift_adaptation_200",
        "post_shift_200_800",
    ]
    models = ["MLP", "ChevronSlow", "IDLGated"]
    print("Accuracy by diagnostic window")
    print("model          window                 acc    loss   rho    diff   P")
    for model in models:
        for window in windows:
            row = next(
                r for r in summary if r["model"] == model and r["window"] == window
            )
            print(
                f"{model:<14} {window:<22} "
                f"{row['accuracy']:.3f}  {row['loss']:.3f}  {row['rho']:.3f}  "
                f"{row['diff']:.3f}  {row['persistence']:.3f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4200)
    parser.add_argument("--shift-at", type=int, default=2600)
    parser.add_argument("--burst-period", type=int, default=520)
    parser.add_argument("--burst-len", type=int, default=28)
    parser.add_argument("--label-noise", type=float, default=0.02)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--input-dim", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=14)
    parser.add_argument("--out-dir", default="runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
