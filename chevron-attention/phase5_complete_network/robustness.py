"""Robustness sweeps for the Phase 5 Chevron memory network."""

from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import replace
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from phase5_complete_network.experiment import (
    CycleConfig,
    IDLConfig,
    RecallAndControlTask,
    SignalNoise,
    TaskConfig,
    TrainConfig,
    cycle_metrics,
    evaluate,
    run_cycle,
    select_device,
    train_model,
)


Metric = Dict[str, float]


def stats(values: Iterable[float]) -> Tuple[float, float, float, float]:
    items = list(values)
    return (
        statistics.mean(items),
        statistics.stdev(items) if len(items) > 1 else 0.0,
        min(items),
        max(items),
    )


def success_rate(values: Iterable[float], threshold: float = 0.5) -> float:
    items = list(values)
    return sum(value > threshold for value in items) / len(items)


def cycle_runs(
    models: Sequence[torch.nn.Module],
    task: RecallAndControlTask,
    stream_seeds: Sequence[int],
    cycle: CycleConfig,
    idl: IDLConfig,
    device: torch.device,
    noise: SignalNoise = SignalNoise(),
    method: str = "integrated_idl",
) -> List[Metric]:
    return [
        cycle_metrics(run_cycle(method, model, task, cycle, idl, seed, device, noise))
        for model in models
        for seed in stream_seeds
    ]


def print_size_sweep(
    models: Sequence[torch.nn.Module],
    base: TaskConfig,
    device: torch.device,
) -> None:
    print("\n## Unseen memory/control-set sizes")
    print("facts controls answer_mean answer_min retrieval_min context_min")
    settings = ((4, 1), (6, 4), (8, 4), (8, 8), (10, 8), (12, 8))
    for facts, controls in settings:
        eval_task = RecallAndControlTask(
            replace(base, num_facts=facts, max_controls=controls)
        )
        metrics = [
            evaluate(model, eval_task, random.Random(80_000 + index), device)
            for index, model in enumerate(models)
        ]
        answer = stats(item["answer_accuracy"] for item in metrics)
        retrieval = stats(item["retrieval_accuracy"] for item in metrics)
        context = stats(item["context_accuracy"] for item in metrics)
        print(
            "%5d %8d %11.4f %10.4f %13.4f %11.4f"
            % (facts, controls, answer[0], answer[2], retrieval[2], context[2])
        )


def print_duration_sweep(
    models: Sequence[torch.nn.Module],
    task: RecallAndControlTask,
    streams: Sequence[int],
    device: torch.device,
) -> None:
    print("\n## Duration boundary")
    print("short_steps idl_N idl_preserve always_N always_preserve")
    for duration in (2, 5, 10, 15, 20, 30, 40):
        cycle = replace(CycleConfig(), short_revoke=duration)
        integrated = cycle_runs(models, task, streams, cycle, IDLConfig(), device)
        always = cycle_runs(
            models, task, streams, cycle, IDLConfig(), device, method="always_update"
        )
        print(
            "%11d %.4f %.3f %.4f %.3f"
            % (
                duration,
                stats(m["retained_after_short"] for m in integrated)[0],
                success_rate(m["short_probe_preserve"] for m in integrated),
                stats(m["retained_after_short"] for m in always)[0],
                success_rate(m["short_probe_preserve"] for m in always),
            )
        )

    print("\nlong_steps idl_N idl_consolidate fixed_N fixed_consolidate")
    for duration in (20, 30, 40, 50, 60, 70, 90, 120):
        cycle = replace(CycleConfig(), long_revoke=duration, long_restore=duration)
        integrated = cycle_runs(models, task, streams, cycle, IDLConfig(), device)
        fixed = cycle_runs(
            models, task, streams, cycle, IDLConfig(), device, method="fixed_slow"
        )
        print(
            "%10d %.4f %.3f %.4f %.3f"
            % (
                duration,
                stats(m["retained_after_long_revoke"] for m in integrated)[0],
                success_rate(m["long_probe_consolidate"] for m in integrated),
                stats(m["retained_after_long_revoke"] for m in fixed)[0],
                success_rate(m["long_probe_consolidate"] for m in fixed),
            )
        )


def print_noise_sweep(
    models: Sequence[torch.nn.Module],
    task: RecallAndControlTask,
    streams: Sequence[int],
    device: torch.device,
) -> None:
    print("\n## Imperfect retention signal")
    print("noise_kind level answer_min short_success revoke_success full_cycle")
    settings = (
        ("gaussian", 0.05, SignalNoise(gaussian_std=0.05)),
        ("gaussian", 0.10, SignalNoise(gaussian_std=0.10)),
        ("gaussian", 0.20, SignalNoise(gaussian_std=0.20)),
        ("gaussian", 0.35, SignalNoise(gaussian_std=0.35)),
        ("dropout", 0.10, SignalNoise(dropout_probability=0.10)),
        ("dropout", 0.25, SignalNoise(dropout_probability=0.25)),
        ("dropout", 0.50, SignalNoise(dropout_probability=0.50)),
        ("flip", 0.01, SignalNoise(flip_probability=0.01)),
        ("flip", 0.05, SignalNoise(flip_probability=0.05)),
        ("flip", 0.10, SignalNoise(flip_probability=0.10)),
        ("flip", 0.20, SignalNoise(flip_probability=0.20)),
    )
    for name, level, noise in settings:
        metrics = cycle_runs(
            models, task, streams, CycleConfig(), IDLConfig(), device, noise=noise
        )
        print(
            "%-10s %5.2f %10.4f %13.3f %14.3f %15.3f"
            % (
                name,
                level,
                stats(m["answer_accuracy"] for m in metrics)[2],
                success_rate(m["short_probe_preserve"] for m in metrics),
                success_rate(m["long_probe_consolidate"] for m in metrics),
                success_rate(m["full_revoke_restore_cycle"] for m in metrics),
            )
        )


def print_hyperparameter_sweep(
    models: Sequence[torch.nn.Module],
    task: RecallAndControlTask,
    streams: Sequence[int],
    device: torch.device,
) -> None:
    print("\n## IDL beta/threshold sensitivity")
    print("Each cell is short-preserve / revoke-consolidate success rate")
    thresholds = (0.20, 0.30, 0.35, 0.40, 0.50)
    print("beta\\theta " + " ".join("%9.2f" % value for value in thresholds))
    for beta in (0.970, 0.980, 0.985, 0.990, 0.995):
        cells = []
        for threshold in thresholds:
            metrics = cycle_runs(
                models,
                task,
                streams,
                CycleConfig(),
                replace(IDLConfig(), beta=beta, threshold=threshold),
                device,
            )
            preserve = success_rate(m["short_probe_preserve"] for m in metrics)
            consolidate = success_rate(m["long_probe_consolidate"] for m in metrics)
            cells.append("%.2f/%.2f" % (preserve, consolidate))
        print("%10.3f " % beta + " ".join("%9s" % cell for cell in cells))

    print("\neta_n short_success revoke_success full_cycle")
    for eta in (0.002, 0.005, 0.010, 0.020, 0.040, 0.080, 0.120, 0.160):
        metrics = cycle_runs(
            models,
            task,
            streams,
            CycleConfig(),
            replace(IDLConfig(), eta_n=eta),
            device,
        )
        print(
            "%5.3f %13.3f %14.3f %15.3f"
            % (
                eta,
                success_rate(m["short_probe_preserve"] for m in metrics),
                success_rate(m["long_probe_consolidate"] for m in metrics),
                success_rate(m["full_revoke_restore_cycle"] for m in metrics),
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[1, 7, 13])
    parser.add_argument("--stream-seeds", type=int, nargs="+", default=[101, 211, 307, 401, 503])
    parser.add_argument("--steps", type=int, default=700)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    task_config = TaskConfig()
    task = RecallAndControlTask(task_config)
    models = []
    for index, seed in enumerate(args.model_seeds, 1):
        model = train_model(task, TrainConfig(steps=args.steps), seed, device)
        models.append(model)
        print("trained model %d/%d seed=%d" % (index, len(args.model_seeds), seed))
    print_size_sweep(models, task_config, device)
    print_duration_sweep(models, task, args.stream_seeds, device)
    print_noise_sweep(models, task, args.stream_seeds, device)
    print_hyperparameter_sweep(models, task, args.stream_seeds, device)


if __name__ == "__main__":
    main()
