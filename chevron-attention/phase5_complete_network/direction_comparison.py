"""Compare directional persistence rules under clean and corrupted signals."""

from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import replace
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from phase5_complete_network.experiment import (
    CycleConfig,
    CycleRecord,
    IDLConfig,
    RecallAndControlTask,
    SignalNoise,
    TaskConfig,
    TrainConfig,
    cycle_metrics,
    run_cycle,
    select_device,
    train_model,
)


MODES = ("hard_reset", "two_trace", "signed_hysteresis")


def mean(values: Iterable[float]) -> float:
    return statistics.mean(list(values))


def rate(values: Iterable[float]) -> float:
    items = list(values)
    return sum(value > 0.5 for value in items) / len(items)


def crossing_step(records: Sequence[CycleRecord], phase: str, upward: bool) -> int:
    phase_records = [record for record in records if record.phase == phase]
    for step, record in enumerate(phase_records, 1):
        crossed = record.retained_target >= 0.5 if upward else record.retained_target < 0.5
        if crossed:
            return step
    return len(phase_records) + 1


def runs(
    models: Sequence[torch.nn.Module],
    task: RecallAndControlTask,
    streams: Sequence[int],
    mode: str,
    device: torch.device,
    noise: SignalNoise = SignalNoise(),
    cycle: CycleConfig = CycleConfig(),
) -> List[Tuple[Dict[str, float], int, int]]:
    results = []
    for model in models:
        for seed in streams:
            records = run_cycle(
                "integrated_idl",
                model,
                task,
                cycle,
                replace(IDLConfig(), persistence_mode=mode),
                seed,
                device,
                noise,
            )
            metrics = cycle_metrics(records)
            restore_step = (
                crossing_step(records, "long_restore", upward=True)
                if metrics["long_probe_consolidate"] > 0.5
                else cycle.long_restore + 1
            )
            results.append(
                (
                    metrics,
                    crossing_step(records, "long_revoke", upward=False),
                    restore_step,
                )
            )
    return results


def print_clean_comparison(
    models: Sequence[torch.nn.Module],
    task: RecallAndControlTask,
    streams: Sequence[int],
    device: torch.device,
) -> None:
    print("\n## Clean reversal and latency")
    print("mode short revoke full_cycle revoke_step restore_step N_revoke N_restore")
    for mode in MODES:
        results = runs(models, task, streams, mode, device)
        metrics = [result[0] for result in results]
        print(
            "%-17s %.3f %.3f %.3f %11.1f %12.1f %.4f %.4f"
            % (
                mode,
                rate(m["short_probe_preserve"] for m in metrics),
                rate(m["long_probe_consolidate"] for m in metrics),
                rate(m["full_revoke_restore_cycle"] for m in metrics),
                mean(result[1] for result in results),
                mean(result[2] for result in results),
                mean(m["retained_after_long_revoke"] for m in metrics),
                mean(m["retained_after_long_restore"] for m in metrics),
            )
        )


def print_flip_comparison(
    models: Sequence[torch.nn.Module],
    task: RecallAndControlTask,
    streams: Sequence[int],
    device: torch.device,
) -> None:
    print("\n## Random sign-flip corruption")
    print("mode flip short revoke full_cycle answer_min revoke_step restore_step")
    for probability in (0.01, 0.05, 0.10, 0.20, 0.30):
        for mode in MODES:
            results = runs(
                models,
                task,
                streams,
                mode,
                device,
                SignalNoise(flip_probability=probability),
            )
            metrics = [result[0] for result in results]
            print(
                "%-17s %.2f %.3f %.3f %.3f %10.4f %11.1f %12.1f"
                % (
                    mode,
                    probability,
                    rate(m["short_probe_preserve"] for m in metrics),
                    rate(m["long_probe_consolidate"] for m in metrics),
                    rate(m["full_revoke_restore_cycle"] for m in metrics),
                    min(m["answer_accuracy"] for m in metrics),
                    mean(result[1] for result in results),
                    mean(result[2] for result in results),
                )
            )


def print_duration_comparison(
    models: Sequence[torch.nn.Module],
    task: RecallAndControlTask,
    streams: Sequence[int],
    device: torch.device,
) -> None:
    print("\n## Clean duration boundary by rule")
    print("mode long_steps revoke_success full_cycle N_revoke N_restore")
    for duration in (20, 30, 40, 50, 70, 90):
        cycle = replace(CycleConfig(), long_revoke=duration, long_restore=duration)
        for mode in MODES:
            results = runs(models, task, streams, mode, device, cycle=cycle)
            metrics = [result[0] for result in results]
            print(
                "%-17s %10d %14.3f %15.3f %.4f %.4f"
                % (
                    mode,
                    duration,
                    rate(m["long_probe_consolidate"] for m in metrics),
                    rate(m["full_revoke_restore_cycle"] for m in metrics),
                    mean(m["retained_after_long_revoke"] for m in metrics),
                    mean(m["retained_after_long_restore"] for m in metrics),
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
    task = RecallAndControlTask(TaskConfig())
    models = []
    for index, seed in enumerate(args.model_seeds, 1):
        models.append(train_model(task, TrainConfig(steps=args.steps), seed, device))
        print("trained model %d/%d seed=%d" % (index, len(args.model_seeds), seed))
    print_clean_comparison(models, task, args.stream_seeds, device)
    print_flip_comparison(models, task, args.stream_seeds, device)
    print_duration_comparison(models, task, args.stream_seeds, device)


if __name__ == "__main__":
    main()
