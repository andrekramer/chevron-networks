from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader


@dataclass
class EvalResult:
    metrics: dict[str, float]
    predictions: dict[str, torch.Tensor]


def _mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _recovery_times(
    correct: torch.Tensor,
    event_mask: torch.Tensor,
    *,
    threshold: float,
    window: int,
    max_horizon: int,
) -> list[float]:
    times: list[float] = []
    event_indices = event_mask.nonzero(as_tuple=False).flatten().tolist()
    for idx in event_indices:
        recovered = None
        end_limit = min(len(correct), idx + max_horizon + window)
        for start in range(idx, end_limit - window + 1):
            acc = correct[start : start + window].float().mean().item()
            if acc >= threshold:
                recovered = start - idx
                break
        if recovered is not None:
            times.append(float(recovered))
    return times


def evaluate(
    model: torch.nn.Module,
    dataset,
    *,
    batch_size: int,
    device: torch.device,
    recovery_threshold: float = 0.8,
    recovery_window: int = 16,
    max_recovery_horizon: int = 100,
) -> EvalResult:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows: dict[str, list[torch.Tensor]] = defaultdict(list)

    with torch.no_grad():
        for batch in loader:
            x = batch["context"].to(device)
            y = batch["target"].to(device)
            output = model(x)
            pred = output.logits.argmax(dim=-1)
            rows["correct"].append((pred == y).cpu())
            rows["pred"].append(pred.cpu())
            rows["target"].append(batch["target"].cpu())
            for key in ("sequence_id", "target_index", "regime", "switch", "distractor", "noise"):
                rows[key].append(batch[key].cpu())

    values = {key: torch.cat(parts, dim=0) for key, parts in rows.items()}
    correct = values["correct"].bool()
    metrics = {"accuracy": correct.float().mean().item()}

    for name in ("switch", "distractor", "noise"):
        mask = values[name].bool()
        if mask.any():
            metrics[f"{name}_accuracy"] = correct[mask].float().mean().item()
            if (~mask).any():
                metrics[f"non_{name}_accuracy"] = correct[~mask].float().mean().item()
        else:
            metrics[f"{name}_accuracy"] = float("nan")

    switch_recoveries: list[float] = []
    distractor_recoveries: list[float] = []
    post_distractor_acc: list[float] = []
    for seq_id in values["sequence_id"].unique().tolist():
        seq_mask = values["sequence_id"] == seq_id
        order = values["target_index"][seq_mask].argsort()
        seq_correct = correct[seq_mask][order]
        seq_switch = values["switch"][seq_mask][order].bool()
        seq_distractor = values["distractor"][seq_mask][order].bool()

        switch_recoveries.extend(
            _recovery_times(
                seq_correct,
                seq_switch,
                threshold=recovery_threshold,
                window=recovery_window,
                max_horizon=max_recovery_horizon,
            )
        )

        starts = seq_distractor & ~torch.roll(seq_distractor, shifts=1)
        starts[0] = seq_distractor[0]
        distractor_recoveries.extend(
            _recovery_times(
                seq_correct,
                starts,
                threshold=recovery_threshold,
                window=recovery_window,
                max_horizon=max_recovery_horizon,
            )
        )

        ends = seq_distractor & ~torch.roll(seq_distractor, shifts=-1)
        ends[-1] = seq_distractor[-1]
        for end_idx in ends.nonzero(as_tuple=False).flatten().tolist():
            start = end_idx + 1
            stop = min(len(seq_correct), start + recovery_window)
            if stop > start:
                post_distractor_acc.append(seq_correct[start:stop].float().mean().item())

    metrics["post_switch_recovery"] = _mean_or_nan(switch_recoveries)
    metrics["distractor_recovery"] = _mean_or_nan(distractor_recoveries)
    metrics["post_distractor_accuracy"] = _mean_or_nan(post_distractor_acc)
    metrics["num_switches"] = float(values["switch"].bool().sum().item())
    metrics["num_distractor_steps"] = float(values["distractor"].bool().sum().item())
    return EvalResult(metrics=metrics, predictions=values)


def evaluate_cpa_stateful(
    model: torch.nn.Module,
    dataset,
    *,
    device: torch.device,
    recovery_threshold: float = 0.8,
    recovery_window: int = 16,
    max_recovery_horizon: int = 100,
) -> EvalResult:
    if not hasattr(model, "initial_state") or not hasattr(model, "logits_from_state"):
        raise TypeError("stateful evaluation is only available for CPA-style recurrent models")

    model.eval()
    rows: dict[str, list[torch.Tensor]] = defaultdict(list)
    with torch.no_grad():
        for seq in dataset.sequences:
            A, N = model.initial_state(1, device)
            bits = seq.bits.to(device)
            for t in range(dataset.context_length):
                A, N, _ = model.step(bits[t : t + 1], A, N)
            for target_index in range(dataset.context_length, len(seq.bits)):
                logits = model.logits_from_state(A, N)
                pred = logits.argmax(dim=-1).cpu()
                target = seq.bits[target_index : target_index + 1].cpu()
                rows["correct"].append(pred == target)
                rows["pred"].append(pred)
                rows["target"].append(target)
                rows["sequence_id"].append(torch.tensor([seq.sequence_id], dtype=torch.long))
                rows["target_index"].append(torch.tensor([target_index], dtype=torch.long))
                rows["regime"].append(seq.regimes[target_index : target_index + 1].cpu())
                rows["switch"].append(seq.switches[target_index : target_index + 1].cpu())
                rows["distractor"].append(seq.distractors[target_index : target_index + 1].cpu())
                rows["noise"].append(seq.noise[target_index : target_index + 1].cpu())
                A, N, _ = model.step(bits[target_index : target_index + 1], A, N)

    values = {key: torch.cat(parts, dim=0) for key, parts in rows.items()}
    correct = values["correct"].bool()
    metrics = {"accuracy": correct.float().mean().item()}

    for name in ("switch", "distractor", "noise"):
        mask = values[name].bool()
        metrics[f"{name}_accuracy"] = correct[mask].float().mean().item() if mask.any() else float("nan")
        if mask.any() and (~mask).any():
            metrics[f"non_{name}_accuracy"] = correct[~mask].float().mean().item()

    switch_recoveries: list[float] = []
    distractor_recoveries: list[float] = []
    post_distractor_acc: list[float] = []
    for seq_id in values["sequence_id"].unique().tolist():
        seq_mask = values["sequence_id"] == seq_id
        order = values["target_index"][seq_mask].argsort()
        seq_correct = correct[seq_mask][order]
        seq_switch = values["switch"][seq_mask][order].bool()
        seq_distractor = values["distractor"][seq_mask][order].bool()
        switch_recoveries.extend(
            _recovery_times(
                seq_correct,
                seq_switch,
                threshold=recovery_threshold,
                window=recovery_window,
                max_horizon=max_recovery_horizon,
            )
        )
        starts = seq_distractor & ~torch.roll(seq_distractor, shifts=1)
        starts[0] = seq_distractor[0]
        distractor_recoveries.extend(
            _recovery_times(
                seq_correct,
                starts,
                threshold=recovery_threshold,
                window=recovery_window,
                max_horizon=max_recovery_horizon,
            )
        )
        ends = seq_distractor & ~torch.roll(seq_distractor, shifts=-1)
        ends[-1] = seq_distractor[-1]
        for end_idx in ends.nonzero(as_tuple=False).flatten().tolist():
            start = end_idx + 1
            stop = min(len(seq_correct), start + recovery_window)
            if stop > start:
                post_distractor_acc.append(seq_correct[start:stop].float().mean().item())

    metrics["post_switch_recovery"] = _mean_or_nan(switch_recoveries)
    metrics["distractor_recovery"] = _mean_or_nan(distractor_recoveries)
    metrics["post_distractor_accuracy"] = _mean_or_nan(post_distractor_acc)
    metrics["num_switches"] = float(values["switch"].bool().sum().item())
    metrics["num_distractor_steps"] = float(values["distractor"].bool().sum().item())
    return EvalResult(metrics=metrics, predictions=values)


def print_metrics(metrics: dict[str, float]) -> None:
    for key in sorted(metrics):
        value = metrics[key]
        if value != value:
            print(f"{key}: n/a")
        else:
            print(f"{key}: {value:.4f}")
