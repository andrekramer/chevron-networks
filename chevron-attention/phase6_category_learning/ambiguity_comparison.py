"""Phase 6.2: bottom-up ambiguity with top-down template constraints.

Every episode contains a target memory and several decoys. Decoys are more
similar to the bottom-up A query than the target, but their N templates conflict
with the top-down context. All decoys share the wrong value, so ordinary
softmax value averaging becomes increasingly difficult as their number grows.
"""

from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import Tensor

from phase6_category_learning.experiment import complement_code, contrast_match


METHODS = (
    "a_softmax",
    "joint_softmax",
    "joint_top1",
    "complementarity_gate",
    "soft_vigilance",
    "sharp_vigilance",
    "masked_attention",
    "hard_search",
)


@dataclass(frozen=True)
class AmbiguityConfig:
    a_dimension: int = 8
    n_dimension: int = 8
    fillers: int = 4
    target_a_noise: float = 0.080
    decoy_a_noise: float = 0.015
    template_noise: float = 0.015
    a_temperature: float = 0.020
    n_temperature: float = 0.020
    vigilance: float = 0.060
    soft_sharpness: float = 80.0


@dataclass(frozen=True)
class Episode:
    query_a: Tensor
    context_n: Tensor
    keys_a: Tensor
    templates_n: Tensor
    values: Tensor
    target_slot: int
    decoy_slots: Tuple[int, ...]


def _binary_code(dimension: int, generator: torch.Generator) -> Tensor:
    return 0.10 + 0.80 * torch.randint(
        0, 2, (dimension,), generator=generator, dtype=torch.float32
    )


def generate_episode(
    config: AmbiguityConfig,
    decoys: int,
    rng: random.Random,
    generator: torch.Generator,
) -> Episode:
    query_a = 0.15 + 0.70 * torch.rand(config.a_dimension, generator=generator)
    context_n = _binary_code(config.n_dimension, generator)

    keys: List[Tensor] = []
    templates: List[Tensor] = []
    values: List[float] = []
    roles: List[str] = []

    target_key = (
        query_a
        + config.target_a_noise
        * torch.randn(config.a_dimension, generator=generator)
    ).clamp(0.0, 1.0)
    target_template = (
        context_n
        + config.template_noise
        * torch.randn(config.n_dimension, generator=generator)
    ).clamp(0.0, 1.0)
    keys.append(target_key)
    templates.append(target_template)
    values.append(1.0)
    roles.append("target")

    for index in range(decoys):
        key = (
            query_a
            + config.decoy_a_noise
            * torch.randn(config.a_dimension, generator=generator)
        ).clamp(0.0, 1.0)
        template = context_n.clone()
        bit = index % config.n_dimension
        template[bit] = 1.0 - template[bit]
        template = (
            template
            + config.template_noise
            * torch.randn(config.n_dimension, generator=generator)
        ).clamp(0.0, 1.0)
        keys.append(key)
        templates.append(template)
        values.append(0.0)
        roles.append("decoy")

    for _ in range(config.fillers):
        keys.append(torch.rand(config.a_dimension, generator=generator))
        templates.append(_binary_code(config.n_dimension, generator))
        values.append(0.0)
        roles.append("filler")

    order = list(range(len(keys)))
    rng.shuffle(order)
    ordered_roles = [roles[index] for index in order]
    target_slot = ordered_roles.index("target")
    decoy_slots = tuple(i for i, role in enumerate(ordered_roles) if role == "decoy")
    return Episode(
        query_a,
        context_n,
        torch.stack([keys[index] for index in order]),
        torch.stack([templates[index] for index in order]),
        torch.tensor([values[index] for index in order]),
        target_slot,
        decoy_slots,
    )


def mean_squared_distances(query: Tensor, memories: Tensor) -> Tensor:
    return ((memories - query.unsqueeze(0)) ** 2).mean(-1)


def n_signals(episode: Episode) -> Tuple[Tensor, Tensor]:
    mismatches: List[float] = []
    gates: List[float] = []
    encoded_context = complement_code(episode.context_n)
    for template in episode.templates_n:
        mismatch, gate = contrast_match(encoded_context, complement_code(template))
        mismatches.append(mismatch)
        gates.append(gate)
    return torch.tensor(mismatches), torch.tensor(gates)


def predict(method: str, episode: Episode, config: AmbiguityConfig) -> Tuple[int, int]:
    a_distance = mean_squared_distances(episode.query_a, episode.keys_a)
    n_distance = mean_squared_distances(episode.context_n, episode.templates_n)
    mismatch, complementarity = n_signals(episode)
    a_logits = -a_distance / config.a_temperature
    resets = 0

    if method == "hard_search":
        for slot in a_distance.argsort().tolist():
            if float(mismatch[slot].item()) <= config.vigilance:
                return int(episode.values[slot].item() >= 0.5), resets
            resets += 1
        return 0, resets

    if method == "joint_top1":
        logits = a_logits - n_distance / config.n_temperature
        slot = int(logits.argmax().item())
        return int(episode.values[slot].item() >= 0.5), resets

    if method == "a_softmax":
        weights = a_logits.softmax(0)
    elif method == "joint_softmax":
        weights = (a_logits - n_distance / config.n_temperature).softmax(0)
    elif method == "complementarity_gate":
        weights = a_logits.softmax(0) * complementarity
        weights = weights / weights.sum()
    elif method in ("soft_vigilance", "sharp_vigilance"):
        sharpness = (
            config.soft_sharpness if method == "soft_vigilance" else 2.0 * config.soft_sharpness
        )
        gate = torch.sigmoid(
            sharpness * (config.vigilance - mismatch)
        )
        weights = a_logits.softmax(0) * gate
        weights = weights / weights.sum()
    elif method == "masked_attention":
        masked_logits = a_logits.masked_fill(mismatch > config.vigilance, -1e4)
        weights = masked_logits.softmax(0)
    else:
        raise ValueError("unknown method: %s" % method)

    probability_one = float((weights * episode.values).sum().item())
    return int(probability_one >= 0.5), resets


def evaluate(
    method: str,
    config: AmbiguityConfig,
    decoys: int,
    seed: int,
    episodes: int,
) -> Dict[str, float]:
    rng = random.Random(seed + 60_000 + decoys)
    generator = torch.Generator().manual_seed(seed + 70_000 + decoys)
    samples = [generate_episode(config, decoys, rng, generator) for _ in range(episodes)]
    return evaluate_episodes(method, config, samples)


def evaluate_episodes(
    method: str,
    config: AmbiguityConfig,
    samples: Sequence[Episode],
) -> Dict[str, float]:
    correct = 0
    resets = 0
    target_rank = 0.0
    target_match = 0.0
    decoy_match = 0.0
    target_gate = 0.0
    decoy_gate = 0.0
    for episode in samples:
        prediction, episode_resets = predict(method, episode, config)
        correct += int(prediction == 1)
        resets += episode_resets
        distances = mean_squared_distances(episode.query_a, episode.keys_a)
        order = distances.argsort().tolist()
        target_rank += order.index(episode.target_slot) + 1
        mismatches, gates = n_signals(episode)
        target_match += float(mismatches[episode.target_slot].item())
        decoy_match += sum(
            float(mismatches[i].item()) for i in episode.decoy_slots
        ) / len(episode.decoy_slots)
        target_gate += float(gates[episode.target_slot].item())
        decoy_gate += sum(
            float(gates[i].item()) for i in episode.decoy_slots
        ) / len(episode.decoy_slots)
    count = len(samples)
    return {
        "accuracy": correct / count,
        "resets": resets / count,
        "target_a_rank": target_rank / count,
        "target_n_mismatch": target_match / count,
        "decoy_n_mismatch": decoy_match / count,
        "target_complementarity": target_gate / count,
        "decoy_complementarity": decoy_gate / count,
    }


def summarize(values: Sequence[float]) -> Tuple[float, float]:
    return (
        statistics.mean(values),
        statistics.stdev(values) if len(values) > 1 else 0.0,
    )


def run_comparison(
    config: AmbiguityConfig,
    decoy_counts: Sequence[int],
    seeds: Sequence[int],
    episodes: int,
) -> Dict[str, Dict[int, Dict[str, Tuple[float, float]]]]:
    output: Dict[str, Dict[int, Dict[str, Tuple[float, float]]]] = {
        method: {} for method in METHODS
    }
    for decoys in decoy_counts:
        per_method: Dict[str, List[Dict[str, float]]] = {method: [] for method in METHODS}
        for seed in seeds:
            rng = random.Random(seed + 60_000 + decoys)
            generator = torch.Generator().manual_seed(seed + 70_000 + decoys)
            samples = [
                generate_episode(config, decoys, rng, generator) for _ in range(episodes)
            ]
            for method in METHODS:
                per_method[method].append(evaluate_episodes(method, config, samples))
        for method in METHODS:
            rows = per_method[method]
            output[method][decoys] = {
                key: summarize([row[key] for row in rows]) for key in rows[0]
            }
    return output


def print_results(results: Dict[str, Dict[int, Dict[str, Tuple[float, float]]]]) -> None:
    decoy_counts = list(next(iter(results.values())).keys())
    print("accuracy mean+/-sd")
    print("method".ljust(25) + "".join(("d=%d" % d).rjust(18) for d in decoy_counts))
    for method in METHODS:
        cells = []
        for decoys in decoy_counts:
            average, spread = results[method][decoys]["accuracy"]
            cells.append(("%.4f+/-%.4f" % (average, spread)).rjust(18))
        print(method.ljust(25) + "".join(cells))
    print("\nhard-search diagnostics")
    for decoys in decoy_counts:
        row = results["hard_search"][decoys]
        print(
            "  decoys=%-2d resets=%.2f target_A_rank=%.2f target_N_mismatch=%.4f decoy_N_mismatch=%.4f target_G=%.4f decoy_G=%.4f"
            % (
                decoys,
                row["resets"][0],
                row["target_a_rank"][0],
                row["target_n_mismatch"][0],
                row["decoy_n_mismatch"][0],
                row["target_complementarity"][0],
                row["decoy_complementarity"][0],
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--decoys", type=int, nargs="+", default=[1, 3, 7, 15, 31])
    parser.add_argument("--episodes", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_comparison(AmbiguityConfig(), args.decoys, args.seeds, args.episodes)
    print_results(results)


if __name__ == "__main__":
    main()
