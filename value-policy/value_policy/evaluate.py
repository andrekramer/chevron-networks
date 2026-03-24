from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from value_policy.env import ContextualReversalBanditEnv
from value_policy.io import load_checkpoint
from value_policy.ppo import gather_rollout, set_seed, summarize_rollout


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a saved Value / Policy model checkpoint.")
    parser.add_argument("checkpoint")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--intervention", choices=["zero_p", "zero_v", "shuffle_p", "shuffle_v"])
    parser.add_argument("--channel-interventions", action="store_true")
    return parser


def evaluate_checkpoint(
    checkpoint_path: str,
    device: str = "cpu",
    num_envs: int = 256,
    episodes: int = 1024,
    seed: int = 17,
    greedy: bool = False,
    intervention: str | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    torch_device = torch.device(device)
    model, env_config, _, _, history, _ = load_checkpoint(checkpoint_path, device=torch_device)
    summaries = []
    with torch.no_grad():
        remaining = episodes
        while remaining > 0:
            batch_size = min(num_envs, remaining)
            env = ContextualReversalBanditEnv(env_config, batch_size, torch_device)
            rollout = gather_rollout(
                model,
                env,
                env_config.horizon,
                torch_device,
                deterministic=greedy,
                intervention=intervention,
            )
            summaries.append(summarize_rollout(rollout))
            remaining -= batch_size

    metrics: dict[str, float] = {}
    for key in summaries[0]:
        metrics[key] = sum(summary[key] for summary in summaries) / len(summaries)

    return {
        "checkpoint": checkpoint_path,
        "evaluation_mode": "greedy" if greedy else "sampled",
        "intervention": intervention,
        "episodes": episodes,
        "train_final": history[-1] if history else None,
        "eval": metrics,
    }


def evaluate_channel_interventions(
    checkpoint_path: str,
    device: str = "cpu",
    num_envs: int = 256,
    episodes: int = 1024,
    seed: int = 17,
) -> dict[str, Any]:
    base = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        num_envs=num_envs,
        episodes=episodes,
        seed=seed,
        greedy=False,
        intervention=None,
    )
    interventions: dict[str, Any] = {"base": base["eval"]}
    for intervention in ["zero_p", "zero_v", "shuffle_p", "shuffle_v"]:
        result = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            num_envs=num_envs,
            episodes=episodes,
            seed=seed,
            greedy=False,
            intervention=intervention,
        )
        deltas = {
            key: result["eval"][key] - base["eval"][key]
            for key in base["eval"]
        }
        interventions[intervention] = {
            "eval": result["eval"],
            "delta_vs_base": deltas,
        }
    return interventions


def main() -> None:
    args = build_parser().parse_args()
    payload = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        device=args.device,
        num_envs=args.num_envs,
        episodes=args.episodes,
        seed=args.seed,
        greedy=args.greedy,
        intervention=args.intervention,
    )
    if args.channel_interventions:
        payload["channel_interventions"] = evaluate_channel_interventions(
            checkpoint_path=args.checkpoint,
            device=args.device,
            num_envs=args.num_envs,
            episodes=args.episodes,
            seed=args.seed,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
