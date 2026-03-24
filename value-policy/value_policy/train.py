from __future__ import annotations

import argparse
import json

from value_policy.config import EnvConfig, ModelConfig, PPOConfig
from value_policy.ppo import train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the Value / Policy experiment scaffold.")
    parser.add_argument("--model", default="structured_chevron", choices=["mlp", "gru", "lstm", "free_chevron", "structured_chevron"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--updates", type=int, default=50)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--encoder-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cue-prob", type=float, default=0.60)
    parser.add_argument("--wait-penalty", type=float, default=-0.01)
    parser.add_argument("--reversal-cue-delay", type=int, default=1)
    parser.add_argument("--disable-uncertainty-scale", action="store_true")
    parser.add_argument("--gated-policy", action="store_true")
    parser.add_argument("--checkpoint-path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    env_config = EnvConfig(
        cue_prob=args.cue_prob,
        wait_penalty=args.wait_penalty,
        reversal_cue_delay=args.reversal_cue_delay,
    )
    model_config = ModelConfig(
        model_name=args.model,
        hidden_size=args.hidden_size,
        encoder_size=args.encoder_size,
        uncertainty_scale=not args.disable_uncertainty_scale,
        gated_policy=args.gated_policy,
    )
    ppo_config = PPOConfig(
        num_envs=args.num_envs,
        updates=args.updates,
        seed=args.seed,
        device=args.device,
    )
    _, history = train_experiment(
        env_config,
        model_config,
        ppo_config,
        checkpoint_path=args.checkpoint_path,
    )
    print(json.dumps(history[-1], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
