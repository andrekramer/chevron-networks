from __future__ import annotations

import argparse
import json

import torch

from chevron_agent.config import Config, load_config
from chevron_agent.train import build_agent, evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(config.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = Config(**checkpoint.get("config", config.__dict__))
    observation_channels = 3 if config.observation_mode == "rgb" else 9
    agent = build_agent(config, action_dim=6, in_channels=observation_channels).to(device)
    agent.load_state_dict(checkpoint["model_state_dict"])
    agent.eval()
    metrics = evaluate(agent, config, device, episodes=args.episodes or config.eval_episodes)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
