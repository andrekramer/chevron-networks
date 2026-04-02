from __future__ import annotations

import argparse
from pathlib import Path

import torch

from chevron_agent.analysis.visualize import save_frame
from chevron_agent.config import Config, load_config
from chevron_agent.envs import make_env
from chevron_agent.envs.cue_grid import RewardSpec
from chevron_agent.train import build_agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
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

    reward_spec = RewardSpec(
        correct=config.reward_correct,
        wrong=config.reward_wrong,
        trap=config.reward_trap,
        lure_immediate=config.reward_lure_immediate,
        lure_delayed=config.reward_lure_delayed,
        wait=config.reward_wait,
        step=config.reward_step,
        progress=config.reward_progress,
        regress=config.reward_regress,
    )
    env = make_env(
        grid_size=config.grid_size,
        max_steps=config.max_steps,
        reversal_period=config.reversal_period,
        reveal_wait_steps=config.reveal_wait_steps,
        lure_delay_steps=config.lure_delay_steps,
        observation_mode=config.observation_mode,
        fixed_layout=config.fixed_layout,
        auto_interact=config.auto_interact,
        layout_pool_size=config.layout_pool_size,
        include_trap=config.include_trap,
        include_lure=config.include_lure,
        reward_spec=reward_spec,
        **config.env_kwargs,
    )()
    obs, _ = env.reset(seed=config.seed)
    hidden = agent.initial_state(1, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 0
    done = False
    truncated = False
    while not (done or truncated):
        if obs.shape[0] == 3:
            save_frame(obs.transpose(1, 2, 0), output_dir / f"frame_{frame_idx:03d}.png")
        obs_tensor = torch.tensor(obs[None, ...], dtype=torch.float32, device=device)
        done_mask = torch.ones(1, 1, device=device)
        with torch.no_grad():
            outputs = agent(obs_tensor, hidden, done_mask)
            action = outputs["logits"].argmax(dim=-1).item()
            hidden = outputs["hidden"]
        obs, _, done, truncated, _ = env.step(action)
        frame_idx += 1


if __name__ == "__main__":
    main()
