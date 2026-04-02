from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from chevron_agent.config import Config, load_config
from chevron_agent.envs import make_env
from chevron_agent.envs.cue_grid import RewardSpec
from chevron_agent.models import BaselineGRUAgent, ChevronAgent
from chevron_agent.rl.ppo import PPOTrainer
from chevron_agent.rl.rollout import RolloutCollector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_envs(config: Config):
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
    return gym.vector.SyncVectorEnv(
        [
            make_env(
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
            )
            for _ in range(config.num_envs)
        ]
    )


def build_agent(config: Config, action_dim: int, in_channels: int):
    if config.model_type == "baseline":
        return BaselineGRUAgent(
            hidden_dim=config.hidden_dim,
            action_dim=action_dim,
            image_size=config.grid_size,
            in_channels=in_channels,
        )
    if config.model_type == "chevron":
        return ChevronAgent(
            hidden_dim=config.hidden_dim,
            action_dim=action_dim,
            leak=config.leak,
            epsilon=config.epsilon,
            use_tension_gate=config.use_tension_gate,
            image_size=config.grid_size,
            in_channels=in_channels,
        )
    msg = f"Unknown model_type: {config.model_type}"
    raise ValueError(msg)


def initial_hidden(agent, batch_size: int, device: torch.device):
    return agent.initial_state(batch_size, device)


def save_checkpoint(agent, optimizer, config: Config, update: int, run_dir: Path) -> Path:
    ckpt_path = run_dir / f"update_{update:05d}.pt"
    latest_path = run_dir / "latest.pt"
    payload = {
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.__dict__,
        "update": update,
    }
    torch.save(payload, ckpt_path)
    torch.save(payload, latest_path)
    return ckpt_path


def save_named_checkpoint(
    agent,
    optimizer,
    config: Config,
    update: int,
    run_dir: Path,
    filename: str,
    extra: dict | None = None,
) -> Path:
    ckpt_path = run_dir / filename
    payload = {
        "model_state_dict": agent.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.__dict__,
        "update": update,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, ckpt_path)
    return ckpt_path


def evaluate(agent, config: Config, device: torch.device, episodes: int) -> dict[str, float]:
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
    returns = []
    successes = 0.0
    for episode in range(episodes):
        obs, _ = env.reset(seed=config.seed + episode)
        hidden = initial_hidden(agent, 1, device)
        done = False
        truncated = False
        total_reward = 0.0
        while not (done or truncated):
            obs_tensor = torch.tensor(obs[None, ...], dtype=torch.float32, device=device)
            done_mask = torch.ones(1, 1, device=device)
            with torch.no_grad():
                outputs = agent(obs_tensor, hidden, done_mask)
                action = outputs["logits"].argmax(dim=-1).item()
                hidden = outputs["hidden"]
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
        successes += float(reward > 0)
        returns.append(total_reward)
    return {
        "return_mean": float(np.mean(returns)),
        "success_rate": successes / max(1, episodes),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.seed)
    device = torch.device(config.device)

    run_dir = Path(config.log_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config.__dict__, indent=2), encoding="utf-8")

    envs = build_envs(config)
    agent = build_agent(
        config,
        action_dim=envs.single_action_space.n,
        in_channels=envs.single_observation_space.shape[0],
    ).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate)
    collector = RolloutCollector(envs=envs, device=device, config=config)
    trainer = PPOTrainer(agent=agent, optimizer=optimizer, config=config, device=device)
    hidden = initial_hidden(agent, config.num_envs, device)

    metrics_path = run_dir / "metrics.jsonl"
    best_eval_success = float("-inf")
    best_eval_return = float("-inf")
    best_eval_update = 0
    for update in range(1, config.total_updates + 1):
        batch, hidden, rollout_metrics = collector.collect(agent, hidden)
        train_metrics = trainer.update(batch)
        log_record = {"update": update, **rollout_metrics, **train_metrics}
        if update % config.eval_interval == 0 or update == 1:
            eval_metrics = evaluate(agent, config, device, episodes=config.eval_episodes)
            log_record.update({f"eval/{k}": v for k, v in eval_metrics.items()})
            eval_success = eval_metrics["success_rate"]
            eval_return = eval_metrics["return_mean"]
            is_best = (eval_success > best_eval_success) or (
                eval_success == best_eval_success and eval_return > best_eval_return
            )
            if is_best:
                best_eval_success = eval_success
                best_eval_return = eval_return
                best_eval_update = update
                save_named_checkpoint(
                    agent,
                    optimizer,
                    config,
                    update,
                    run_dir,
                    "best_eval.pt",
                    extra={"best_eval_success_rate": best_eval_success, "best_eval_return_mean": best_eval_return},
                )
            log_record["best_eval/success_rate"] = best_eval_success
            log_record["best_eval/return_mean"] = best_eval_return
            log_record["best_eval/update"] = best_eval_update
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log_record) + "\n")
        if update % config.checkpoint_interval == 0 or update == config.total_updates:
            save_checkpoint(agent, optimizer, config, update, run_dir)
        print(log_record)


if __name__ == "__main__":
    main()
