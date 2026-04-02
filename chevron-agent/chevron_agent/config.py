from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    seed: int = 0
    device: str = "cpu"
    run_name: str = "debug"
    model_type: str = "baseline"
    total_updates: int = 100
    num_envs: int = 8
    rollout_steps: int = 128
    learning_rate: float = 3e-4
    hidden_dim: int = 128
    leak: float = 0.1
    epsilon: float = 1e-3
    use_tension_gate: bool = False
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatches: int = 4
    eval_interval: int = 25
    eval_episodes: int = 20
    max_steps: int = 40
    reversal_period: int = 50
    reveal_wait_steps: int = 2
    grid_size: int = 9
    observation_mode: str = "rgb"
    reward_correct: float = 1.0
    reward_wrong: float = -1.0
    reward_trap: float = -1.0
    reward_lure_immediate: float = 0.2
    reward_lure_delayed: float = -0.6
    lure_delay_steps: int = 3
    reward_wait: float = -0.01
    reward_step: float = -0.005
    log_dir: str = "runs"
    checkpoint_interval: int = 25
    fixed_layout: bool = False
    auto_interact: bool = False
    layout_pool_size: int = 0
    include_trap: bool = True
    include_lure: bool = True
    reward_progress: float = 0.0
    reward_regress: float = 0.0
    env_kwargs: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return Config(**raw)
