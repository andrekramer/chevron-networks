from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from value_policy.config import EnvConfig, ModelConfig, PPOConfig
from value_policy.models import ActorCritic, build_model


def save_checkpoint(
    path: str | Path,
    model: ActorCritic,
    env_config: EnvConfig,
    model_config: ModelConfig,
    ppo_config: PPOConfig,
    history: list[dict[str, float]],
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "env_config": asdict(env_config),
        "model_config": asdict(model_config),
        "ppo_config": asdict(ppo_config),
        "history": history,
    }
    torch.save(checkpoint, Path(path))


def load_checkpoint(
    path: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[ActorCritic, EnvConfig, ModelConfig, PPOConfig, list[dict[str, float]], dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=device)
    env_config = EnvConfig(**checkpoint["env_config"])
    model_config = ModelConfig(**checkpoint["model_config"])
    ppo_config = PPOConfig(**checkpoint["ppo_config"])
    model = build_model(env_config, model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    history = checkpoint.get("history", [])
    return model, env_config, model_config, ppo_config, history, checkpoint
