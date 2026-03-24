from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from value_policy.config import EnvConfig, ModelConfig


def _zeros(*shape: int, device: torch.device) -> torch.Tensor:
    return torch.zeros(*shape, device=device)


@dataclass
class SequenceOutput:
    logits: torch.Tensor
    values: torch.Tensor
    extras: dict[str, torch.Tensor]


class ActorCritic(nn.Module):
    def __init__(self, env_config: EnvConfig, model_config: ModelConfig) -> None:
        super().__init__()
        self.env_config = env_config
        self.model_config = model_config
        self.action_dim = env_config.action_dim

    def initial_state(self, batch_size: int, device: torch.device) -> Any:
        raise NotImplementedError

    def forward_step(
        self,
        obs: torch.Tensor,
        state: Any,
        intervention: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Any, dict[str, torch.Tensor]]:
        raise NotImplementedError

    def rollout(self, obs_seq: torch.Tensor, intervention: str | None = None) -> SequenceOutput:
        batch_size = obs_seq.size(0)
        state = self.initial_state(batch_size, obs_seq.device)
        logits_steps = []
        values_steps = []
        extras_steps: dict[str, list[torch.Tensor]] = {}
        for t in range(obs_seq.size(1)):
            logits, value, state, extras = self.forward_step(obs_seq[:, t], state, intervention=intervention)
            logits_steps.append(logits)
            values_steps.append(value)
            for key, value_tensor in extras.items():
                extras_steps.setdefault(key, []).append(value_tensor)
        stacked_extras = {key: torch.stack(value_list, dim=1) for key, value_list in extras_steps.items()}
        return SequenceOutput(
            logits=torch.stack(logits_steps, dim=1),
            values=torch.stack(values_steps, dim=1),
            extras=stacked_extras,
        )

    def act(
        self,
        obs: torch.Tensor,
        state: Any,
        intervention: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any, dict[str, torch.Tensor]]:
        logits, value, next_state, extras = self.forward_step(obs, state, intervention=intervention)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, value, entropy, next_state, extras

    def act_deterministic(
        self,
        obs: torch.Tensor,
        state: Any,
        intervention: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any, dict[str, torch.Tensor]]:
        logits, value, next_state, extras = self.forward_step(obs, state, intervention=intervention)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, value, entropy, next_state, extras


class MLPActorCritic(ActorCritic):
    def __init__(self, env_config: EnvConfig, model_config: ModelConfig) -> None:
        super().__init__(env_config, model_config)
        hidden = model_config.hidden_size
        self.net = nn.Sequential(
            nn.Linear(env_config.obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, env_config.action_dim)
        self.value_head = nn.Linear(hidden, 1)

    def initial_state(self, batch_size: int, device: torch.device) -> None:
        return None

    def forward_step(
        self,
        obs: torch.Tensor,
        state: None,
        intervention: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, None, dict[str, torch.Tensor]]:
        h = self.net(obs)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        extras = {"feature_norm": h.norm(dim=-1)}
        return logits, value, None, extras


class RecurrentActorCritic(ActorCritic):
    cell_cls: type[nn.Module]

    def __init__(self, env_config: EnvConfig, model_config: ModelConfig) -> None:
        super().__init__(env_config, model_config)
        hidden = model_config.hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(env_config.obs_dim, model_config.encoder_size),
            nn.Tanh(),
        )
        self.cell = self.cell_cls(model_config.encoder_size, hidden)
        self.policy_head = nn.Linear(hidden, env_config.action_dim)
        self.value_head = nn.Linear(hidden, 1)
        self.hidden_size = hidden

    def initial_state(self, batch_size: int, device: torch.device) -> Any:
        h = _zeros(batch_size, self.hidden_size, device=device)
        if isinstance(self.cell, nn.LSTMCell):
            return h, _zeros(batch_size, self.hidden_size, device=device)
        return h

    def forward_step(
        self,
        obs: torch.Tensor,
        state: Any,
        intervention: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Any, dict[str, torch.Tensor]]:
        x = self.encoder(obs)
        next_state = self.cell(x, state)
        h = next_state[0] if isinstance(next_state, tuple) else next_state
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        extras = {"hidden_norm": h.norm(dim=-1)}
        return logits, value, next_state, extras


class GRUActorCritic(RecurrentActorCritic):
    cell_cls = nn.GRUCell


class LSTMActorCritic(RecurrentActorCritic):
    cell_cls = nn.LSTMCell


class ChevronCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, uncertainty_scale: bool) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.uncertainty_scale = uncertainty_scale
        self.obs_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
        )
        self.up = nn.Linear(hidden_size, hidden_size)
        self.uv = nn.Linear(hidden_size, hidden_size)
        self.m_pp = nn.Parameter(torch.ones(hidden_size) * 0.5)
        self.m_pv = nn.Parameter(torch.zeros(hidden_size))
        self.m_vp = nn.Parameter(torch.zeros(hidden_size))
        self.m_vv = nn.Parameter(torch.ones(hidden_size) * 0.5)
        self.lambda_p = nn.Parameter(torch.full((hidden_size,), 0.1))
        self.lambda_v = nn.Parameter(torch.full((hidden_size,), 0.1))
        self.eta_p = nn.Parameter(torch.full((hidden_size,), 0.9))
        self.eta_v = nn.Parameter(torch.full((hidden_size,), 0.9))

    def forward(
        self, obs: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
        p, v = state
        h = self.obs_encoder(obs)
        p_mix = self.m_pp * p + self.m_pv * v + self.up(h)
        v_mix = self.m_vp * p + self.m_vv * v + self.uv(h)
        p_tilde = torch.tanh(p_mix)
        v_tilde = torch.tanh(v_mix)
        stance = p_tilde - v_tilde
        lean = torch.sigmoid(stance)
        gain = torch.sqrt(1e-4 + lean * (1.0 - lean))
        if not self.uncertainty_scale:
            gain = torch.ones_like(gain)

        lambda_p = torch.sigmoid(self.lambda_p)
        lambda_v = torch.sigmoid(self.lambda_v)
        eta_p = torch.sigmoid(self.eta_p)
        eta_v = torch.sigmoid(self.eta_v)
        next_p = (1.0 - lambda_p) * p + eta_p * gain * p_tilde
        next_v = (1.0 - lambda_v) * v + eta_v * gain * v_tilde
        extras = {
            "p_mean": next_p.mean(dim=-1),
            "v_mean": next_v.mean(dim=-1),
            "tension": lean.mean(dim=-1),
            "gain": gain.mean(dim=-1),
        }
        return (next_p, next_v), extras


class FreeChevronActorCritic(ActorCritic):
    def __init__(self, env_config: EnvConfig, model_config: ModelConfig) -> None:
        super().__init__(env_config, model_config)
        hidden = model_config.hidden_size
        self.cell = ChevronCell(env_config.obs_dim, hidden, model_config.uncertainty_scale)
        self.policy_head = nn.Linear(hidden * 2, env_config.action_dim)
        self.value_head = nn.Linear(hidden * 2, 1)
        self.hidden_size = hidden

    @staticmethod
    def _apply_intervention(
        state: tuple[torch.Tensor, torch.Tensor],
        intervention: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        p, v = state
        if intervention is None:
            return p, v
        if intervention == "zero_p":
            return torch.zeros_like(p), v
        if intervention == "zero_v":
            return p, torch.zeros_like(v)
        if intervention == "shuffle_p":
            return p[torch.randperm(p.size(0), device=p.device)], v
        if intervention == "shuffle_v":
            return p, v[torch.randperm(v.size(0), device=v.device)]
        raise ValueError(f"unsupported intervention={intervention!r}")

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return _zeros(batch_size, self.hidden_size, device=device), _zeros(batch_size, self.hidden_size, device=device)

    def forward_step(
        self,
        obs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        intervention: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
        next_state, extras = self.cell(obs, state)
        next_state = self._apply_intervention(next_state, intervention)
        z = torch.cat(next_state, dim=-1)
        logits = self.policy_head(z)
        value = self.value_head(z).squeeze(-1)
        return logits, value, next_state, extras


class StructuredChevronActorCritic(ActorCritic):
    def __init__(self, env_config: EnvConfig, model_config: ModelConfig) -> None:
        super().__init__(env_config, model_config)
        hidden = model_config.hidden_size
        self.cell = ChevronCell(env_config.obs_dim, hidden, model_config.uncertainty_scale)
        self.gated_policy = model_config.gated_policy
        self.policy_head = nn.Linear(hidden, env_config.action_dim)
        self.value_head = nn.Linear(hidden, 1)
        self.gate_head = nn.Linear(hidden, env_config.action_dim) if self.gated_policy else None
        self.hidden_size = hidden

    @staticmethod
    def _apply_intervention(
        state: tuple[torch.Tensor, torch.Tensor],
        intervention: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return FreeChevronActorCritic._apply_intervention(state, intervention)

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        return _zeros(batch_size, self.hidden_size, device=device), _zeros(batch_size, self.hidden_size, device=device)

    def forward_step(
        self,
        obs: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
        intervention: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
        next_state, extras = self.cell(obs, state)
        p, v = self._apply_intervention(next_state, intervention)
        next_state = (p, v)
        logits = self.policy_head(p)
        if self.gate_head is not None:
            logits = logits * (1.0 - torch.sigmoid(self.gate_head(v)))
        value = self.value_head(v).squeeze(-1)
        return logits, value, next_state, extras


def build_model(env_config: EnvConfig, model_config: ModelConfig) -> ActorCritic:
    factories: dict[str, type[ActorCritic]] = {
        "mlp": MLPActorCritic,
        "gru": GRUActorCritic,
        "lstm": LSTMActorCritic,
        "free_chevron": FreeChevronActorCritic,
        "structured_chevron": StructuredChevronActorCritic,
    }
    try:
        model_cls = factories[model_config.model_name]
    except KeyError as exc:
        raise ValueError(f"unknown model_name={model_config.model_name!r}") from exc
    return model_cls(env_config, model_config)
