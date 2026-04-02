from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


def _normalize_advantages(advantages: torch.Tensor) -> torch.Tensor:
    return (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)


def _index_state(state, env_indices: torch.Tensor):
    if isinstance(state, tuple):
        return tuple(part.index_select(0, env_indices) for part in state)
    return state.index_select(0, env_indices)


def _initial_sequence_state(initial_states, env_indices: torch.Tensor):
    if isinstance(initial_states, tuple):
        return tuple(part[0].index_select(0, env_indices) for part in initial_states)
    return initial_states[0].index_select(0, env_indices)


class PPOTrainer:
    def __init__(self, agent, optimizer, config, device: torch.device) -> None:
        self.agent = agent
        self.optimizer = optimizer
        self.config = config
        self.device = device

    def update(self, batch) -> dict[str, float]:
        obs = batch.observations
        actions = batch.actions
        old_logprobs = batch.logprobs
        advantages = _normalize_advantages(batch.advantages)
        returns = batch.returns
        dones = batch.dones

        num_envs = obs.shape[1]
        minibatch_size = math.ceil(num_envs / self.config.minibatches)

        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        steps = 0

        for _ in range(self.config.ppo_epochs):
            permutation = torch.randperm(num_envs, device=self.device)
            for start in range(0, num_envs, minibatch_size):
                indices = permutation[start : start + minibatch_size]
                mb_obs = obs[:, indices]
                mb_actions = actions[:, indices]
                mb_old_logprobs = old_logprobs[:, indices]
                mb_advantages = advantages[:, indices]
                mb_returns = returns[:, indices]
                mb_dones = dones[:, indices]

                hidden = _initial_sequence_state(batch.initial_states, indices)
                logits_seq = []
                values_seq = []
                prev_done = torch.zeros(len(indices), device=self.device)

                for t in range(mb_obs.shape[0]):
                    done_mask = (1.0 - prev_done).unsqueeze(-1)
                    outputs = self.agent(mb_obs[t], hidden, done_mask)
                    logits_seq.append(outputs["logits"])
                    values_seq.append(outputs["value"])
                    hidden = outputs["hidden"]
                    prev_done = mb_dones[t]

                logits = torch.stack(logits_seq)
                values = torch.stack(values_seq)
                dist = Categorical(logits=logits)
                new_logprobs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()
                ratio = (new_logprobs - mb_old_logprobs).exp()
                unclipped = ratio * mb_advantages
                clipped = torch.clamp(ratio, 1.0 - self.config.clip_coef, 1.0 + self.config.clip_coef) * mb_advantages
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = F.mse_loss(values, mb_returns)
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"] += float(policy_loss.detach())
                metrics["value_loss"] += float(value_loss.detach())
                metrics["entropy"] += float(entropy.detach())
                steps += 1

        return {key: value / max(1, steps) for key, value in metrics.items()}
