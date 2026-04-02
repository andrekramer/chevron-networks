from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.distributions import Categorical

from chevron_agent.rl.advantages import compute_gae
from chevron_agent.rl.buffers import RolloutBatch


def _stack_state(states: list):
    first = states[0]
    if isinstance(first, tuple):
        return tuple(torch.stack([s[idx] for s in states], dim=0) for idx in range(len(first)))
    return torch.stack(states, dim=0)


class RolloutCollector:
    def __init__(self, envs, device: torch.device, config) -> None:
        self.envs = envs
        self.device = device
        self.config = config
        obs, _ = envs.reset(seed=config.seed)
        self.obs = torch.tensor(obs, dtype=torch.float32, device=device)
        self.done = torch.zeros(config.num_envs, dtype=torch.float32, device=device)

    def collect(self, agent, hidden):
        obs_list = []
        actions_list = []
        logprobs_list = []
        rewards_list = []
        dones_list = []
        values_list = []
        hidden_list = []
        metrics = defaultdict(list)

        for _ in range(self.config.rollout_steps):
            done_mask = (1.0 - self.done).unsqueeze(-1)
            hidden_list.append(_clone_state(hidden))
            outputs = agent(self.obs, hidden, done_mask)
            dist = Categorical(logits=outputs["logits"])
            action = dist.sample()
            next_obs, reward, terminated, truncated, infos = self.envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            obs_list.append(self.obs.detach().clone())
            actions_list.append(action.detach())
            logprobs_list.append(dist.log_prob(action).detach())
            rewards_list.append(torch.tensor(reward, dtype=torch.float32, device=self.device))
            dones_list.append(torch.tensor(done, dtype=torch.float32, device=self.device))
            values_list.append(outputs["value"].detach())

            if "A" in outputs:
                metrics["A_mean"].append(outputs["A"].mean().detach())
                metrics["N_mean"].append(outputs["N"].mean().detach())
                metrics["A_minus_N_mean"].append((outputs["A"] - outputs["N"]).mean().detach())
                if outputs["tension"] is not None:
                    metrics["tension_mean"].append(outputs["tension"].mean().detach())

            self.obs = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
            self.done = torch.tensor(done, dtype=torch.float32, device=self.device)
            hidden = _clone_state(outputs["hidden"])

        done_mask = (1.0 - self.done).unsqueeze(-1)
        next_value = agent(self.obs, hidden, done_mask)["value"].detach()

        rewards = torch.stack(rewards_list)
        values = torch.stack(values_list)
        dones = torch.stack(dones_list)
        advantages, returns = compute_gae(
            rewards=rewards,
            values=values,
            dones=dones,
            next_value=next_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )
        batch = RolloutBatch(
            observations=torch.stack(obs_list),
            actions=torch.stack(actions_list),
            logprobs=torch.stack(logprobs_list),
            rewards=rewards,
            dones=dones,
            values=values,
            advantages=advantages,
            returns=returns,
            initial_states=_stack_state(hidden_list),
        )
        scalar_metrics = {key: torch.stack(val).mean().item() for key, val in metrics.items()}
        return batch, hidden, scalar_metrics


def _clone_state(state):
    if isinstance(state, tuple):
        return tuple(part.detach().clone() for part in state)
    return state.detach().clone()
