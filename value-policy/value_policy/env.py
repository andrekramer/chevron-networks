from __future__ import annotations

from dataclasses import dataclass

import torch

from value_policy.config import EnvConfig


@dataclass(slots=True)
class StepInfo:
    contexts: torch.Tensor
    reversal_now: torch.Tensor
    reversal_cue_now: torch.Tensor
    reversal_step: torch.Tensor
    timestep: torch.Tensor


class ContextualReversalBanditEnv:
    def __init__(self, config: EnvConfig, num_envs: int, device: torch.device) -> None:
        self.config = config
        self.num_envs = num_envs
        self.device = device
        self._step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._contexts = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._reversal_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._reversal_cue_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._active = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self) -> tuple[torch.Tensor, StepInfo]:
        self._step.zero_()
        self._active.fill_(True)
        self._contexts = torch.bernoulli(
            0.5 * torch.ones(self.num_envs, device=self.device)
        ).long()
        reversal_mask = torch.bernoulli(
            self.config.p_reversal * torch.ones(self.num_envs, device=self.device)
        ).bool()
        reversal_steps = torch.randint(
            self.config.reversal_min_step,
            self.config.reversal_max_step + 1,
            (self.num_envs,),
            device=self.device,
        )
        self._reversal_step = torch.where(
            reversal_mask,
            reversal_steps,
            torch.full_like(reversal_steps, -1),
        )
        self._reversal_cue_step = torch.where(
            reversal_mask,
            self._reversal_step + self.config.reversal_cue_delay,
            torch.full_like(reversal_steps, -1),
        )
        obs, info = self._build_observation()
        return obs, info

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, StepInfo]:
        actions = actions.to(self.device)
        current_context = self._contexts.clone()
        reward = torch.full((self.num_envs,), self.config.wait_penalty, device=self.device)
        choose_a = actions == 0
        choose_b = actions == 1
        reward = torch.where(choose_a, torch.where(current_context == 0, 1.0, -1.0), reward)
        reward = torch.where(choose_b, torch.where(current_context == 1, 1.0, -1.0), reward)

        self._step += 1
        done = self._step >= self.config.horizon
        self._active = ~done

        obs, info = self._build_observation()
        return obs, reward, done, info

    def _build_observation(self) -> tuple[torch.Tensor, StepInfo]:
        reversal_now = self._step == self._reversal_step
        reversal_cue_now = self._step == self._reversal_cue_step
        self._contexts = torch.where(reversal_now, 1 - self._contexts, self._contexts)

        positive_prob = torch.where(
            self._contexts == 0,
            torch.full_like(self._contexts, self.config.cue_prob, dtype=torch.float32),
            torch.full_like(self._contexts, 1.0 - self.config.cue_prob, dtype=torch.float32),
        )
        positive_sample = torch.bernoulli(positive_prob).to(torch.float32)
        sign = positive_sample * 2.0 - 1.0
        reversal_flag = reversal_cue_now.to(torch.float32)
        distractors = torch.randn(
            self.num_envs,
            self.config.distractor_dim,
            device=self.device,
        )
        obs = torch.cat(
            [sign.unsqueeze(-1), reversal_flag.unsqueeze(-1), distractors],
            dim=-1,
        )
        info = StepInfo(
            contexts=self._contexts.clone(),
            reversal_now=reversal_now.clone(),
            reversal_cue_now=reversal_cue_now.clone(),
            reversal_step=self._reversal_step.clone(),
            timestep=self._step.clone(),
        )
        return obs, info
