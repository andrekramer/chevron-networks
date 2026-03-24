from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import nn

from value_policy.config import EnvConfig, ModelConfig, PPOConfig
from value_policy.env import ContextualReversalBanditEnv, StepInfo
from value_policy.io import save_checkpoint
from value_policy.models import ActorCritic, build_model


@dataclass
class RolloutBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    entropies: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    infos: list[StepInfo]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, horizon = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(batch_size, device=rewards.device)
    next_value = torch.zeros(batch_size, device=rewards.device)
    next_nonterminal = torch.ones(batch_size, device=rewards.device)
    for t in reversed(range(horizon)):
        delta = rewards[:, t] + gamma * next_value * next_nonterminal - values[:, t]
        last_advantage = delta + gamma * gae_lambda * next_nonterminal * last_advantage
        advantages[:, t] = last_advantage
        next_value = values[:, t]
        next_nonterminal = 1.0 - dones[:, t]
    returns = advantages + values
    return advantages, returns


def gather_rollout(
    model: ActorCritic,
    env: ContextualReversalBanditEnv,
    horizon: int,
    device: torch.device,
    deterministic: bool = False,
    intervention: str | None = None,
) -> RolloutBatch:
    observations = []
    actions = []
    rewards = []
    dones = []
    entropies = []
    log_probs = []
    values = []
    infos: list[StepInfo] = []

    obs, info = env.reset()
    infos.append(info)
    state = model.initial_state(env.num_envs, device)
    for _ in range(horizon):
        observations.append(obs)
        if deterministic:
            action, log_prob, value, entropy, state, _ = model.act_deterministic(obs, state, intervention=intervention)
        else:
            action, log_prob, value, entropy, state, _ = model.act(obs, state, intervention=intervention)
        next_obs, reward, done, info = env.step(action)
        actions.append(action)
        rewards.append(reward)
        dones.append(done.to(torch.float32))
        entropies.append(entropy)
        log_probs.append(log_prob)
        values.append(value)
        infos.append(info)
        obs = next_obs

    return RolloutBatch(
        observations=torch.stack(observations, dim=1),
        actions=torch.stack(actions, dim=1),
        rewards=torch.stack(rewards, dim=1),
        dones=torch.stack(dones, dim=1),
        entropies=torch.stack(entropies, dim=1),
        old_log_probs=torch.stack(log_probs, dim=1),
        old_values=torch.stack(values, dim=1),
        infos=infos,
    )


def ppo_update(
    model: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    config: PPOConfig,
) -> dict[str, float]:
    batch_size, _, _ = rollout.observations.shape
    mb_size = max(1, batch_size // config.minibatches)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    for _ in range(config.epochs):
        perm = torch.randperm(batch_size, device=rollout.observations.device)
        for start in range(0, batch_size, mb_size):
            batch_idx = perm[start : start + mb_size]
            sequence_output = model.rollout(rollout.observations[batch_idx])
            dist = torch.distributions.Categorical(logits=sequence_output.logits)
            log_probs = dist.log_prob(rollout.actions[batch_idx])
            entropy = dist.entropy()
            ratio = (log_probs - rollout.old_log_probs[batch_idx]).exp()
            clipped = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps)
            policy_loss = -torch.minimum(ratio * advantages[batch_idx], clipped * advantages[batch_idx]).mean()
            value_loss = torch.nn.functional.mse_loss(sequence_output.values, returns[batch_idx])
            entropy_bonus = entropy.mean()
            loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_bonus
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            stats["policy_loss"] += float(policy_loss.detach())
            stats["value_loss"] += float(value_loss.detach())
            stats["entropy"] += float(entropy_bonus.detach())

    denom = config.epochs * config.minibatches
    return {key: value / denom for key, value in stats.items()}


def summarize_rollout(rollout: RolloutBatch) -> dict[str, float]:
    rewards = rollout.rewards
    actions = rollout.actions
    contexts = torch.stack([info.contexts for info in rollout.infos[1:]], dim=1)
    reversal_flags = torch.stack([info.reversal_now for info in rollout.infos[1:]], dim=1)
    wait_mask = actions == 2
    bad_action = ((actions == 0) & (contexts == 1)) | ((actions == 1) & (contexts == 0))
    reversal_any = reversal_flags.any(dim=1)
    reward_after_reversal = torch.zeros(rewards.size(0), device=rewards.device)
    recovery_steps = torch.full((rewards.size(0),), rewards.size(1), device=rewards.device, dtype=torch.float32)

    for env_idx in range(rewards.size(0)):
        positions = torch.nonzero(reversal_flags[env_idx], as_tuple=False).flatten()
        if positions.numel() == 0:
            continue
        start = int(positions[0].item())
        end = min(start + 2, rewards.size(1))
        reward_after_reversal[env_idx] = rewards[env_idx, start:end].sum()
        action_slice = actions[env_idx, start:]
        context_slice = contexts[env_idx, start:]
        correct_mask = ((action_slice == 0) & (context_slice == 0)) | ((action_slice == 1) & (context_slice == 1))
        correct_positions = torch.nonzero(correct_mask, as_tuple=False).flatten()
        if correct_positions.numel() > 0:
            recovery_steps[env_idx] = float(correct_positions[0].item() + 1)

    post_reversal_wait = wait_mask[reversal_flags].float().mean().item() if reversal_flags.any() else 0.0
    post_reversal_bad = bad_action[reversal_flags].float().mean().item() if reversal_flags.any() else 0.0
    first_two_commit = (actions[:, :2] != 2).float().mean().item()
    return {
        "mean_episode_reward": rewards.sum(dim=1).mean().item(),
        "mean_action_entropy": rollout.entropies.mean().item(),
        "wait_rate": wait_mask.float().mean().item(),
        "post_reversal_wait_rate": post_reversal_wait,
        "post_reversal_bad_action_rate": post_reversal_bad,
        "reward_first_two_steps_after_reversal": reward_after_reversal[reversal_any].mean().item() if reversal_any.any() else 0.0,
        "steps_to_recover_after_reversal": recovery_steps[reversal_any].mean().item() if reversal_any.any() else 0.0,
        "premature_commit_rate": first_two_commit,
    }


def train_experiment(
    env_config: EnvConfig,
    model_config: ModelConfig,
    ppo_config: PPOConfig,
    checkpoint_path: str | None = None,
) -> tuple[ActorCritic, list[dict[str, float]]]:
    set_seed(ppo_config.seed)
    device = torch.device(ppo_config.device)
    env = ContextualReversalBanditEnv(env_config, ppo_config.num_envs, device)
    model = build_model(env_config, model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_config.learning_rate)

    history: list[dict[str, float]] = []
    for update in range(1, ppo_config.updates + 1):
        with torch.no_grad():
            rollout = gather_rollout(model, env, env_config.horizon, device)
            advantages, returns = compute_gae(
                rollout.rewards,
                rollout.old_values,
                rollout.dones,
                ppo_config.gamma,
                ppo_config.gae_lambda,
            )

        update_stats = ppo_update(model, optimizer, rollout, advantages, returns, ppo_config)
        rollout_stats = summarize_rollout(rollout)
        row = {"update": float(update), **rollout_stats, **update_stats}
        history.append(row)
    if checkpoint_path is not None:
        save_checkpoint(checkpoint_path, model, env_config, model_config, ppo_config, history)
    return model, history
