from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces


@dataclass(frozen=True)
class DelayedContextSpec:
    old_contexts: int = 8
    new_contexts: int = 4
    families: int = 4
    address_dim: int = 8
    diagnostic_dim: int = 8
    establishment_steps: int = 600
    introduction_steps: int = 600
    outcome_delay: int = 3
    observation_noise: float = 0.12
    reward_correct: float = 1.0
    reward_wrong: float = -1.0

    @property
    def total_contexts(self) -> int:
        return self.old_contexts + self.new_contexts

    @property
    def decision_steps(self) -> int:
        return self.establishment_steps + self.introduction_steps


@dataclass(frozen=True)
class PendingOutcome:
    due_step: int
    decision_index: int
    context_id: int
    action: int
    correct_action: int


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-8)


class DelayedContextEnv(gym.Env[np.ndarray, int]):
    """A contextual bandit whose action outcomes arrive several steps later.

    The observation contains two independently corrupted views of the current
    context followed by two public phase flags. Latent context IDs and correct
    actions appear only in ``info`` for auditing and must not be passed to an
    agent.
    """

    metadata = {"render_modes": []}

    def __init__(self, spec: DelayedContextSpec | None = None) -> None:
        super().__init__()
        self.spec = spec or DelayedContextSpec()
        if self.spec.old_contexts % self.spec.families != 0:
            raise ValueError("old_contexts must be divisible by families")
        if self.spec.new_contexts != self.spec.families:
            raise ValueError("v1 expects one new context per family")
        if self.spec.outcome_delay < 0:
            raise ValueError("outcome_delay must be non-negative")

        observation_dim = self.spec.address_dim + self.spec.diagnostic_dim + 2
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(2)

        self.rng = np.random.default_rng()
        self.address_prototypes = np.empty((0, self.spec.address_dim), dtype=np.float32)
        self.diagnostic_prototypes = np.empty((0, self.spec.diagnostic_dim), dtype=np.float32)
        self.correct_actions = np.empty(0, dtype=np.int64)
        self.context_schedule = np.empty(0, dtype=np.int64)
        self.pending: list[PendingOutcome] = []
        self.step_index = 0

    @property
    def address_slice(self) -> slice:
        return slice(0, self.spec.address_dim)

    @property
    def diagnostic_slice(self) -> slice:
        start = self.spec.address_dim
        return slice(start, start + self.spec.diagnostic_dim)

    @property
    def decision_active(self) -> bool:
        return self.step_index < self.spec.decision_steps

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        self.rng = np.random.default_rng(seed)
        self.step_index = 0
        self.pending = []
        self._make_lifetime()
        return self._observation(), self._info()

    def step(self, action: int):
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action: {action}")
        if self.step_index >= self.spec.decision_steps + self.spec.outcome_delay:
            raise RuntimeError("step called after lifetime ended")

        if self.decision_active:
            context_id = int(self.context_schedule[self.step_index])
            correct_action = int(self.correct_actions[context_id])
            self.pending.append(
                PendingOutcome(
                    due_step=self.step_index + self.spec.outcome_delay,
                    decision_index=self.step_index,
                    context_id=context_id,
                    action=int(action),
                    correct_action=correct_action,
                )
            )

        reward = 0.0
        delivered: PendingOutcome | None = None
        for index, outcome in enumerate(self.pending):
            if outcome.due_step == self.step_index:
                delivered = self.pending.pop(index)
                reward = (
                    self.spec.reward_correct
                    if delivered.action == delivered.correct_action
                    else self.spec.reward_wrong
                )
                break

        self.step_index += 1
        truncated = self.step_index >= self.spec.decision_steps + self.spec.outcome_delay
        info = self._info(delivered=delivered)
        return self._observation(), float(reward), False, truncated, info

    def _make_lifetime(self) -> None:
        self.address_prototypes, self.diagnostic_prototypes = self._make_prototypes()

        action_balance = np.arange(self.spec.total_contexts, dtype=np.int64) % 2
        self.rng.shuffle(action_balance)
        self.correct_actions = action_balance

        old_ids = np.arange(self.spec.old_contexts, dtype=np.int64)
        all_ids = np.arange(self.spec.total_contexts, dtype=np.int64)
        old_schedule = self._balanced_schedule(old_ids, self.spec.establishment_steps)
        mixed_schedule = self._balanced_schedule(all_ids, self.spec.introduction_steps)
        self.context_schedule = np.concatenate([old_schedule, mixed_schedule])

    def _make_prototypes(self) -> tuple[np.ndarray, np.ndarray]:
        address_family = self.rng.normal(size=(self.spec.families, self.spec.address_dim))
        diagnostic_family = self.rng.normal(size=(self.spec.families, self.spec.diagnostic_dim))
        address_family = np.stack([_unit(v) for v in address_family])
        diagnostic_family = np.stack([_unit(v) for v in diagnostic_family])

        addresses = []
        diagnostics = []
        old_per_family = self.spec.old_contexts // self.spec.families
        for context_id in range(self.spec.total_contexts):
            if context_id < self.spec.old_contexts:
                family = context_id // old_per_family
            else:
                family = context_id - self.spec.old_contexts
            address_identity = _unit(self.rng.normal(size=self.spec.address_dim))
            diagnostic_identity = _unit(self.rng.normal(size=self.spec.diagnostic_dim))
            addresses.append(_unit(0.78 * address_family[family] + 0.63 * address_identity))
            diagnostics.append(_unit(0.48 * diagnostic_family[family] + 0.88 * diagnostic_identity))
        return np.asarray(addresses, dtype=np.float32), np.asarray(diagnostics, dtype=np.float32)

    def _balanced_schedule(self, ids: np.ndarray, length: int) -> np.ndarray:
        repeats, remainder = divmod(length, len(ids))
        schedule = np.tile(ids, repeats)
        if remainder:
            schedule = np.concatenate([schedule, self.rng.choice(ids, size=remainder, replace=False)])
        self.rng.shuffle(schedule)
        return schedule

    def _observation(self) -> np.ndarray:
        if not self.decision_active:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        context_id = int(self.context_schedule[self.step_index])
        address = self.address_prototypes[context_id] + self.rng.normal(
            scale=self.spec.observation_noise,
            size=self.spec.address_dim,
        )
        diagnostic = self.diagnostic_prototypes[context_id] + self.rng.normal(
            scale=self.spec.observation_noise,
            size=self.spec.diagnostic_dim,
        )
        address = _unit(address).astype(np.float32)
        diagnostic = _unit(diagnostic).astype(np.float32)
        in_introduction = float(self.step_index >= self.spec.establishment_steps)
        phase = np.asarray([1.0 - in_introduction, in_introduction], dtype=np.float32)
        return np.concatenate([address, diagnostic, phase]).astype(np.float32)

    def _info(self, delivered: PendingOutcome | None = None) -> dict[str, int | bool | None]:
        context_id = int(self.context_schedule[self.step_index]) if self.decision_active else None
        info: dict[str, int | bool | None] = {
            "decision_active": self.decision_active,
            "decision_index": self.step_index if self.decision_active else None,
            "context_id": context_id,
            "is_new_context": context_id is not None and context_id >= self.spec.old_contexts,
            "outcome_decision_index": None,
            "outcome_context_id": None,
            "outcome_action": None,
            "outcome_correct_action": None,
        }
        if delivered is not None:
            info.update(
                {
                    "outcome_decision_index": delivered.decision_index,
                    "outcome_context_id": delivered.context_id,
                    "outcome_action": delivered.action,
                    "outcome_correct_action": delivered.correct_action,
                }
            )
        return info
