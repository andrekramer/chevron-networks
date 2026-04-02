from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from chevron_agent.envs.render import render_layers, render_symbolic_layers


EMPTY = 0
AGENT = 1
CUE_BLUE = 2
CUE_RED = 3
MASKED_BLUE = 4
MASKED_RED = 5
TARGET_A = 6
TARGET_B = 7
TRAP = 8
LURE = 9


@dataclass
class RewardSpec:
    correct: float = 1.0
    wrong: float = -1.0
    trap: float = -1.0
    lure_immediate: float = 0.2
    lure_delayed: float = -0.6
    wait: float = -0.01
    step: float = -0.005
    progress: float = 0.0
    regress: float = 0.0


class CueGridEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        grid_size: int = 9,
        max_steps: int = 40,
        reversal_period: int = 50,
        reveal_wait_steps: int = 2,
        lure_delay_steps: int = 3,
        observation_mode: str = "rgb",
        fixed_layout: bool = False,
        auto_interact: bool = False,
        layout_pool_size: int = 0,
        include_trap: bool = True,
        include_lure: bool = True,
        reward_spec: RewardSpec | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.reversal_period = reversal_period
        self.reveal_wait_steps = reveal_wait_steps
        self.lure_delay_steps = lure_delay_steps
        self.observation_mode = observation_mode
        self.fixed_layout = fixed_layout
        self.auto_interact = auto_interact
        self.layout_pool_size = layout_pool_size
        self.include_trap = include_trap
        self.include_lure = include_lure
        self.reward_spec = reward_spec or RewardSpec()

        obs_channels = 3 if observation_mode == "rgb" else 9

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_channels, grid_size, grid_size),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(6)
        self.rng = np.random.default_rng(seed)

        self.episode_index = 0
        self.steps = 0
        self.wait_reveals = 0
        self.current_cue = "blue"
        self.correct_target = "a"
        self.pending_lure_penalty: int | None = None
        self.lure_triggered = False
        self.agent_pos = (0, 0)
        self.cue_pos = (0, 0)
        self.target_a_pos = (0, 0)
        self.target_b_pos = (0, 0)
        self.trap_pos = (0, 0)
        self.lure_pos = (0, 0)
        self.layout_pool: list[tuple[tuple[int, int], ...]] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        del options
        self.steps = 0
        self.wait_reveals = 0
        self.pending_lure_penalty = None
        self.lure_triggered = False
        self._sample_layout()
        self._set_mapping()
        obs = self._get_obs()
        info = self._get_info(was_ambiguous=self.wait_reveals < self.reveal_wait_steps)
        return obs, info

    def step(self, action: int):
        reward = self.reward_spec.step
        terminated = False
        truncated = False
        self.steps += 1
        was_ambiguous = self.wait_reveals < self.reveal_wait_steps
        prev_distance = self._distance_to_correct_target()

        if action == 5:
            reward += self.reward_spec.wait
            self.wait_reveals = min(self.reveal_wait_steps, self.wait_reveals + 1)
        elif action == 4:
            reward, terminated = self._handle_interact(reward)
        else:
            self._move(action)
            reward, terminated = self._handle_move_outcome(reward)
            reward += self._movement_shaping(prev_distance)

        if self.pending_lure_penalty is not None:
            self.pending_lure_penalty -= 1
            if self.pending_lure_penalty <= 0:
                reward += self.reward_spec.lure_delayed
                self.pending_lure_penalty = None

        if self.steps >= self.max_steps and not terminated:
            truncated = True

        obs = self._get_obs()
        info = self._get_info(was_ambiguous=was_ambiguous)
        if terminated or truncated:
            self.episode_index += 1
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._get_obs().transpose(1, 2, 0)

    def _sample_layout(self) -> None:
        if self.fixed_layout:
            mid = self.grid_size // 2
            self.agent_pos = (self.grid_size - 2, mid)
            self.cue_pos = (1, mid)
            self.target_a_pos = (self.grid_size - 2, 1)
            self.target_b_pos = (self.grid_size - 2, self.grid_size - 2)
            self.trap_pos = (1, 1)
            self.lure_pos = (1, self.grid_size - 2)
            return
        if self.layout_pool_size > 0:
            if not self.layout_pool:
                self.layout_pool = [self._draw_unique_layout() for _ in range(self.layout_pool_size)]
            layout = self.layout_pool[int(self.rng.integers(0, len(self.layout_pool)))]
            (
                self.agent_pos,
                self.cue_pos,
                self.target_a_pos,
                self.target_b_pos,
                self.trap_pos,
                self.lure_pos,
            ) = layout
            return
        self._assign_layout(self._draw_unique_layout())

    def _draw_unique_layout(self) -> tuple[tuple[int, int], ...]:
        positions = [(y, x) for y in range(self.grid_size) for x in range(self.grid_size)]
        self.rng.shuffle(positions)
        layout = list(positions[:6])
        if not self.include_trap:
            layout[4] = layout[1]
        if not self.include_lure:
            layout[5] = layout[1]
        return tuple(layout)

    def _assign_layout(self, layout: tuple[tuple[int, int], ...]) -> None:
        (
            self.agent_pos,
            self.cue_pos,
            self.target_a_pos,
            self.target_b_pos,
            self.trap_pos,
            self.lure_pos,
        ) = layout

    def _set_mapping(self) -> None:
        reversed_mapping = (self.episode_index // self.reversal_period) % 2 == 1
        self.current_cue = "blue" if self.rng.random() < 0.5 else "red"
        if self.current_cue == "blue":
            self.correct_target = "b" if reversed_mapping else "a"
        else:
            self.correct_target = "a" if reversed_mapping else "b"

    def _move(self, action: int) -> None:
        y, x = self.agent_pos
        if action == 0:
            y -= 1
        elif action == 1:
            y += 1
        elif action == 2:
            x -= 1
        elif action == 3:
            x += 1
        self.agent_pos = (int(np.clip(y, 0, self.grid_size - 1)), int(np.clip(x, 0, self.grid_size - 1)))

    def _handle_interact(self, reward: float) -> tuple[float, bool]:
        terminated = False
        if self.agent_pos == self.target_a_pos:
            reward += self.reward_spec.correct if self.correct_target == "a" else self.reward_spec.wrong
            terminated = True
        elif self.agent_pos == self.target_b_pos:
            reward += self.reward_spec.correct if self.correct_target == "b" else self.reward_spec.wrong
            terminated = True
        elif self.agent_pos == self.trap_pos:
            if self.include_trap:
                reward += self.reward_spec.trap
                terminated = True
        elif self.agent_pos == self.lure_pos:
            if self.include_lure:
                reward += self.reward_spec.lure_immediate
                self.pending_lure_penalty = self.lure_delay_steps
                self.lure_triggered = True
        return reward, terminated

    def _handle_move_outcome(self, reward: float) -> tuple[float, bool]:
        terminated = False
        if self.auto_interact:
            return self._handle_interact(reward)
        if self.include_trap and self.agent_pos == self.trap_pos:
            reward += self.reward_spec.trap
            terminated = True
        return reward, terminated

    def _cue_token(self) -> int:
        if self.current_cue == "blue":
            return CUE_BLUE if self.wait_reveals >= self.reveal_wait_steps else MASKED_BLUE
        return CUE_RED if self.wait_reveals >= self.reveal_wait_steps else MASKED_RED

    def _correct_target_pos(self) -> tuple[int, int]:
        return self.target_a_pos if self.correct_target == "a" else self.target_b_pos

    def _distance_to_correct_target(self) -> int:
        target_y, target_x = self._correct_target_pos()
        agent_y, agent_x = self.agent_pos
        return abs(target_y - agent_y) + abs(target_x - agent_x)

    def _movement_shaping(self, prev_distance: int) -> float:
        if self.reward_spec.progress == 0.0 and self.reward_spec.regress == 0.0:
            return 0.0
        new_distance = self._distance_to_correct_target()
        if new_distance < prev_distance:
            return self.reward_spec.progress
        if new_distance > prev_distance:
            return -self.reward_spec.regress
        return 0.0

    def _get_obs(self) -> np.ndarray:
        layers = [
            (self._cue_token(), self.cue_pos),
            (TARGET_A, self.target_a_pos),
            (TARGET_B, self.target_b_pos),
            (AGENT, self.agent_pos),
        ]
        if self.include_trap:
            layers.append((TRAP, self.trap_pos))
        if self.include_lure:
            layers.append((LURE, self.lure_pos))
        if self.observation_mode == "symbolic":
            return render_symbolic_layers(self.grid_size, layers)
        return render_layers(self.grid_size, layers)

    def _get_info(self, was_ambiguous: bool) -> dict:
        return {
            "cue_type": self.current_cue,
            "correct_target": self.correct_target,
            "reversal_block": self.episode_index // self.reversal_period,
            "was_ambiguous": was_ambiguous,
            "lure_triggered": self.lure_triggered,
            "delayed_lure_penalty_applied": self.pending_lure_penalty is None and self.lure_triggered,
            "episode_step": self.steps,
        }


def make_env(**kwargs) -> Callable[[], CueGridEnv]:
    def _factory() -> CueGridEnv:
        return CueGridEnv(**kwargs)

    return _factory
