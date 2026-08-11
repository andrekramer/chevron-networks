from __future__ import annotations

import unittest

import numpy as np

from chevron_agent.envs.delayed_context import DelayedContextEnv, DelayedContextSpec


class DelayedContextEnvTests(unittest.TestCase):
    def test_schedule_is_balanced_and_new_contexts_arrive_later(self) -> None:
        spec = DelayedContextSpec(establishment_steps=80, introduction_steps=120)
        env = DelayedContextEnv(spec)
        env.reset(seed=7)

        old = env.context_schedule[: spec.establishment_steps]
        mixed = env.context_schedule[spec.establishment_steps :]
        self.assertEqual(set(old.tolist()), set(range(8)))
        self.assertEqual(set(mixed.tolist()), set(range(12)))
        np.testing.assert_array_equal(np.bincount(old, minlength=12)[:8], np.full(8, 10))
        np.testing.assert_array_equal(np.bincount(mixed, minlength=12), np.full(12, 10))

    def test_outcome_is_delivered_after_exact_delay(self) -> None:
        spec = DelayedContextSpec(
            establishment_steps=8,
            introduction_steps=12,
            outcome_delay=3,
            observation_noise=0.0,
        )
        env = DelayedContextEnv(spec)
        _, info = env.reset(seed=11)
        first_context = int(info["context_id"])
        first_action = int(env.correct_actions[first_context])

        _, reward0, _, _, info0 = env.step(first_action)
        _, reward1, _, _, info1 = env.step(0)
        _, reward2, _, _, info2 = env.step(0)
        _, reward3, _, _, info3 = env.step(0)

        self.assertEqual([reward0, reward1, reward2], [0.0, 0.0, 0.0])
        self.assertEqual(reward3, 1.0)
        self.assertIsNone(info0["outcome_decision_index"])
        self.assertIsNone(info1["outcome_decision_index"])
        self.assertIsNone(info2["outcome_decision_index"])
        self.assertEqual(info3["outcome_decision_index"], 0)
        self.assertEqual(info3["outcome_context_id"], first_context)

    def test_zero_delay_delivers_current_action_outcome(self) -> None:
        spec = DelayedContextSpec(
            establishment_steps=8,
            introduction_steps=12,
            outcome_delay=0,
        )
        env = DelayedContextEnv(spec)
        _, info = env.reset(seed=13)
        context_id = int(info["context_id"])
        correct_action = int(env.correct_actions[context_id])
        _, reward, _, _, outcome_info = env.step(correct_action)
        self.assertEqual(reward, 1.0)
        self.assertEqual(outcome_info["outcome_decision_index"], 0)

    def test_seed_reproduces_lifetime_but_views_are_independently_noisy(self) -> None:
        env_a = DelayedContextEnv()
        env_b = DelayedContextEnv()
        obs_a, _ = env_a.reset(seed=23)
        obs_b, _ = env_b.reset(seed=23)
        np.testing.assert_allclose(obs_a, obs_b)
        np.testing.assert_array_equal(env_a.context_schedule, env_b.context_schedule)
        np.testing.assert_array_equal(env_a.correct_actions, env_b.correct_actions)

        address = obs_a[env_a.address_slice]
        diagnostic = obs_a[env_a.diagnostic_slice]
        self.assertFalse(np.allclose(address, diagnostic))

    def test_flush_delivers_every_pending_outcome(self) -> None:
        spec = DelayedContextSpec(
            establishment_steps=8,
            introduction_steps=12,
            outcome_delay=3,
        )
        env = DelayedContextEnv(spec)
        _, info = env.reset(seed=29)
        delivered = []
        truncated = False
        while not truncated:
            action = 0
            _, _, _, truncated, info = env.step(action)
            if info["outcome_decision_index"] is not None:
                delivered.append(int(info["outcome_decision_index"]))
        self.assertEqual(delivered, list(range(spec.decision_steps)))
        self.assertEqual(env.pending, [])


if __name__ == "__main__":
    unittest.main()
