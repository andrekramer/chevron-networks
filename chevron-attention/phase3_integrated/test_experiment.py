import unittest

from phase3_integrated.experiment import Config, Schedule, run_seed
from phase3_integrated.stochastic import StochasticConfig, run_stochastic_seed


class ScheduleTest(unittest.TestCase):
    def test_phase_boundaries(self):
        schedule = Schedule(
            stable=2,
            transient_revoke=3,
            recovery=4,
            sustained_revoke=5,
            post_revoke=6,
            sustained_restore=7,
            final=8,
        )
        self.assertEqual(schedule.total, 35)
        self.assertEqual(schedule.phase(1), "stable")
        self.assertEqual(schedule.phase(2), "transient_revoke")
        self.assertEqual(schedule.phase(5), "recovery")
        self.assertEqual(schedule.phase(9), "sustained_revoke")
        self.assertEqual(schedule.phase(14), "post_revoke")
        self.assertEqual(schedule.phase(20), "sustained_restore")
        self.assertEqual(schedule.phase(27), "final")


class IntegratedExperimentTest(unittest.TestCase):
    def test_integrated_idl_protects_transient_and_consolidates_persistent_policy(self):
        result = run_seed(Config(), 7)
        integrated = result["integrated_idl"]
        always = result["always_update"]
        fixed = result["fixed_slow"]
        context_only = result["context_only"]

        self.assertEqual(integrated["retrieval_accuracy"], 1.0)
        self.assertEqual(integrated["transient_current_revoke_accuracy"], 1.0)
        self.assertEqual(integrated["sustained_revoke_current_accuracy"], 1.0)
        self.assertEqual(integrated["restore_current_accuracy"], 1.0)
        self.assertGreater(integrated["recovery_active_accuracy"], 0.95)
        self.assertGreater(integrated["post_revoke_retained_accuracy"], 0.95)
        self.assertGreater(integrated["final_active_accuracy"], 0.95)

        self.assertLess(
            integrated["transient_retained_drift"],
            always["transient_retained_drift"],
        )
        self.assertGreater(
            integrated["post_revoke_retained_accuracy"],
            context_only["post_revoke_retained_accuracy"],
        )
        self.assertLess(
            integrated["revoke_consolidation_steps"],
            fixed["revoke_consolidation_steps"],
        )

    def test_result_is_deterministic_for_a_seed(self):
        first = run_seed(Config(), 17)
        second = run_seed(Config(), 17)
        self.assertEqual(first, second)

    def test_stochastic_idl_preserves_short_and_consolidates_long_overrides(self):
        result = run_stochastic_seed(StochasticConfig(episodes=12), 7)
        integrated = result["integrated_idl"]
        always = result["always_update"]
        context_only = result["context_only"]

        self.assertGreater(integrated["short_context_accuracy"], 0.95)
        self.assertGreater(integrated["long_context_accuracy"], 0.95)
        self.assertGreater(integrated["short_probe_preserve_accuracy"], 0.90)
        self.assertGreater(integrated["long_probe_consolidate_accuracy"], 0.90)
        self.assertGreater(
            integrated["short_probe_preserve_accuracy"],
            always["short_probe_preserve_accuracy"],
        )
        self.assertGreater(
            integrated["long_probe_consolidate_accuracy"],
            context_only["long_probe_consolidate_accuracy"],
        )


if __name__ == "__main__":
    unittest.main()
