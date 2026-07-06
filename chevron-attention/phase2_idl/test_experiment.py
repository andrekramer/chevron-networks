import unittest

import torch

from phase2_idl.experiment import (
    Config,
    Schedule,
    make_regimes,
    run_method,
    run_seed,
)


class ScheduleTest(unittest.TestCase):
    def test_phase_boundaries(self):
        schedule = Schedule(stable=3, transient=2, recovery=4, sustained=5)
        self.assertEqual(schedule.total, 14)
        self.assertEqual(schedule.phase(2), "stable")
        self.assertEqual(schedule.phase(3), "transient")
        self.assertEqual(schedule.phase(5), "recovery")
        self.assertEqual(schedule.phase(9), "sustained")


class ExperimentTest(unittest.TestCase):
    def test_run_is_deterministic(self):
        config = Config(
            dimensions=4,
            schedule=Schedule(stable=20, transient=3, recovery=10, sustained=20),
        )
        first = run_seed(config, 11)
        second = run_seed(config, 11)
        self.assertEqual(first, second)

    def test_methods_share_identical_fast_learner(self):
        config = Config(
            dimensions=4,
            schedule=Schedule(stable=20, transient=3, recovery=10, sustained=20),
        )
        generator = torch.Generator().manual_seed(4)
        regimes = make_regimes(config.dimensions, generator)
        idl = run_method("idl", config, 4, regimes)
        scaled = run_method("idl_scaled", config, 4, regimes)
        always = run_method("always_slow", config, 4, regimes)
        for idl_state, scaled_state, always_state in zip(idl.a, scaled.a, always.a):
            self.assertTrue(torch.equal(idl_state, always_state))
            self.assertTrue(torch.equal(scaled_state, always_state))

    def test_default_idl_protects_then_adapts(self):
        result = run_seed(Config(), 7)
        idl = result["idl"]
        always = result["always_slow"]
        fixed_low = result["fixed_slow_low"]
        self.assertLess(idl["transient_n_drift"], always["transient_n_drift"])
        self.assertLess(idl["sustained_adaptation_steps"], fixed_low["sustained_adaptation_steps"])
        self.assertLess(idl["transient_gate_mean"], 0.5)
        self.assertGreater(idl["sustained_early_gate_mean"], 0.5)
        self.assertLess(idl["pre_transient_base_error"], 0.1)
        self.assertLess(idl["sustained_final_error"], 0.02)

    def test_scaled_idl_consolidates_small_persistent_change(self):
        config = Config()
        result = run_seed(config, 7, shift_magnitude=0.25)
        absolute = result["idl"]
        scaled = result["idl_scaled"]
        always = result["always_slow"]
        self.assertLess(
            scaled["sustained_adaptation_steps"],
            absolute["sustained_adaptation_steps"],
        )
        self.assertLess(scaled["sustained_final_error"], 0.01)
        self.assertLess(scaled["transient_n_drift"], always["transient_n_drift"])


if __name__ == "__main__":
    unittest.main()
