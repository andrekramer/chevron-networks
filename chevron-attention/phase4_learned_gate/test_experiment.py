import unittest

import torch

from phase4_learned_gate.experiment import (
    GateTrainConfig,
    StochasticConfig,
    run_seed,
)


class Phase4ExperimentTest(unittest.TestCase):
    def test_learned_gate_supports_short_and_long_retention(self):
        result = run_seed(
            StochasticConfig(episodes=8),
            GateTrainConfig(steps=250, batch_size=96, d_model=32, hidden_size=48),
            seed=7,
            device=torch.device("cpu"),
        )
        integrated = result["integrated_idl"]
        always = result["always_update"]
        context_only = result["context_only"]

        self.assertGreater(integrated["context_gate_accuracy"], 0.95)
        self.assertGreater(integrated["short_context_accuracy"], 0.95)
        self.assertGreater(integrated["long_context_accuracy"], 0.95)
        self.assertGreater(integrated["short_probe_preserve_accuracy"], 0.85)
        self.assertGreater(integrated["long_probe_consolidate_accuracy"], 0.85)
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
