import random
import unittest

import torch

from phase5_complete_network.experiment import (
    ChevronMemoryNetwork,
    CycleConfig,
    IDLConfig,
    RecallAndControlTask,
    SignalNoise,
    TaskConfig,
    TrainConfig,
    cycle_metrics,
    losses,
    run_cycle,
    train_model,
)


class CompleteNetworkTest(unittest.TestCase):
    def setUp(self):
        self.task = RecallAndControlTask(
            TaskConfig(num_keys=8, num_values=8, num_facts=4, max_controls=3)
        )

    def test_qk_a_v_n_forward_and_backward(self):
        batch = self.task.batch(8, random.Random(3))
        model = ChevronMemoryNetwork(self.task.config, d_model=16)
        outputs = model(batch)
        self.assertEqual(outputs["answer_logits"].shape, (8, 9))
        self.assertEqual(outputs["alpha"].shape, (8, 4))
        self.assertEqual(outputs["context_logits"].shape, (8, 4, 3))
        objective = losses(outputs, batch, TrainConfig(d_model=16))["total"]
        objective.backward()
        self.assertIsNotNone(model.q_a.weight.grad)
        self.assertIsNotNone(model.k_a.weight.grad)
        self.assertIsNotNone(model.v_n.weight.grad)
        self.assertTrue(torch.isfinite(objective))

    def test_learned_network_drives_stability_plasticity_cycle(self):
        device = torch.device("cpu")
        model = train_model(
            self.task,
            TrainConfig(steps=350, batch_size=96, d_model=32),
            seed=7,
            device=device,
        )
        cycle = CycleConfig(
            stable=8,
            short_revoke=10,
            short_probe=12,
            long_revoke=65,
            revoke_probe=12,
            long_restore=65,
            final_probe=12,
            target_query_probability=1.0,
        )
        integrated = cycle_metrics(
            run_cycle("integrated_idl", model, self.task, cycle, IDLConfig(), 7, device)
        )
        always = cycle_metrics(
            run_cycle("always_update", model, self.task, cycle, IDLConfig(), 7, device)
        )
        context_only = cycle_metrics(
            run_cycle("context_only", model, self.task, cycle, IDLConfig(), 7, device)
        )
        self.assertGreater(integrated["retrieval_accuracy"], 0.98)
        self.assertGreater(integrated["context_accuracy"], 0.98)
        self.assertGreater(integrated["short_probe_preserve"], 0.90)
        self.assertGreater(integrated["long_probe_consolidate"], 0.90)
        self.assertGreater(integrated["restore_probe_consolidate"], 0.90)
        self.assertGreater(integrated["full_revoke_restore_cycle"], 0.90)
        self.assertGreater(
            integrated["short_probe_preserve"], always["short_probe_preserve"]
        )
        self.assertGreater(
            integrated["long_probe_consolidate"],
            context_only["long_probe_consolidate"],
        )

    def test_signal_noise_is_reproducible(self):
        device = torch.device("cpu")
        model = train_model(
            self.task,
            TrainConfig(steps=150, batch_size=64, d_model=24),
            seed=11,
            device=device,
        )
        cycle = CycleConfig(
            stable=2,
            short_revoke=3,
            short_probe=2,
            long_revoke=8,
            revoke_probe=2,
            long_restore=8,
            final_probe=2,
        )
        noise = SignalNoise(gaussian_std=0.1, dropout_probability=0.1)
        first = run_cycle(
            "integrated_idl", model, self.task, cycle, IDLConfig(), 19, device, noise
        )
        second = run_cycle(
            "integrated_idl", model, self.task, cycle, IDLConfig(), 19, device, noise
        )
        self.assertEqual(first, second)

    def test_directional_persistence_modes_complete_clean_cycle(self):
        device = torch.device("cpu")
        model = train_model(
            self.task,
            TrainConfig(steps=250, batch_size=96, d_model=32),
            seed=23,
            device=device,
        )
        for mode in ("hard_reset", "two_trace", "signed_hysteresis"):
            metrics = cycle_metrics(
                run_cycle(
                    "integrated_idl",
                    model,
                    self.task,
                    CycleConfig(target_query_probability=1.0),
                    IDLConfig(persistence_mode=mode),
                    23,
                    device,
                )
            )
            self.assertGreater(metrics["short_probe_preserve"], 0.9, mode)
            self.assertGreater(metrics["long_probe_consolidate"], 0.9, mode)
            self.assertGreater(metrics["restore_probe_consolidate"], 0.9, mode)
            self.assertGreater(metrics["full_revoke_restore_cycle"], 0.9, mode)




if __name__ == "__main__":
    unittest.main()
