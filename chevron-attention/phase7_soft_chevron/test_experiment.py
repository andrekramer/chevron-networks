import unittest

import torch
from dataclasses import replace

from phase7_soft_chevron.experiment import (
    JOINT_ATTENTION,
    SOFT_CHEVRON,
    CategoryMatchingTask,
    TaskConfig,
    TrainConfig,
    build_model,
    evaluate,
    losses,
    train_model,
)
from phase7_soft_chevron.continual_memory import (
    ContinualCategories,
    ContinualConfig,
    forward_memory,
)
from phase7_soft_chevron.continual_closure import (
    JointWriteController,
    random_active_mask,
)


class PhaseSevenTest(unittest.TestCase):
    def setUp(self):
        self.task = CategoryMatchingTask(
            TaskConfig(
                a_dimension=8,
                n_dimension=6,
                num_groups=2,
                group_size=3,
                num_values=8,
            )
        )

    def test_soft_chevron_formula_and_gradients(self):
        device = torch.device("cpu")
        config = TrainConfig(d_model=24)
        model = train_model(SOFT_CHEVRON, self.task, config, 3, device)
        generator = torch.Generator().manual_seed(4)
        batch = self.task.batch(12, generator, device)
        outputs = model(batch)
        self.assertEqual(outputs["alpha"].shape, (12, 6))
        self.assertEqual(outputs["r"].shape, (12, 6))
        self.assertTrue(torch.allclose(outputs["alpha"].sum(-1), torch.ones(12)))
        self.assertTrue(
            torch.allclose(
                outputs["slot_mass"].sum(-1) + outputs["null_mass"],
                torch.ones(12),
                atol=1e-6,
            )
        )
        objective = losses(SOFT_CHEVRON, outputs, batch, self.task.config, config)["total"]
        model.zero_grad(set_to_none=True)
        objective.backward()
        self.assertIsNotNone(model.q_a.weight.grad)
        self.assertIsNotNone(model.k_a.weight.grad)
        self.assertIsNotNone(model.match_a.weight.grad)
        self.assertIsNotNone(model.match_n.weight.grad)
        self.assertIsNotNone(model.output.v_n.weight.grad)
        self.assertIsNotNone(model.theta_logit.grad)
        self.assertIsNotNone(model.k_raw.grad)

    def test_learned_soft_chevron_and_joint_baseline_solve_task(self):
        device = torch.device("cpu")
        config = TrainConfig(steps=400, batch_size=96, d_model=32)
        chevron = train_model(SOFT_CHEVRON, self.task, config, 7, device)
        joint = train_model(JOINT_ATTENTION, self.task, config, 7, device)
        chevron_metrics = evaluate(
            SOFT_CHEVRON, chevron, self.task, 7, device, batches=8, batch_size=96
        )
        joint_metrics = evaluate(
            JOINT_ATTENTION, joint, self.task, 7, device, batches=8, batch_size=96
        )
        self.assertGreater(chevron_metrics["answer_accuracy"], 0.95)
        self.assertGreater(chevron_metrics["matched_accuracy"], 0.95)
        self.assertGreater(chevron_metrics["no_match_accuracy"], 0.95)
        self.assertGreater(chevron_metrics["group_accuracy"], 0.95)
        self.assertGreater(chevron_metrics["target_r"], 0.90)
        self.assertLess(chevron_metrics["decoy_r"], 0.10)
        self.assertGreater(joint_metrics["answer_accuracy"], 0.95)

    def test_soft_chevron_can_learn_from_answers_without_auxiliary_losses(self):
        device = torch.device("cpu")
        config = replace(
            TrainConfig(steps=450, batch_size=96, d_model=32),
            retrieval_weight=0.0,
            gate_weight=0.0,
        )
        model = train_model(SOFT_CHEVRON, self.task, config, 13, device)
        result = evaluate(
            SOFT_CHEVRON, model, self.task, 13, device, batches=8, batch_size=96
        )
        self.assertGreater(result["answer_accuracy"], 0.90)
        self.assertGreater(result["no_match_accuracy"], 0.95)

    def test_gate_initialization_is_configurable(self):
        config = TrainConfig(theta_init=0.30, sharpness_init=5.0, d_model=16)
        model = build_model(SOFT_CHEVRON, self.task.config, config)
        generator = torch.Generator().manual_seed(31)
        outputs = model(self.task.batch(4, generator))
        self.assertAlmostEqual(float(outputs["theta"].item()), 0.30, places=5)
        self.assertAlmostEqual(float(outputs["sharpness"].item()), 5.0, places=5)

    def test_random_matching_initializations_are_distinct(self):
        shared_config = TrainConfig(match_init="shared_random", d_model=16)
        independent_config = TrainConfig(match_init="independent_random", d_model=16)
        torch.manual_seed(41)
        shared = build_model(SOFT_CHEVRON, self.task.config, shared_config)
        torch.manual_seed(41)
        independent = build_model(SOFT_CHEVRON, self.task.config, independent_config)
        self.assertTrue(torch.allclose(shared.match_a.weight, shared.match_n.weight))
        self.assertFalse(
            torch.allclose(independent.match_a.weight, independent.match_n.weight)
        )

    def test_current_match_noise_does_not_change_retained_templates(self):
        clean_task = CategoryMatchingTask(replace(self.task.config, match_noise=0.0))
        noisy_task = CategoryMatchingTask(replace(self.task.config, match_noise=0.20))
        clean = clean_task.batch(16, torch.Generator().manual_seed(53))
        noisy = noisy_task.batch(16, torch.Generator().manual_seed(53))
        self.assertTrue(torch.equal(clean.templates_n, noisy.templates_n))
        self.assertFalse(torch.equal(clean.match_a, noisy.match_a))

    def test_persistent_forward_masks_unused_slots(self):
        model = build_model(SOFT_CHEVRON, self.task.config, TrainConfig(d_model=16))
        stream = ContinualCategories(self.task.config, 61)
        memory = stream.initial_memory()
        query, match = stream.sample(stream.base_labels[0], 0.0)
        batch, active = memory.batch(query, match)
        outputs = forward_memory(SOFT_CHEVRON, model, batch, active)
        inactive = ~active
        self.assertTrue(torch.equal(outputs["slot_mass"][0, inactive], torch.zeros(inactive.sum())))
        self.assertTrue(
            torch.allclose(
                outputs["slot_mass"].sum(-1) + outputs["null_mass"],
                torch.ones(1),
                atol=1e-6,
            )
        )

    def test_persistent_write_respects_slot_mass(self):
        stream = ContinualCategories(self.task.config, 67)
        memory = stream.initial_memory()
        original_first = memory.slots[0].template_n.clone()
        original_second = memory.slots[1].template_n.clone()
        _query, match = stream.sample(stream.novel_labels[0], 0.0)
        weights = torch.zeros(self.task.config.num_slots)
        weights[0] = 0.5
        memory.write(memory.slots[0].key_a, match, weights, ContinualConfig())
        self.assertFalse(torch.equal(memory.slots[0].template_n, original_first))
        self.assertTrue(torch.equal(memory.slots[1].template_n, original_second))

    def test_joint_write_controller_is_parameter_matched(self):
        parameters = sum(
            value.numel() for value in JointWriteController().parameters()
        )
        self.assertEqual(parameters, 177)

    def test_controller_masks_keep_matched_target_active(self):
        generator = torch.Generator().manual_seed(71)
        batch = self.task.batch(64, generator)
        active = random_active_mask(batch, 4, generator)
        self.assertTrue(torch.all(active.sum(-1) >= 4))
        rows = torch.arange(batch.answers.size(0))[batch.matched]
        self.assertTrue(torch.all(active[rows, batch.target_slots[batch.matched]]))


if __name__ == "__main__":
    unittest.main()
