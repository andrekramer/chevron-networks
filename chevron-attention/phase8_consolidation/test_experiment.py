from __future__ import annotations

import unittest

import torch

from phase7_soft_chevron.continual_closure import JointWriteController, Prepared
from phase7_soft_chevron.continual_memory import ContinualCategories
from phase7_soft_chevron.experiment import (
    JOINT_ATTENTION,
    SOFT_CHEVRON,
    TaskConfig,
    TrainConfig,
    build_model,
)
from phase8_consolidation.experiment import (
    ORACLE,
    DevelopmentConfig,
    Scenario,
    center_probability,
    relative_threshold_evidence,
    run_method,
    scenario_labels,
    selected_development_config,
)
from phase8_consolidation.provisional_memory import ProvisionalConfig


class PhaseEightExperimentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.task = TaskConfig(
            a_dimension=8,
            n_dimension=6,
            num_groups=2,
            group_size=3,
            num_values=8,
        )
        self.stream = ContinualCategories(self.task, 211)

    def test_transient_repeats_one_pattern(self) -> None:
        scenario = Scenario("transient", "transient", 3)
        labels = scenario_labels(self.stream, scenario)
        self.assertEqual(labels, [self.stream.novel_labels[0]] * 3)

    def test_sustained_blocks_each_novel_category(self) -> None:
        scenario = Scenario("sustained", "sustained", 4)
        labels = scenario_labels(self.stream, scenario)
        expected = []
        for label in self.stream.novel_labels:
            expected.extend([label] * 4)
        self.assertEqual(labels, expected)

    def test_interleaved_sustained_cycles_categories(self) -> None:
        scenario = Scenario("interleaved", "sustained", 2, interleaved=True)
        labels = scenario_labels(self.stream, scenario)
        self.assertEqual(labels, list(self.stream.novel_labels) * 2)

    def test_centered_probability_preserves_threshold_and_order(self) -> None:
        self.assertAlmostEqual(center_probability(0.02, 0.02), 0.5, places=7)
        self.assertLess(center_probability(0.01, 0.02), 0.5)
        self.assertGreater(center_probability(0.04, 0.02), 0.5)

    def test_relative_threshold_evidence_maps_boundary_to_one(self) -> None:
        self.assertAlmostEqual(relative_threshold_evidence(0.02, 0.02), 1.0)
        self.assertLess(relative_threshold_evidence(0.01, 0.02), 1.0)
        self.assertEqual(relative_threshold_evidence(0.04, 0.02), 1.0)

    def test_confirmation_candidate_config_is_frozen(self) -> None:
        config = selected_development_config()
        self.assertEqual(config.consolidation_threshold, 0.20)
        self.assertEqual(config.minimum_support, 5)
        self.assertTrue(config.reject_retained_matches_on_entry)

    def test_temporal_oracle_obeys_minimum_support(self) -> None:
        soft = build_model(SOFT_CHEVRON, self.task, TrainConfig(d_model=16))
        joint = build_model(JOINT_ATTENTION, self.task, TrainConfig(d_model=16))
        prepared = Prepared(
            soft=soft,
            joint=joint,
            controller=JointWriteController(),
            thresholds={},
            calibration={},
        )
        provisional = ProvisionalConfig(minimum_support=3)
        development = DevelopmentConfig(
            final_per_category=1,
            memory_capacity=self.task.num_slots,
        )
        transient = run_method(
            ORACLE,
            prepared,
            self.task,
            Scenario("transient", "transient", 3, warmup_observations=0, recovery_observations=0),
            211,
            provisional,
            development,
        )
        sustained = run_method(
            ORACLE,
            prepared,
            self.task,
            Scenario("sustained", "sustained", 3, warmup_observations=0, recovery_observations=0),
            211,
            provisional,
            development,
        )
        self.assertEqual(transient["false_consolidations"], 0)
        self.assertEqual(transient["novel_categories_retained"], 0)
        self.assertEqual(sustained["novel_categories_retained"], 2)
        self.assertEqual(sustained["mean_acquisition_delay"], 3)


if __name__ == "__main__":
    unittest.main()
