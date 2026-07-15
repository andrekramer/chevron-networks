import unittest

import torch

from phase6_category_learning.experiment import (
    CategoryStream,
    ChevronARTMemory,
    MemoryConfig,
    MLPConfig,
    TaskConfig,
    complement_code,
    contrast_components,
    metrics,
    run_method,
)
from phase6_category_learning.drift_comparison import (
    DriftConfig,
    DriftStream,
    DriftTaskConfig,
    metrics as drift_metrics,
    run_method as run_drift_method,
)
from phase6_category_learning.ambiguity_comparison import (
    AmbiguityConfig,
    evaluate as evaluate_ambiguity,
    generate_episode,
    mean_squared_distances,
    n_signals,
    predict as predict_ambiguity,
)


class PhaseSixTest(unittest.TestCase):
    def test_symmetric_smoothing_preserves_contrast_identity(self):
        a = torch.tensor([0.0, 0.2, 0.8, 1.0])
        n = torch.tensor([0.0, 0.6, 0.4, 1.0])
        epsilon = 1e-6
        contrast, gate = contrast_components(a, n, epsilon)
        p = (a + epsilon / 2.0) / (a + n + epsilon)
        self.assertTrue(torch.allclose(contrast, 2.0 * p - 1.0))
        self.assertTrue(torch.allclose(gate, 2.0 * torch.sqrt(p * (1.0 - p))))
        self.assertTrue(bool(((gate >= 0.0) & (gate <= 1.0)).all()))

    def test_short_candidate_is_not_retained_but_persistent_candidate_is(self):
        config = MemoryConfig()
        learner = ChevronARTMemory(config)
        x = torch.full((16,), 0.25)
        for _ in range(5):
            prediction = learner.predict(x)
            learner.observe(x, 3, prediction)
        self.assertNotIn(3, learner.retained_labels)
        self.assertIsNotNone(learner.candidate)

        y = torch.full((16,), 0.75)
        for _ in range(10):
            prediction = learner.predict(y)
            learner.observe(y, 4, prediction)
        self.assertIn(4, learner.retained_labels)
        self.assertNotIn(3, learner.retained_labels)

    def test_search_can_reset_first_choice_and_find_later_template(self):
        config = MemoryConfig(retained_slots=3)
        learner = ChevronARTMemory(config)
        x0 = torch.tensor([0.2] * 16)
        x1 = torch.tensor([0.8] * 16)
        for label, x in ((0, x0), (1, x1)):
            for _ in range(10):
                prediction = learner.predict(x)
                learner.observe(x, label, prediction)
        # Make slot zero's fast A key win attention while retaining a mismatched
        # N template. Vigilance must veto it and continue to the second slot.
        query = complement_code(x1)
        learner.slots[0].a = query.clone()
        prediction = learner.predict(x1)
        self.assertEqual(prediction.label, 1)
        self.assertGreaterEqual(prediction.resets, 1)

    def test_seeded_stream_shows_category_retention_advantage(self):
        task = TaskConfig()
        memory = MemoryConfig()
        stream = CategoryStream(task, 7).build()
        chevron = metrics(
            run_method("chevron_art", stream, task, memory, MLPConfig(), 7), task
        )
        attention = metrics(
            run_method("standard_attention", stream, task, memory, MLPConfig(), 7), task
        )
        persistent_attention = metrics(
            run_method("persistent_attention", stream, task, memory, MLPConfig(), 7), task
        )
        self.assertEqual(chevron["base_categories_retained"], 3.0)
        self.assertEqual(chevron["transient_categories_retained"], 0.0)
        self.assertEqual(chevron["persistent_category_retained"], 1.0)
        self.assertGreater(chevron["old_retention_after_transients"], 0.95)
        self.assertGreater(
            chevron["old_retention_after_transients"],
            attention["old_retention_after_transients"] + 0.20,
        )
        self.assertGreater(persistent_attention["old_retention_after_transients"], 0.95)

    def test_dual_trace_recovers_base_and_adapts_to_persistent_drift(self):
        task = DriftTaskConfig()
        config = DriftConfig()
        stream = DriftStream(task, 7).build()
        dual = drift_metrics(run_drift_method("chevron_dual", stream, task, config))
        fast = drift_metrics(run_drift_method("fast_single", stream, task, config))
        persistent = drift_metrics(
            run_drift_method("persistent_single", stream, task, config)
        )
        self.assertGreater(dual["short_current_probe"], 0.95)
        self.assertGreater(dual["base_recovery_probe"], 0.95)
        self.assertGreater(dual["final_shift_probe"], 0.95)
        self.assertGreater(dual["retained_shift_probe"], 0.95)
        self.assertGreater(
            dual["base_recovery_probe"], fast["base_recovery_probe"] + 0.5
        )
        self.assertGreater(
            dual["short_current_probe"], persistent["short_current_probe"] + 0.5
        )

    def test_vigilance_rejects_bottom_up_winner_and_finds_context_match(self):
        config = AmbiguityConfig()
        import random

        generator = torch.Generator().manual_seed(91)
        episode = generate_episode(config, 5, random.Random(81), generator)
        distances = mean_squared_distances(episode.query_a, episode.keys_a)
        self.assertNotEqual(int(distances.argmin().item()), episode.target_slot)
        mismatch, _gate = n_signals(episode)
        self.assertLess(float(mismatch[episode.target_slot].item()), config.vigilance)
        self.assertTrue(
            all(float(mismatch[index].item()) > config.vigilance for index in episode.decoy_slots)
        )
        prediction, resets = predict_ambiguity("hard_search", episode, config)
        self.assertEqual(prediction, 1)
        self.assertGreaterEqual(resets, 1)

    def test_contextual_search_survives_many_decoys(self):
        config = AmbiguityConfig()
        hard = evaluate_ambiguity("hard_search", config, 31, 7, 100)
        top1 = evaluate_ambiguity("joint_top1", config, 31, 7, 100)
        masked = evaluate_ambiguity("masked_attention", config, 31, 7, 100)
        a_only = evaluate_ambiguity("a_softmax", config, 31, 7, 100)
        self.assertGreater(hard["accuracy"], 0.98)
        self.assertGreater(top1["accuracy"], 0.98)
        self.assertGreater(masked["accuracy"], 0.98)
        self.assertLess(a_only["accuracy"], 0.05)


if __name__ == "__main__":
    unittest.main()
