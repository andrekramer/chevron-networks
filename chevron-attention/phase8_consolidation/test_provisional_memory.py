from __future__ import annotations

import unittest

import torch

from phase8_consolidation.provisional_memory import (
    ProvisionalConfig,
    ProvisionalMemory,
    ResidualEvidence,
    normalized_mismatch,
    residual_evidence,
)


class ProvisionalMemoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.key = torch.tensor([0.2, 0.4, 0.6, 0.8])
        self.old = torch.tensor([0.1, 0.1, 0.9, 0.9, 0.1, 0.9])
        self.novel = torch.tensor([0.9, 0.1, 0.9, 0.9, 0.1, 0.9])
        self.strong = ResidualEvidence(1.0, 1.0, 1.0, 0.0, 0.0)

    def test_residual_mass_is_not_novelty(self) -> None:
        alpha = torch.tensor([0.34, 0.33, 0.33])
        matched = residual_evidence(alpha, torch.tensor([1.0, 0.0, 0.0]))
        novel = residual_evidence(alpha, torch.tensor([0.1, 0.1, 0.1]))
        self.assertAlmostEqual(matched.remaining_mass, 0.66, places=5)
        self.assertAlmostEqual(matched.novelty, 0.0, places=5)
        self.assertAlmostEqual(matched.eligible_mass, 0.0, places=5)
        self.assertAlmostEqual(novel.remaining_mass, 0.90, places=5)
        self.assertAlmostEqual(novel.novelty, 0.90, places=5)
        self.assertAlmostEqual(novel.eligible_mass, 0.81, places=5)
        self.assertAlmostEqual(novel.admitted_mass + novel.remaining_mass, 1.0, places=6)

    def test_top_a_set_excludes_irrelevant_high_assent(self) -> None:
        result = residual_evidence(
            torch.tensor([0.49, 0.48, 0.02, 0.01]),
            torch.tensor([0.0, 0.0, 1.0, 1.0]),
            top_a_candidates=2,
        )
        self.assertGreater(result.novelty, 0.99)
        self.assertGreater(result.eligible_mass, 0.90)

    def test_isolated_contradiction_decays_without_consolidating(self) -> None:
        bank = ProvisionalMemory()
        event = bank.observe(self.key, self.novel, 2, self.strong, [self.old])
        self.assertEqual(event.kind, "created")
        self.assertEqual(bank.active_count, 1)
        for _ in range(12):
            bank.tick()
        self.assertEqual(bank.consolidations, 0)
        self.assertEqual(bank.active_count, 0)

    def test_short_coherent_disturbance_stays_provisional(self) -> None:
        bank = ProvisionalMemory()
        for _ in range(3):
            event = bank.observe(self.key, self.novel, 2, self.strong, [self.old])
        self.assertEqual(event.kind, "updated")
        self.assertEqual(bank.consolidations, 0)
        candidate = next(item for item in bank.candidates if item is not None)
        self.assertEqual(candidate.support, 3)
        self.assertLess(candidate.persistence, bank.config.consolidation_threshold)

    def test_sustained_novelty_consolidates_after_delay(self) -> None:
        bank = ProvisionalMemory()
        event = None
        for observation in range(1, 8):
            event = bank.observe(self.key, self.novel, 2, self.strong, [self.old])
            if event.consolidation is not None:
                break
        assert event is not None
        self.assertEqual(event.kind, "consolidated")
        self.assertGreaterEqual(observation, bank.config.minimum_support)
        self.assertEqual(event.consolidation.value_id, 2)
        self.assertEqual(event.consolidation.purity, 1.0)
        self.assertEqual(bank.active_count, 0)

    def test_mature_non_distinct_candidate_is_rejected_and_cleared(self) -> None:
        bank = ProvisionalMemory(
            ProvisionalConfig(reject_retained_matches_on_entry=False)
        )
        event = None
        for _ in range(8):
            event = bank.observe(self.key, self.old, 0, self.strong, [self.old])
            if event.kind == "rejected_not_distinct":
                break
        assert event is not None
        self.assertEqual(event.kind, "rejected_not_distinct")
        self.assertEqual(bank.consolidations, 0)
        self.assertEqual(bank.retained_match_rejections, 1)
        self.assertEqual(bank.active_count, 0)

    def test_retained_match_is_rejected_before_candidate_allocation(self) -> None:
        bank = ProvisionalMemory()
        event = bank.observe(self.key, self.old, 0, self.strong, [self.old])
        self.assertEqual(event.kind, "rejected_retained_match")
        self.assertEqual(bank.active_count, 0)
        self.assertEqual(bank.retained_match_rejections, 1)

    def test_interleaved_novelty_uses_separate_candidates(self) -> None:
        config = ProvisionalConfig(consolidation_threshold=1.0, minimum_support=20)
        bank = ProvisionalMemory(config)
        other = torch.tensor([0.1, 0.9, 0.1, 0.1, 0.9, 0.1])
        for _ in range(2):
            bank.observe(self.key, self.novel, 2, self.strong, [self.old])
            bank.observe(1.0 - self.key, other, 5, self.strong, [self.old])
        candidates = [item for item in bank.candidates if item is not None]
        self.assertEqual(len(candidates), 2)
        self.assertEqual({item.value_id for item in candidates}, {2, 5})
        self.assertTrue(all(item.purity == 1.0 for item in candidates))

    def test_candidate_bank_has_fixed_capacity_and_replaces_weakest(self) -> None:
        config = ProvisionalConfig(capacity=2, consolidation_threshold=1.0)
        bank = ProvisionalMemory(config)
        first = torch.tensor([0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
        second = torch.tensor([0.1, 0.9, 0.1, 0.1, 0.1, 0.1])
        third = torch.tensor([0.1, 0.1, 0.9, 0.1, 0.1, 0.1])
        bank.observe(self.key, first, 1, self.strong)
        bank.observe(self.key, second, 2, self.strong)
        event = bank.observe(self.key, third, 3, self.strong)
        self.assertEqual(event.kind, "replaced")
        self.assertEqual(bank.active_count, 2)
        self.assertEqual(bank.replacements, 1)

    def test_normalized_mismatch_is_symmetric_and_scale_relative(self) -> None:
        left = torch.tensor([0.2, 0.8])
        right = torch.tensor([0.3, 0.7])
        self.assertAlmostEqual(
            normalized_mismatch(left, right), normalized_mismatch(right, left), places=7
        )
        self.assertLess(normalized_mismatch(10 * left, 10 * right), 0.11)


if __name__ == "__main__":
    unittest.main()
