from __future__ import annotations

import unittest

import numpy as np

from chevron_agent.memory.delayed_context import Candidate, ChevronMemory, MemoryConfig, unit


class DelayedMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = MemoryConfig(capacity=4, buffer_capacity=4)

    def test_empty_memory_preserves_residual_mass(self) -> None:
        memory = ChevronMemory(4, 4, seed=0, config=self.config, use_buffer=True)
        trace = memory.read(unit(np.ones(4)), unit(np.ones(4)))
        self.assertEqual(trace.q, 1.0)
        self.assertEqual(trace.admitted_mass, 0.0)
        self.assertEqual(trace.z, 0.0)
        self.assertTrue(trace.allocation_trigger)

    def test_read_and_residual_mass_are_conserved(self) -> None:
        memory = ChevronMemory(4, 4, seed=0, config=self.config, use_buffer=True)
        rng = np.random.default_rng(2)
        for context_id in range(4):
            memory._allocate_slot(
                unit(rng.normal(size=4)),
                unit(rng.normal(size=4)),
                target=(-1.0 if context_id % 2 == 0 else 1.0),
                audit_context_id=context_id,
                promoted=False,
            )
        for _ in range(50):
            trace = memory.read(unit(rng.normal(size=4)), unit(rng.normal(size=4)))
            self.assertAlmostEqual(trace.admitted_mass + trace.q, 1.0, places=12)
            self.assertTrue(np.all((trace.alpha >= 0.0) & (trace.alpha <= 1.0)))
            self.assertTrue(np.all((trace.assent >= 0.0) & (trace.assent <= 1.0)))

    def test_assent_and_retrieval_are_causally_distinct(self) -> None:
        memory = ChevronMemory(4, 4, seed=0, config=self.config, use_buffer=True)
        address_a = unit(np.asarray([1.0, 0.2, 0.0, 0.0]))
        address_b = unit(np.asarray([0.0, 0.0, 0.2, 1.0]))
        diagnostic_a = unit(np.asarray([1.0, 0.0, 0.0, 0.0]))
        diagnostic_b = unit(np.asarray([0.0, 0.0, 0.0, 1.0]))
        memory._allocate_slot(address_a, diagnostic_a, -1.0, 0, promoted=False)
        memory._allocate_slot(address_b, diagnostic_b, 1.0, 1, promoted=False)

        base = memory.read(address_a, diagnostic_a)
        changed_content = memory.read(address_a, diagnostic_b)
        changed_address = memory.read(address_b, diagnostic_a)

        np.testing.assert_allclose(base.alpha, changed_content.alpha)
        self.assertGreater(float(np.max(np.abs(base.assent - changed_content.assent))), 0.5)
        np.testing.assert_allclose(base.mismatch, changed_address.mismatch)
        self.assertGreater(float(np.max(np.abs(base.alpha - changed_address.alpha))), 0.5)

    def test_buffer_does_not_write_before_outcome(self) -> None:
        memory = ChevronMemory(4, 4, seed=0, config=self.config, use_buffer=True)
        address = unit(np.ones(4))
        diagnostic = unit(np.asarray([1.0, 0.0, 0.0, 0.0]))
        memory.act(address, diagnostic, 0, 0)
        self.assertEqual(len(memory.slots), 0)
        self.assertEqual(memory.premature_writes, 0)

        memory.observe(1.0)
        self.assertEqual(len(memory.slots), 0)
        self.assertEqual(len(memory.candidates), 1)

    def test_immediate_ablation_writes_before_outcome(self) -> None:
        memory = ChevronMemory(
            4,
            4,
            seed=0,
            config=self.config,
            use_buffer=False,
            immediate_write=True,
        )
        memory.act(unit(np.ones(4)), unit(np.ones(4)), 0, 0)
        self.assertEqual(len(memory.slots), 1)
        self.assertEqual(memory.premature_writes, 1)

    def test_write_threshold_must_be_stricter_than_read_threshold(self) -> None:
        with self.assertRaises(ValueError):
            MemoryConfig(theta_read=0.1, theta_write=0.1)

    def test_per_slot_residual_affects_candidate_association_only_in_full_mode(self) -> None:
        full = ChevronMemory(4, 4, seed=0, config=self.config, use_buffer=True)
        scalar = ChevronMemory(
            4,
            4,
            seed=0,
            config=self.config,
            use_buffer=True,
            per_slot_residual=False,
        )
        address = unit(np.ones(4))
        diagnostic = unit(np.asarray([1.0, 0.0, 0.0, 0.0]))
        candidate = Candidate(
            address=address,
            diagnostic=diagnostic,
            value_sum=1.0,
            support=1,
            last_seen=0,
            residual_signature=np.asarray([1.0, 0.0, 0.0, 0.0]),
        )
        orthogonal_residual = np.asarray([0.0, 1.0, 0.0, 0.0])
        full_score = full._candidate_similarity(
            address,
            diagnostic,
            candidate,
            orthogonal_residual,
        )
        scalar_score = scalar._candidate_similarity(
            address,
            diagnostic,
            candidate,
            orthogonal_residual,
        )
        self.assertAlmostEqual(full_score, 0.8)
        self.assertAlmostEqual(scalar_score, 1.0)


if __name__ == "__main__":
    unittest.main()
