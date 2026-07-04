import random
import unittest

import torch

from chevron_attention import (
    ACTIVE,
    REVOKED,
    RESTORED,
    ChevronAttention,
    RecallTask,
    losses,
)


class RecallTaskTest(unittest.TestCase):
    def setUp(self):
        self.task = RecallTask(num_keys=8, num_values=8, num_facts=3)

    def test_modes_have_expected_target_permission_and_answer(self):
        rng = random.Random(1)
        for mode, expected_permission in (
            (ACTIVE, 1.0),
            (REVOKED, 0.0),
            (RESTORED, 1.0),
        ):
            row = self.task._example(rng, mode)
            target_slot = row[3]
            self.assertEqual(row[4][target_slot], expected_permission)
            self.assertEqual(row[5] == self.task.idk_class, mode == REVOKED)

    def test_a_stream_masks_controls(self):
        a_tokens, n_tokens, *_ = self.task._example(random.Random(2), REVOKED)
        fact_end = 1 + 3 * self.task.num_facts
        self.assertTrue(all(token == self.task.PAD for token in a_tokens[fact_end:-2]))
        self.assertNotEqual(a_tokens, n_tokens)

    def test_forward_and_backward(self):
        batch = self.task.batch(6, random.Random(3))
        model = ChevronAttention(
            self.task,
            max_length=batch.n_tokens.size(1),
            d_model=16,
            nhead=4,
            num_layers=1,
        )
        outputs = model(batch)
        self.assertEqual(outputs["answer_logits"].shape, (6, 9))
        self.assertEqual(outputs["alpha"].shape, (6, 3))
        self.assertEqual(outputs["gates"].shape, (6, 3))
        objective = losses(outputs, batch, 1.0, 1.0)["total"]
        objective.backward()
        self.assertTrue(torch.isfinite(objective))


if __name__ == "__main__":
    unittest.main()

