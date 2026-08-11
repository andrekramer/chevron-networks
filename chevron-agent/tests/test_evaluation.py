import unittest

from chevron_agent.train import is_successful_terminal


class EvaluationTests(unittest.TestCase):
    def test_only_positive_terminal_outcomes_count_as_success(self) -> None:
        self.assertTrue(is_successful_terminal(True, 0.998))
        self.assertFalse(is_successful_terminal(True, -1.002))
        self.assertFalse(is_successful_terminal(False, 0.018))
        self.assertFalse(is_successful_terminal(False, -0.002))


if __name__ == "__main__":
    unittest.main()
