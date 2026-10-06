"""Parser/statistic contract fixtures; no benchmark data or scientific results."""
import importlib.util
import math
import unittest
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "scripts/analyze_rollouts.py"
SPEC = importlib.util.spec_from_file_location("rollout_log_analyzer", SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LogContractTests(unittest.TestCase):
    def report(self, *records, **kwargs):
        return MODULE.summarize(enumerate(records, 1), **kwargs)

    def test_continuous_rewards_do_not_use_binary_formula(self):
        result = self.report({"rewards": [0.25, 0.75]})
        self.assertEqual(result["nonbinary_reward_groups"], 1)
        self.assertIsNone(result["mean_binary_zero_signal_plugin_prediction"])

    def test_external_binary_probability_and_missing_token_scope(self):
        result = self.report({"rewards": [0, 1], "token_counts": [3, 7]}, {"rewards": [1, 1]},
                             binary_success_probability=0.25)
        self.assertAlmostEqual(result["mean_binary_zero_signal_iid_prediction"], 0.25 ** 2 + 0.75 ** 2)
        self.assertAlmostEqual(result["token_coverage_fraction"], 0.5)
        self.assertIsNone(result["informative_token_fraction"])
        self.assertEqual(result["informative_token_fraction_observed_subset"], 1)

    def test_zero_token_count_is_not_missing(self):
        result = self.report({"rewards": [0, 0], "token_counts": [0, 0]})
        self.assertEqual(result["total_rollout_tokens"], 0)
        self.assertEqual(result["groups_missing_token_counts"], 0)
        self.assertIsNone(result["informative_token_fraction"])

    def test_empty_log_has_no_fabricated_statistic(self):
        result = self.report()
        self.assertEqual(result["groups"], 0)
        self.assertIsNone(result["zero_signal_fraction"])
        self.assertIsNone(result["mean_binary_zero_signal_plugin_prediction"])

    def test_invalid_groups_are_rejected(self):
        for rewards in ([], [1], [True, 0], [math.nan, 0], [math.inf, 1], ["1", 0]):
            with self.subTest(rewards=rewards), self.assertRaises(ValueError):
                self.report({"rewards": rewards})

    def test_invalid_token_counts_are_rejected(self):
        for tokens in ([1], [-1, 2], [True, 2], [1.5, 2]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                self.report({"rewards": [0, 1], "token_counts": tokens})


if __name__ == "__main__":
    unittest.main(verbosity=2)
