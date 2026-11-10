"""The command line: python3 -m qlearning."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout

from qlearning.__main__ import build_argument_parser, main


class CommandLineTests(unittest.TestCase):
    def test_short_run_prints_the_full_report(self) -> None:
        printed_output = io.StringIO()
        with redirect_stdout(printed_output):
            exit_code = main(["--episode-count", "200", "--seed", "3"])
        report_text = printed_output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("episode_count: 200", report_text)
        self.assertIn("Learned policy", report_text)
        self.assertIn("Learned values", report_text)
        self.assertIn("Check against value iteration", report_text)

    def test_options_override_the_defaults(self) -> None:
        printed_output = io.StringIO()
        with redirect_stdout(printed_output):
            main(["--episode-count", "50", "--discount-factor", "0.5", "--living-reward", "-0.1"])
        report_text = printed_output.getvalue()
        self.assertIn("episode_count: 50", report_text)
        self.assertIn("discount_factor: 0.5", report_text)
        self.assertIn("living_reward: -0.1", report_text)

    def test_impossible_values_are_reported_as_a_usage_error(self) -> None:
        with redirect_stderr(io.StringIO()) as error_output, self.assertRaises(SystemExit):
            main(["--episode-count", "0"])
        self.assertIn("episode_count", error_output.getvalue())

    def test_every_hyperparameter_has_an_option(self) -> None:
        help_text = build_argument_parser().format_help()
        for option in ["--living-reward", "--discount-factor", "--learning-rate-start",
                       "--learning-rate-end", "--exploration-rate-start",
                       "--exploration-rate-end", "--episode-count", "--max-steps-per-episode",
                       "--slip-probability", "--seed"]:
            self.assertIn(option, help_text)


if __name__ == "__main__":
    unittest.main()
