"""The hyperparameter sweep used by the tune-hyperparameters skill."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from qlearning.sweep import (
    SweepRow,
    build_grid,
    evaluate_configuration,
    format_table,
    main,
    parse_number_list,
    rank_rows,
)
from qlearning.training import Hyperparameters

TINY_RUN = Hyperparameters(episode_count=100)


def make_row(mean_optimal: float, worst: int, value_error: float) -> SweepRow:
    return SweepRow(
        hyperparameters=Hyperparameters(),
        mean_optimal_cell_count=mean_optimal,
        worst_optimal_cell_count=worst,
        mean_start_value_error=value_error,
        mean_goal_rate=1.0,
        mean_average_return=9.0,
    )


class ParseNumberListTests(unittest.TestCase):
    def test_reads_comma_separated_floats(self) -> None:
        self.assertEqual(parse_number_list("0.1, 0.5,1", float), [0.1, 0.5, 1.0])

    def test_reads_integers(self) -> None:
        self.assertEqual(parse_number_list("1000,5000", int), [1000, 5000])

    def test_rejects_text_that_is_not_a_number(self) -> None:
        with self.assertRaises(ValueError):
            parse_number_list("0.1,fast", float)

    def test_rejects_an_empty_list(self) -> None:
        with self.assertRaises(ValueError):
            parse_number_list(" , ", float)


class BuildGridTests(unittest.TestCase):
    def test_makes_every_combination(self) -> None:
        grid = build_grid(
            Hyperparameters(),
            {"discount_factor": [0.9, 0.99], "learning_rate_start": [0.1, 0.5, 1.0]},
        )
        self.assertEqual(len(grid), 6)
        self.assertIn(Hyperparameters(discount_factor=0.99, learning_rate_start=0.1), grid)

    def test_unlisted_fields_keep_their_base_value(self) -> None:
        base = Hyperparameters(living_reward=-0.1)
        grid = build_grid(base, {"discount_factor": [0.5]})
        self.assertEqual(grid, [Hyperparameters(living_reward=-0.1, discount_factor=0.5)])

    def test_no_lists_gives_only_the_base(self) -> None:
        self.assertEqual(build_grid(Hyperparameters(), {}), [Hyperparameters()])


class EvaluateConfigurationTests(unittest.TestCase):
    def test_scores_are_in_range(self) -> None:
        row = evaluate_configuration(TINY_RUN, seed_count=2, evaluation_episode_count=20)
        self.assertTrue(0 <= row.worst_optimal_cell_count <= row.mean_optimal_cell_count <= 15)
        self.assertTrue(0.0 <= row.mean_goal_rate <= 1.0)
        self.assertGreaterEqual(row.mean_start_value_error, 0.0)

    def test_same_seeds_give_the_same_scores(self) -> None:
        first_row = evaluate_configuration(TINY_RUN, seed_count=2, evaluation_episode_count=20)
        second_row = evaluate_configuration(TINY_RUN, seed_count=2, evaluation_episode_count=20)
        self.assertEqual(first_row, second_row)


class RankingAndTableTests(unittest.TestCase):
    def test_ranking_prefers_more_optimal_cells_then_worst_seed_then_value_error(self) -> None:
        rows = [make_row(14.0, 13, 0.01), make_row(14.5, 13, 0.2), make_row(14.0, 14, 0.3)]
        ranked = rank_rows(rows)
        self.assertEqual(
            [(row.mean_optimal_cell_count, row.worst_optimal_cell_count) for row in ranked],
            [(14.5, 13), (14.0, 14), (14.0, 13)],
        )

    def test_table_has_a_column_for_each_varied_field(self) -> None:
        table = format_table([make_row(14.0, 13, 0.01)], ["discount_factor"])
        header_line = table.splitlines()[0]
        self.assertIn("discount_factor", header_line)
        self.assertIn("optimal cells", header_line)
        self.assertNotIn("living_reward", header_line)

    def test_table_is_markdown_with_one_line_per_row(self) -> None:
        table = format_table([make_row(14.0, 13, 0.01), make_row(13.0, 12, 0.02)], [])
        table_lines = table.splitlines()
        self.assertEqual(len(table_lines), 4)
        self.assertEqual(table_lines[1], "|---|---|---|---|---|")
        self.assertEqual(table_lines[2], "| 14.0/15 | 13/15 | 0.010 | 100.0% | 9.000 |")


class SweepCommandLineTests(unittest.TestCase):
    def run_main(self, command_line_arguments: list[str]) -> str:
        printed_output = io.StringIO()
        with redirect_stdout(printed_output):
            self.assertEqual(main(command_line_arguments), 0)
        return printed_output.getvalue()

    def test_sweep_prints_one_row_per_combination(self) -> None:
        output = self.run_main(
            ["--learning-rate-start", "0.2,0.8", "--episode-count", "50", "--seeds", "2",
             "--evaluation-episodes", "10"]
        )
        self.assertIn("2 combination(s) x 2 seeds", output)
        table_lines = output.strip().splitlines()[1:]
        self.assertEqual(len(table_lines), 4)  # header, separator and two rows

    def test_optimal_policies_mode_prints_a_grid_per_living_reward(self) -> None:
        output = self.run_main(
            ["--optimal-policies", "--living-reward", "0,-0.5", "--discount-factor", "0.9",
             "--evaluation-episodes", "10"]
        )
        self.assertIn("living_reward 0.0, discount_factor 0.9", output)
        self.assertIn("living_reward -0.5, discount_factor 0.9", output)
        self.assertEqual(output.count("+10\n"), 2)


if __name__ == "__main__":
    unittest.main()
