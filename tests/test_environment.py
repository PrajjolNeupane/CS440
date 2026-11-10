"""Environment rules from .claude/CLAUDE.md: layout, moves, walls, edges, slips, rewards."""

from __future__ import annotations

import unittest
from collections import Counter

from qlearning.environment import (
    DOWN,
    EXIT,
    GOAL_STATE,
    LEFT,
    MOVE_ACTIONS,
    PENALTY_STATE,
    RIGHT,
    START_STATE,
    UP,
    VacuumEnvironment,
    move,
)

LIVING_REWARD = -0.04
OPPOSITE_ACTIONS = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}


def make_environment(slip_probability: float = 0.2, seed: int = 0) -> VacuumEnvironment:
    return VacuumEnvironment(
        living_reward=LIVING_REWARD, slip_probability=slip_probability, seed=seed
    )


class GridLayoutTests(unittest.TestCase):
    def test_start_is_top_left(self) -> None:
        self.assertEqual(START_STATE, (0, 0))
        self.assertEqual(make_environment().reset(), (0, 0))

    def test_every_cell_except_the_wall_is_open(self) -> None:
        open_states = make_environment().open_states()
        self.assertEqual(len(open_states), 15)
        self.assertNotIn((1, 1), open_states)

    def test_terminal_cells_only_allow_exit(self) -> None:
        environment = make_environment()
        self.assertEqual(environment.legal_actions(GOAL_STATE), (EXIT,))
        self.assertEqual(environment.legal_actions(PENALTY_STATE), (EXIT,))

    def test_other_cells_allow_the_four_moves(self) -> None:
        environment = make_environment()
        for state in [(0, 0), (2, 2), (3, 0), (3, 2)]:
            self.assertEqual(environment.legal_actions(state), MOVE_ACTIONS)

    def test_the_wall_has_no_legal_actions(self) -> None:
        with self.assertRaises(ValueError):
            make_environment().legal_actions((1, 1))


class MoveTests(unittest.TestCase):
    def test_each_direction_moves_one_cell(self) -> None:
        self.assertEqual(move((2, 2), UP), (1, 2))
        self.assertEqual(move((2, 2), DOWN), (3, 2))
        self.assertEqual(move((2, 2), LEFT), (2, 1))
        self.assertEqual(move((2, 2), RIGHT), (2, 3))

    def test_moving_into_wall_keeps_position(self) -> None:
        self.assertEqual(move((0, 1), DOWN), (0, 1))
        self.assertEqual(move((1, 0), RIGHT), (1, 0))
        self.assertEqual(move((2, 1), UP), (2, 1))
        self.assertEqual(move((1, 2), LEFT), (1, 2))

    def test_moving_off_grid_keeps_position(self) -> None:
        self.assertEqual(move((0, 0), UP), (0, 0))
        self.assertEqual(move((0, 0), LEFT), (0, 0))
        self.assertEqual(move((0, 3), RIGHT), (0, 3))
        self.assertEqual(move((3, 0), DOWN), (3, 0))


class StepTests(unittest.TestCase):
    def test_move_gives_living_reward_and_does_not_end_episode(self) -> None:
        environment = make_environment(slip_probability=0.0)
        environment.reset()
        self.assertEqual(environment.step(RIGHT), ((0, 1), LIVING_REWARD, False))

    def test_bumping_into_wall_still_gives_living_reward(self) -> None:
        environment = make_environment(slip_probability=0.0)
        environment.reset((0, 1))
        self.assertEqual(environment.step(DOWN), ((0, 1), LIVING_REWARD, False))

    def test_arriving_in_terminal_does_not_end_episode(self) -> None:
        environment = make_environment(slip_probability=0.0)
        environment.reset((2, 3))
        self.assertEqual(environment.step(DOWN), (GOAL_STATE, LIVING_REWARD, False))

    def test_exit_from_goal_gives_plus_ten_and_ends_episode(self) -> None:
        environment = make_environment()
        environment.reset(GOAL_STATE)
        self.assertEqual(environment.step(EXIT), (GOAL_STATE, 10.0, True))

    def test_exit_from_penalty_cell_gives_minus_one_and_ends_episode(self) -> None:
        environment = make_environment()
        environment.reset(PENALTY_STATE)
        self.assertEqual(environment.step(EXIT), (PENALTY_STATE, -1.0, True))

    def test_exit_is_not_legal_outside_terminal_cells(self) -> None:
        environment = make_environment()
        environment.reset()
        with self.assertRaises(ValueError):
            environment.step(EXIT)

    def test_moves_are_not_legal_in_terminal_cells(self) -> None:
        environment = make_environment()
        environment.reset(GOAL_STATE)
        with self.assertRaises(ValueError):
            environment.step(UP)

    def test_step_after_episode_is_over_raises(self) -> None:
        environment = make_environment()
        environment.reset(GOAL_STATE)
        environment.step(EXIT)
        with self.assertRaises(RuntimeError):
            environment.step(EXIT)

    def test_reset_starts_a_new_episode(self) -> None:
        environment = make_environment(slip_probability=0.0)
        environment.reset(GOAL_STATE)
        environment.step(EXIT)
        self.assertEqual(environment.reset(), START_STATE)
        self.assertEqual(environment.step(DOWN), ((1, 0), LIVING_REWARD, False))


class SlipTests(unittest.TestCase):
    def test_each_move_slips_to_the_two_perpendicular_directions(self) -> None:
        environment = make_environment()
        expected_outcomes = {
            UP: {(1, 2): 0.8, (2, 1): 0.1, (2, 3): 0.1},
            DOWN: {(3, 2): 0.8, (2, 1): 0.1, (2, 3): 0.1},
            LEFT: {(2, 1): 0.8, (1, 2): 0.1, (3, 2): 0.1},
            RIGHT: {(2, 3): 0.8, (1, 2): 0.1, (3, 2): 0.1},
        }
        for action, expected in expected_outcomes.items():
            outcomes = environment.transition_outcomes((2, 2), action)
            actual = {outcome.next_state: outcome.probability for outcome in outcomes}
            self.assertEqual(actual.keys(), expected.keys(), action)
            for next_state, probability in expected.items():
                self.assertAlmostEqual(actual[next_state], probability, msg=action)

    def test_slips_never_go_backwards(self) -> None:
        environment = make_environment()
        for action in MOVE_ACTIONS:
            backwards_cell = move((2, 2), OPPOSITE_ACTIONS[action])
            next_states = [
                outcome.next_state for outcome in environment.transition_outcomes((2, 2), action)
            ]
            self.assertNotIn(backwards_cell, next_states, action)

    def test_slip_into_the_wall_keeps_position(self) -> None:
        environment = make_environment()
        # From (2, 1), `left` can slip up into the wall at (1, 1), which leaves the robot in place
        outcomes = environment.transition_outcomes((2, 1), LEFT)
        landing_probabilities = {outcome.next_state: outcome.probability for outcome in outcomes}
        self.assertAlmostEqual(landing_probabilities[(2, 0)], 0.8)
        self.assertAlmostEqual(landing_probabilities[(2, 1)], 0.1)
        self.assertAlmostEqual(landing_probabilities[(3, 1)], 0.1)
        self.assertTrue(all(outcome.reward == LIVING_REWARD for outcome in outcomes))

    def test_slip_off_the_grid_keeps_position(self) -> None:
        environment = make_environment()
        # From (0, 0), `up` and a slip to the left both leave the grid, so 90% stays put
        stay_probability = sum(
            outcome.probability
            for outcome in environment.transition_outcomes((0, 0), UP)
            if outcome.next_state == (0, 0)
        )
        self.assertAlmostEqual(stay_probability, 0.9)

    def test_unknown_action_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_environment().transition_outcomes((0, 0), "jump")

    def test_outcome_probabilities_add_up_to_one_everywhere(self) -> None:
        environment = make_environment()
        for state in environment.open_states():
            for action in environment.legal_actions(state):
                total = sum(
                    outcome.probability
                    for outcome in environment.transition_outcomes(state, action)
                )
                self.assertAlmostEqual(total, 1.0, msg=f"{state} {action}")

    def test_sampled_moves_follow_80_10_10(self) -> None:
        environment = make_environment(seed=7)
        trial_count = 10_000
        landing_counts: Counter[tuple[int, int]] = Counter()
        for _ in range(trial_count):
            environment.reset((2, 2))
            landing_counts[environment.step(UP).next_state] += 1
        self.assertAlmostEqual(landing_counts[(1, 2)] / trial_count, 0.8, delta=0.02)
        self.assertAlmostEqual(landing_counts[(2, 1)] / trial_count, 0.1, delta=0.02)
        self.assertAlmostEqual(landing_counts[(2, 3)] / trial_count, 0.1, delta=0.02)
        self.assertEqual(landing_counts[(3, 2)], 0)

    def test_same_seed_gives_same_slips(self) -> None:
        first_environment = make_environment(seed=42)
        second_environment = make_environment(seed=42)
        first_landings = []
        second_landings = []
        for _ in range(200):
            first_environment.reset((2, 2))
            second_environment.reset((2, 2))
            first_landings.append(first_environment.step(RIGHT).next_state)
            second_landings.append(second_environment.step(RIGHT).next_state)
        self.assertEqual(first_landings, second_landings)

    def test_slip_probability_outside_zero_to_one_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_environment(slip_probability=1.5)


if __name__ == "__main__":
    unittest.main()
