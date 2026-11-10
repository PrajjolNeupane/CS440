"""Value iteration, the checker used to judge the learned policy (it reads T and R)."""

from __future__ import annotations

import unittest

from qlearning.environment import DOWN, GOAL_STATE, PENALTY_STATE, RIGHT, UP, VacuumEnvironment
from qlearning.value_iteration import (
    compute_optimal_policy,
    compute_optimal_state_values,
    compute_policy_losses,
)

LIVING_REWARD = -0.04
DISCOUNT_FACTOR = 0.9


class ValueIterationTests(unittest.TestCase):
    def test_terminal_values_equal_their_exit_rewards(self) -> None:
        state_values = compute_optimal_state_values(
            VacuumEnvironment(LIVING_REWARD, seed=0), DISCOUNT_FACTOR
        )
        self.assertAlmostEqual(state_values[GOAL_STATE], 10.0)
        self.assertAlmostEqual(state_values[PENALTY_STATE], -1.0)

    def test_values_without_slips_match_a_hand_calculation(self) -> None:
        environment = VacuumEnvironment(LIVING_REWARD, slip_probability=0.0, seed=0)
        state_values = compute_optimal_state_values(environment, DISCOUNT_FACTOR)
        one_step_from_goal = LIVING_REWARD + DISCOUNT_FACTOR * 10.0
        two_steps_from_goal = LIVING_REWARD + DISCOUNT_FACTOR * one_step_from_goal
        self.assertAlmostEqual(state_values[(3, 2)], one_step_from_goal)
        self.assertAlmostEqual(state_values[(2, 3)], one_step_from_goal)
        self.assertAlmostEqual(state_values[(2, 2)], two_steps_from_goal)

    def test_optimal_policy_heads_for_the_goal_and_away_from_the_penalty(self) -> None:
        policy = compute_optimal_policy(VacuumEnvironment(LIVING_REWARD, seed=0), DISCOUNT_FACTOR)
        self.assertEqual(policy[(3, 2)], RIGHT)
        self.assertEqual(policy[(2, 3)], DOWN)
        self.assertEqual(policy[(3, 0)], UP)

    def test_optimal_policy_loses_nothing(self) -> None:
        environment = VacuumEnvironment(LIVING_REWARD, seed=0)
        policy = compute_optimal_policy(environment, DISCOUNT_FACTOR)
        losses = compute_policy_losses(environment, policy, DISCOUNT_FACTOR)
        self.assertLess(max(losses.values()), 1e-9)

    def test_a_bad_action_has_a_positive_loss(self) -> None:
        environment = VacuumEnvironment(LIVING_REWARD, seed=0)
        policy = compute_optimal_policy(environment, DISCOUNT_FACTOR)
        policy[(2, 1)] = DOWN  # straight into the -1 cell
        losses = compute_policy_losses(environment, policy, DISCOUNT_FACTOR)
        self.assertGreater(losses[(2, 1)], 1.0)


if __name__ == "__main__":
    unittest.main()
