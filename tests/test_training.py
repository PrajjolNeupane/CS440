"""Schedules, the training loop, policy extraction, policy evaluation and the printed grids."""

from __future__ import annotations

import unittest

from qlearning.agent import QLearningAgent
from qlearning.environment import (
    EXIT,
    GOAL_STATE,
    PENALTY_STATE,
    RIGHT,
    START_STATE,
    TERMINAL_REWARDS,
    VacuumEnvironment,
)
from qlearning.training import (
    Hyperparameters,
    evaluate_policy,
    extract_policy,
    extract_state_values,
    format_policy_grid,
    format_value_grid,
    linear_schedule,
    run_training_episode,
    train,
)
from qlearning.value_iteration import compute_optimal_policy, compute_policy_losses

SHORT_RUN = Hyperparameters(episode_count=300, seed=5)


class LinearScheduleTests(unittest.TestCase):
    def test_first_episode_uses_the_start_value(self) -> None:
        self.assertAlmostEqual(linear_schedule(1.0, 0.1, 0, 100), 1.0)

    def test_last_episode_uses_the_end_value(self) -> None:
        self.assertAlmostEqual(linear_schedule(1.0, 0.1, 99, 100), 0.1)

    def test_middle_episode_is_halfway(self) -> None:
        self.assertAlmostEqual(linear_schedule(1.0, 0.0, 50, 101), 0.5)

    def test_equal_start_and_end_stays_constant(self) -> None:
        for episode_index in range(10):
            self.assertAlmostEqual(linear_schedule(0.3, 0.3, episode_index, 10), 0.3)

    def test_single_episode_uses_the_start_value(self) -> None:
        self.assertAlmostEqual(linear_schedule(0.8, 0.2, 0, 1), 0.8)


class HyperparameterValidationTests(unittest.TestCase):
    def test_rates_outside_zero_to_one_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Hyperparameters(discount_factor=1.5)
        with self.assertRaises(ValueError):
            Hyperparameters(exploration_rate_end=-0.1)

    def test_episode_count_must_be_at_least_one(self) -> None:
        with self.assertRaises(ValueError):
            Hyperparameters(episode_count=0)


class TrainingEpisodeTests(unittest.TestCase):
    def test_episode_ends_on_exit_not_on_arrival(self) -> None:
        environment = VacuumEnvironment(living_reward=-0.04, slip_probability=0.0, seed=0)
        # A frozen agent (no learning, no exploration) that follows the optimal policy
        agent = QLearningAgent(learning_rate=0.0, discount_factor=0.9, exploration_rate=0.0, seed=0)
        for state, action in compute_optimal_policy(environment, discount_factor=0.9).items():
            agent.q_values[(state, action)] = 1.0

        total_reward, step_count = run_training_episode(environment, agent, max_steps=200)

        # 6 moves to reach (3, 3) and a 7th step to exit
        self.assertEqual(step_count, 7)
        self.assertAlmostEqual(total_reward, 6 * -0.04 + 10.0)

    def test_episode_stops_at_max_steps(self) -> None:
        environment = VacuumEnvironment(living_reward=-0.04, seed=0)
        agent = QLearningAgent(learning_rate=0.5, discount_factor=0.9, exploration_rate=1.0, seed=0)
        # The goal is at least 7 steps away, so 3 steps can never finish an episode
        total_reward, step_count = run_training_episode(environment, agent, max_steps=3)
        self.assertEqual(step_count, 3)
        self.assertAlmostEqual(total_reward, 3 * -0.04)


class TrainingTests(unittest.TestCase):
    def test_same_seed_gives_the_same_training_run(self) -> None:
        first_result = train(SHORT_RUN)
        second_result = train(SHORT_RUN)
        self.assertEqual(first_result.agent.q_values, second_result.agent.q_values)
        self.assertEqual(first_result.episode_returns, second_result.episode_returns)

    def test_different_seeds_give_different_training_runs(self) -> None:
        first_result = train(SHORT_RUN)
        second_result = train(Hyperparameters(episode_count=300, seed=6))
        self.assertNotEqual(first_result.episode_returns, second_result.episode_returns)

    def test_one_return_per_episode(self) -> None:
        result = train(SHORT_RUN)
        self.assertEqual(len(result.episode_returns), 300)

    def test_schedules_end_at_their_end_values(self) -> None:
        result = train(SHORT_RUN)
        self.assertAlmostEqual(result.agent.learning_rate, SHORT_RUN.learning_rate_end)
        self.assertAlmostEqual(result.agent.exploration_rate, SHORT_RUN.exploration_rate_end)

    def test_policy_has_an_action_for_every_open_cell(self) -> None:
        result = train(SHORT_RUN)
        policy = extract_policy(result.agent, result.environment)
        self.assertEqual(set(policy), set(result.environment.open_states()))
        self.assertEqual(policy[GOAL_STATE], EXIT)
        self.assertEqual(policy[PENALTY_STATE], EXIT)

    def test_state_values_are_the_best_q_values(self) -> None:
        result = train(SHORT_RUN)
        state_values = extract_state_values(result.agent, result.environment)
        for state, value in state_values.items():
            legal_actions = result.environment.legal_actions(state)
            best_q_value = max(result.agent.get_q_value(state, action) for action in legal_actions)
            self.assertEqual(value, best_q_value)

class ChosenHyperparameterTests(unittest.TestCase):
    """Guards the values chosen in docs/hyperparameter-tuning.md. If a change to the code makes
    these fail, the chosen values no longer work as recorded: re-run the tuning skill rather than
    loosening the limits."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.hyperparameters = Hyperparameters()
        result = train(cls.hyperparameters)
        cls.environment = result.environment
        cls.policy = extract_policy(result.agent, result.environment)

    def test_default_hyperparameters_learn_a_near_optimal_policy(self) -> None:
        losses = compute_policy_losses(
            self.environment, self.policy, self.hyperparameters.discount_factor
        )
        suboptimal_cells = [state for state, loss in losses.items() if loss > 1e-6]
        self.assertLess(losses[START_STATE], 1e-6)
        self.assertLessEqual(len(suboptimal_cells), 1, suboptimal_cells)
        # The one near-tie seen in tuning, at (0, 2), costs under 0.01
        self.assertLess(max(losses.values()), 0.01)

    def test_learned_policy_always_reaches_the_goal(self) -> None:
        evaluation = evaluate_policy(
            self.policy, VacuumEnvironment(self.hyperparameters.living_reward, seed=1), 1000, 200
        )
        self.assertGreaterEqual(evaluation.goal_rate, 0.99)


class EvaluatePolicyTests(unittest.TestCase):
    def test_optimal_policy_without_slips_takes_the_shortest_path(self) -> None:
        environment = VacuumEnvironment(living_reward=-0.04, slip_probability=0.0, seed=0)
        policy = compute_optimal_policy(environment, discount_factor=0.9)
        evaluation = evaluate_policy(policy, environment, episode_count=10, max_steps=200)
        # 6 moves to reach (3, 3), then exit
        self.assertEqual(evaluation.goal_rate, 1.0)
        self.assertAlmostEqual(evaluation.average_step_count, 7.0)
        self.assertAlmostEqual(evaluation.average_return, 6 * -0.04 + 10.0)


class GridFormattingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.open_states = VacuumEnvironment(living_reward=-0.04, seed=0).open_states()

    def test_policy_grid_shows_arrows_the_wall_and_terminal_rewards(self) -> None:
        policy = {
            state: EXIT if state in TERMINAL_REWARDS else RIGHT for state in self.open_states
        }
        grid_lines = format_policy_grid(policy).splitlines()
        self.assertEqual(len(grid_lines), 4)
        self.assertEqual(grid_lines[0].split(), ["→", "→", "→", "→"])
        self.assertEqual(grid_lines[1].split(), ["→", "X", "→", "→"])
        self.assertEqual(grid_lines[3].split(), ["→", "-1", "→", "+10"])

    def test_value_grid_shows_two_decimals_and_the_wall(self) -> None:
        state_values = {state: 1.234 for state in self.open_states}
        grid_lines = format_value_grid(state_values).splitlines()
        self.assertEqual(len(grid_lines), 4)
        self.assertEqual(grid_lines[1].split(), ["1.23", "X", "1.23", "1.23"])


if __name__ == "__main__":
    unittest.main()
