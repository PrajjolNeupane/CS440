"""The Q-value update, epsilon-greedy action choice and policy extraction."""

from __future__ import annotations

import ast
import unittest
from collections import Counter
from pathlib import Path

import qlearning.agent
from qlearning.agent import Action, QLearningAgent
from qlearning.environment import DOWN, EXIT, LEFT, MOVE_ACTIONS, RIGHT, UP

STATE = (2, 2)
NEXT_STATE = (2, 3)
GOAL_STATE = (3, 3)
LIVING_REWARD = -0.04


def make_agent(
    learning_rate: float = 0.5,
    discount_factor: float = 0.9,
    exploration_rate: float = 0.0,
    seed: int = 0,
) -> QLearningAgent:
    return QLearningAgent(
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        exploration_rate=exploration_rate,
        seed=seed,
    )


class QValueUpdateTests(unittest.TestCase):
    def test_q_values_start_at_zero(self) -> None:
        agent = make_agent()
        for action in MOVE_ACTIONS:
            self.assertEqual(agent.get_q_value(STATE, action), 0.0)

    def test_non_terminal_update_uses_best_next_q_value(self) -> None:
        agent = make_agent(learning_rate=0.5, discount_factor=0.9)
        agent.q_values[(STATE, RIGHT)] = 1.0
        agent.q_values[(NEXT_STATE, UP)] = 2.0
        agent.q_values[(NEXT_STATE, DOWN)] = 5.0
        agent.q_values[(NEXT_STATE, LEFT)] = -1.0

        agent.update(STATE, RIGHT, LIVING_REWARD, NEXT_STATE, MOVE_ACTIONS, done=False)

        expected = 0.5 * 1.0 + 0.5 * (LIVING_REWARD + 0.9 * 5.0)
        self.assertAlmostEqual(agent.get_q_value(STATE, RIGHT), expected)

    def test_exit_update_has_no_future_term(self) -> None:
        agent = make_agent(learning_rate=0.5, discount_factor=0.9)
        agent.q_values[(GOAL_STATE, EXIT)] = 4.0

        agent.update(GOAL_STATE, EXIT, 10.0, GOAL_STATE, (EXIT,), done=True)

        # With a future term it would be 0.5 * 4 + 0.5 * (10 + 0.9 * 4) = 8.8
        self.assertAlmostEqual(agent.get_q_value(GOAL_STATE, EXIT), 0.5 * 4.0 + 0.5 * 10.0)

    def test_best_next_value_ignores_actions_that_are_not_legal(self) -> None:
        agent = make_agent(learning_rate=0.5, discount_factor=0.9)
        agent.q_values[(NEXT_STATE, EXIT)] = 100.0

        agent.update(STATE, RIGHT, LIVING_REWARD, NEXT_STATE, MOVE_ACTIONS, done=False)

        self.assertAlmostEqual(agent.get_q_value(STATE, RIGHT), 0.5 * LIVING_REWARD)

    def test_learning_rate_one_replaces_the_old_value(self) -> None:
        agent = make_agent(learning_rate=1.0)
        agent.q_values[(GOAL_STATE, EXIT)] = 3.0
        agent.update(GOAL_STATE, EXIT, 10.0, GOAL_STATE, (EXIT,), done=True)
        self.assertAlmostEqual(agent.get_q_value(GOAL_STATE, EXIT), 10.0)

    def test_learning_rate_zero_keeps_the_old_value(self) -> None:
        agent = make_agent(learning_rate=0.0)
        agent.q_values[(GOAL_STATE, EXIT)] = 3.0
        agent.update(GOAL_STATE, EXIT, 10.0, GOAL_STATE, (EXIT,), done=True)
        self.assertAlmostEqual(agent.get_q_value(GOAL_STATE, EXIT), 3.0)

    def test_repeated_updates_move_toward_the_sample(self) -> None:
        agent = make_agent(learning_rate=0.5)
        for _ in range(30):
            agent.update(GOAL_STATE, EXIT, 10.0, GOAL_STATE, (EXIT,), done=True)
        self.assertAlmostEqual(agent.get_q_value(GOAL_STATE, EXIT), 10.0, places=5)


class ActionChoiceTests(unittest.TestCase):
    def choose_many(
        self, agent: QLearningAgent, legal_actions: tuple[Action, ...]
    ) -> Counter[Action]:
        return Counter(agent.choose_action(STATE, legal_actions) for _ in range(2000))

    def test_exploration_rate_zero_always_picks_the_best_action(self) -> None:
        agent = make_agent(exploration_rate=0.0)
        agent.q_values[(STATE, LEFT)] = 1.0
        self.assertEqual(self.choose_many(agent, MOVE_ACTIONS), Counter({LEFT: 2000}))

    def test_exploration_rate_one_picks_legal_actions_evenly(self) -> None:
        agent = make_agent(exploration_rate=1.0)
        agent.q_values[(STATE, LEFT)] = 1.0
        choice_counts = self.choose_many(agent, MOVE_ACTIONS)
        self.assertEqual(set(choice_counts), set(MOVE_ACTIONS))
        for action in MOVE_ACTIONS:
            self.assertAlmostEqual(choice_counts[action] / 2000, 0.25, delta=0.04)

    def test_ties_are_broken_at_random(self) -> None:
        agent = make_agent(exploration_rate=0.0)
        agent.q_values[(STATE, UP)] = 1.0
        agent.q_values[(STATE, DOWN)] = 1.0
        choice_counts = self.choose_many(agent, MOVE_ACTIONS)
        self.assertEqual(set(choice_counts), {UP, DOWN})
        self.assertAlmostEqual(choice_counts[UP] / 2000, 0.5, delta=0.05)

    def test_only_legal_actions_are_chosen(self) -> None:
        agent = make_agent(exploration_rate=1.0)
        self.assertEqual(self.choose_many(agent, (EXIT,)), Counter({EXIT: 2000}))

    def test_same_seed_gives_same_choices(self) -> None:
        first_agent = make_agent(exploration_rate=0.5, seed=11)
        second_agent = make_agent(exploration_rate=0.5, seed=11)
        first_choices = [first_agent.choose_action(STATE, MOVE_ACTIONS) for _ in range(100)]
        second_choices = [second_agent.choose_action(STATE, MOVE_ACTIONS) for _ in range(100)]
        self.assertEqual(first_choices, second_choices)


class PolicyExtractionTests(unittest.TestCase):
    def test_greedy_action_picks_the_highest_q_value(self) -> None:
        agent = make_agent()
        agent.q_values[(STATE, UP)] = 0.5
        agent.q_values[(STATE, RIGHT)] = 2.0
        agent.q_values[(STATE, DOWN)] = -3.0
        self.assertEqual(agent.greedy_action(STATE, MOVE_ACTIONS), RIGHT)

    def test_greedy_action_breaks_ties_by_action_order(self) -> None:
        agent = make_agent()
        self.assertEqual(agent.greedy_action(STATE, MOVE_ACTIONS), MOVE_ACTIONS[0])
        self.assertEqual(agent.greedy_action(STATE, (RIGHT, LEFT)), RIGHT)

    def test_best_value_is_the_max_over_legal_actions(self) -> None:
        agent = make_agent()
        agent.q_values[(STATE, UP)] = -2.0
        agent.q_values[(STATE, LEFT)] = 3.0
        agent.q_values[(STATE, EXIT)] = 50.0
        self.assertEqual(agent.best_value(STATE, MOVE_ACTIONS), 3.0)


class SeparationTests(unittest.TestCase):
    def test_agent_imports_nothing_from_the_rest_of_the_project(self) -> None:
        agent_source = Path(qlearning.agent.__file__).read_text(encoding="utf-8")
        imported_modules = set()
        for node in ast.walk(ast.parse(agent_source)):
            if isinstance(node, ast.Import):
                imported_modules.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.add(node.module)
        project_imports = [name for name in imported_modules if name.startswith("qlearning")]
        self.assertEqual(project_imports, [])


if __name__ == "__main__":
    unittest.main()
