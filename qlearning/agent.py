"""The Q-learning agent (RL slides L1 p33, L2 p5).

The agent learns only from the samples it is handed after each step (state, action, reward,
next state, done) and the list of legal actions for a cell. It has no access to the grid,
T or R.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

# The agent's own names for the types it stores. It imports nothing from the rest of the project,
# so it cannot read the grid, T or R even by accident.
State = tuple[int, int]
Action = str


class QLearningAgent:
    """Keeps a table of Q-values and picks actions epsilon-greedily."""

    def __init__(
        self,
        learning_rate: float,
        discount_factor: float,
        exploration_rate: float,
        seed: int | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        # Every Q-value starts at 0.0, so missing entries are read as 0.0
        self.q_values: dict[tuple[State, Action], float] = {}
        self._random_generator = random.Random(seed)

    def get_q_value(self, state: State, action: Action) -> float:
        """Q(state, action), which is 0.0 until the pair has been updated."""
        return self.q_values.get((state, action), 0.0)

    def best_value(self, state: State, legal_actions: Sequence[Action]) -> float:
        """max over a of Q(state, a), looking only at the legal actions."""
        return max(self.get_q_value(state, action) for action in legal_actions)

    def best_actions(self, state: State, legal_actions: Sequence[Action]) -> list[Action]:
        """Every legal action whose Q-value is the highest, in the order given."""
        highest_value = self.best_value(state, legal_actions)
        return [
            action for action in legal_actions if self.get_q_value(state, action) == highest_value
        ]

    def choose_action(self, state: State, legal_actions: Sequence[Action]) -> Action:
        """Epsilon-greedy: a random legal action with probability exploration_rate, otherwise
        a best action. Ties between best actions are broken at random."""
        if self._random_generator.random() < self.exploration_rate:
            return self._random_generator.choice(list(legal_actions))
        return self._random_generator.choice(self.best_actions(state, legal_actions))

    def greedy_action(self, state: State, legal_actions: Sequence[Action]) -> Action:
        """The learned policy: pi(state) = argmax over a of Q(state, a).

        Ties go to the first action in legal_actions, so the printed policy does not change
        from one call to the next.
        """
        return self.best_actions(state, legal_actions)[0]

    def update(
        self,
        state: State,
        action: Action,
        reward: float,
        next_state: State,
        next_legal_actions: Sequence[Action],
        done: bool,
    ) -> None:
        """One Q-learning update from a single sample.

        sample = reward + discount_factor * max over a' of Q(next_state, a')
        Q(state, action) <- (1 - learning_rate) * Q(state, action) + learning_rate * sample

        When the step ended the episode there is no future, so sample = reward.
        """
        if done:
            sample = reward
        else:
            sample = reward + self.discount_factor * self.best_value(next_state, next_legal_actions)
        old_value = self.get_q_value(state, action)
        self.q_values[(state, action)] = (
            1.0 - self.learning_rate
        ) * old_value + self.learning_rate * sample
