"""Hyperparameters, the training loop, and what comes out of it: the learned policy and values,
how well the policy does, and the printed grids.

The training loop is the only code that talks to both sides: it asks the environment for the
legal actions, passes them to the agent, and hands the agent whatever step() returned.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import NamedTuple

from qlearning.agent import QLearningAgent
from qlearning.environment import (
    COLUMN_COUNT,
    DOWN,
    GOAL_STATE,
    LEFT,
    RIGHT,
    ROW_COUNT,
    TERMINAL_REWARDS,
    UP,
    WALL_CELLS,
    Action,
    State,
    VacuumEnvironment,
)

ACTION_ARROWS: dict[Action, str] = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→"}
GRID_CELL_WIDTH = 7


@dataclass(frozen=True)
class Hyperparameters:
    """Every value that changes how the robot learns. The defaults are the values chosen by the
    tune-hyperparameters sweep; docs/hyperparameter-tuning.md has the evidence."""

    living_reward: float = -0.04
    discount_factor: float = 0.999
    # The learning rate and exploration rate fall in a straight line from their start value at
    # the first episode to their end value at the last one. Equal start and end means constant.
    learning_rate_start: float = 0.1
    learning_rate_end: float = 0.01
    exploration_rate_start: float = 1.0
    exploration_rate_end: float = 0.05
    episode_count: int = 10000
    max_steps_per_episode: int = 200
    slip_probability: float = 0.2
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject values that make no sense, so a typo on the command line fails clearly."""
        for field_name in [
            "discount_factor",
            "learning_rate_start",
            "learning_rate_end",
            "exploration_rate_start",
            "exploration_rate_end",
            "slip_probability",
        ]:
            value = getattr(self, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0 and 1, got {value}")
        if self.episode_count < 1 or self.max_steps_per_episode < 1:
            raise ValueError("episode_count and max_steps_per_episode must be at least 1")


@dataclass
class TrainingResult:
    """The trained agent, the environment it trained in, and the total (undiscounted) reward of
    each training episode."""

    agent: QLearningAgent
    environment: VacuumEnvironment
    episode_returns: list[float] = field(default_factory=list)


class PolicyEvaluation(NamedTuple):
    """How a fixed policy did over many episodes.

    average_return: total (undiscounted) reward per episode
    goal_rate: fraction of episodes that ended with `exit` at (3, 3), between 0 and 1
    average_step_count: steps per episode, including the final `exit`
    """

    average_return: float
    goal_rate: float
    average_step_count: float


def linear_schedule(
    start_value: float, end_value: float, episode_index: int, episode_count: int
) -> float:
    """The value for one episode when it moves in a straight line from start_value (first
    episode) to end_value (last episode). A single episode uses start_value."""
    if episode_count <= 1:
        return start_value
    progress = episode_index / (episode_count - 1)
    return start_value + (end_value - start_value) * progress


def run_training_episode(
    environment: VacuumEnvironment, agent: QLearningAgent, max_steps: int
) -> tuple[float, int]:
    """Run one episode from the start cell, updating Q after every step.

    Returns the total (undiscounted) reward and the number of steps taken.
    """
    state = environment.reset()
    total_reward = 0.0
    for step_number in range(1, max_steps + 1):
        legal_actions = environment.legal_actions(state)
        action = agent.choose_action(state, legal_actions)
        next_state, reward, done = environment.step(action)
        agent.update(
            state, action, reward, next_state, environment.legal_actions(next_state), done
        )
        total_reward += reward
        state = next_state
        if done:
            return total_reward, step_number
    return total_reward, max_steps


def train(hyperparameters: Hyperparameters) -> TrainingResult:
    """Train a fresh agent. The same hyperparameters (including the seed) give the same result."""
    # One seed makes two independent random streams: one for slips, one for the agent's choices
    seed_generator = random.Random(hyperparameters.seed)
    environment = VacuumEnvironment(
        living_reward=hyperparameters.living_reward,
        slip_probability=hyperparameters.slip_probability,
        seed=seed_generator.randrange(2**32),
    )
    agent = QLearningAgent(
        learning_rate=hyperparameters.learning_rate_start,
        discount_factor=hyperparameters.discount_factor,
        exploration_rate=hyperparameters.exploration_rate_start,
        seed=seed_generator.randrange(2**32),
    )
    result = TrainingResult(agent=agent, environment=environment)
    for episode_index in range(hyperparameters.episode_count):
        agent.learning_rate = linear_schedule(
            hyperparameters.learning_rate_start,
            hyperparameters.learning_rate_end,
            episode_index,
            hyperparameters.episode_count,
        )
        agent.exploration_rate = linear_schedule(
            hyperparameters.exploration_rate_start,
            hyperparameters.exploration_rate_end,
            episode_index,
            hyperparameters.episode_count,
        )
        episode_return, _ = run_training_episode(
            environment, agent, hyperparameters.max_steps_per_episode
        )
        result.episode_returns.append(episode_return)
    return result


def extract_policy(agent: QLearningAgent, environment: VacuumEnvironment) -> dict[State, Action]:
    """The greedy action for every open cell."""
    return {
        state: agent.greedy_action(state, environment.legal_actions(state))
        for state in environment.open_states()
    }


def extract_state_values(
    agent: QLearningAgent, environment: VacuumEnvironment
) -> dict[State, float]:
    """V(s) = max over a of Q(s, a) for every open cell."""
    return {
        state: agent.best_value(state, environment.legal_actions(state))
        for state in environment.open_states()
    }


def evaluate_policy(
    policy: dict[State, Action],
    environment: VacuumEnvironment,
    episode_count: int,
    max_steps: int,
) -> PolicyEvaluation:
    """Follow the policy with no exploration and no learning, and measure how it does."""
    total_return = 0.0
    goal_count = 0
    total_steps = 0
    for _ in range(episode_count):
        state = environment.reset()
        for step_number in range(1, max_steps + 1):
            state_before_step = state
            state, reward, done = environment.step(policy[state])
            total_return += reward
            if done:
                if state_before_step == GOAL_STATE:
                    goal_count += 1
                break
        total_steps += step_number
    return PolicyEvaluation(
        average_return=total_return / episode_count,
        goal_rate=goal_count / episode_count,
        average_step_count=total_steps / episode_count,
    )


def format_policy_grid(policy: dict[State, Action]) -> str:
    """One arrow per cell. The wall is X; a terminal cell shows the reward its `exit` collects."""
    lines = []
    for row in range(ROW_COUNT):
        cells = []
        for column in range(COLUMN_COUNT):
            state = (row, column)
            if state in WALL_CELLS:
                cell_text = "X"
            elif state in TERMINAL_REWARDS:
                cell_text = f"{TERMINAL_REWARDS[state]:+g}"
            else:
                cell_text = ACTION_ARROWS[policy[state]]
            cells.append(cell_text.center(GRID_CELL_WIDTH))
        lines.append("".join(cells).rstrip())
    return "\n".join(lines)


def format_value_grid(state_values: dict[State, float]) -> str:
    """V(s) for each cell with two decimals. The wall is X."""
    lines = []
    for row in range(ROW_COUNT):
        cells = []
        for column in range(COLUMN_COUNT):
            state = (row, column)
            cell_text = "X" if state in WALL_CELLS else f"{state_values[state]:.2f}"
            cells.append(cell_text.rjust(GRID_CELL_WIDTH))
        lines.append("".join(cells))
    return "\n".join(lines)
