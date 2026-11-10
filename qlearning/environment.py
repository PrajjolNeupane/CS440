"""The vacuum robot's 4x4 grid world. The rules are listed in .claude/CLAUDE.md.

The environment knows T (where an action can lead, and how likely each result is) and R (the
reward for each result). The Q-learning agent never reads them: it only sees what step() returns
and the legal actions for its cell.
"""

from __future__ import annotations

import random
from typing import NamedTuple

State = tuple[int, int]
"""A grid cell as (row, column), counted from the top-left corner."""

Action = str

UP: Action = "up"
DOWN: Action = "down"
LEFT: Action = "left"
RIGHT: Action = "right"
EXIT: Action = "exit"
MOVE_ACTIONS: tuple[Action, ...] = (UP, DOWN, LEFT, RIGHT)

ROW_COUNT = 4
COLUMN_COUNT = 4
START_STATE: State = (0, 0)
WALL_CELLS: frozenset[State] = frozenset({(1, 1)})
GOAL_STATE: State = (3, 3)
PENALTY_STATE: State = (3, 1)
TERMINAL_REWARDS: dict[State, float] = {GOAL_STATE: 10.0, PENALTY_STATE: -1.0}

# 20% of moves slip: 10% to each side of the chosen direction, never backwards.
DEFAULT_SLIP_PROBABILITY = 0.2

MOVE_OFFSETS: dict[Action, tuple[int, int]] = {
    UP: (-1, 0),
    DOWN: (1, 0),
    LEFT: (0, -1),
    RIGHT: (0, 1),
}
PERPENDICULAR_ACTIONS: dict[Action, tuple[Action, Action]] = {
    UP: (LEFT, RIGHT),
    DOWN: (LEFT, RIGHT),
    LEFT: (UP, DOWN),
    RIGHT: (UP, DOWN),
}


class StepResult(NamedTuple):
    """What the agent gets back after taking one action."""

    next_state: State
    reward: float
    done: bool


class Outcome(NamedTuple):
    """One possible result of an action and its probability: one entry of T and R."""

    probability: float
    next_state: State
    reward: float
    done: bool


def is_open_cell(state: State) -> bool:
    """True for a cell that is on the grid and is not the wall."""
    row, column = state
    on_grid = 0 <= row < ROW_COUNT and 0 <= column < COLUMN_COUNT
    return on_grid and state not in WALL_CELLS


def move(state: State, direction: Action) -> State:
    """The cell one step away in a direction. Hitting the wall or an edge means staying put."""
    row_offset, column_offset = MOVE_OFFSETS[direction]
    target_state = (state[0] + row_offset, state[1] + column_offset)
    return target_state if is_open_cell(target_state) else state


class VacuumEnvironment:
    """The grid the robot moves around in, one episode at a time."""

    def __init__(
        self,
        living_reward: float,
        slip_probability: float = DEFAULT_SLIP_PROBABILITY,
        seed: int | None = None,
    ) -> None:
        if not 0.0 <= slip_probability <= 1.0:
            raise ValueError(f"slip_probability must be between 0 and 1, got {slip_probability}")
        self.living_reward = living_reward
        self.slip_probability = slip_probability
        self._random_generator = random.Random(seed)
        self._current_state = START_STATE
        self._episode_over = False

    def open_states(self) -> list[State]:
        """Every cell the robot can be in (all cells except the wall), row by row."""
        return [
            (row, column)
            for row in range(ROW_COUNT)
            for column in range(COLUMN_COUNT)
            if is_open_cell((row, column))
        ]

    def legal_actions(self, state: State) -> tuple[Action, ...]:
        """Only `exit` in a terminal cell; up, down, left and right everywhere else."""
        if not is_open_cell(state):
            raise ValueError(f"{state} is not a cell the robot can be in")
        if state in TERMINAL_REWARDS:
            return (EXIT,)
        return MOVE_ACTIONS

    def transition_outcomes(self, state: State, action: Action) -> list[Outcome]:
        """T(s, a, s') and R(s, a, s') for one state and action.

        step() samples from this list. Value iteration reads it to check the learned policy.
        The agent never calls it.
        """
        if action not in self.legal_actions(state):
            raise ValueError(f"{action!r} is not a legal action in {state}")
        if action == EXIT:
            return [Outcome(1.0, state, TERMINAL_REWARDS[state], True)]

        first_side, second_side = PERPENDICULAR_ACTIONS[action]
        side_probability = self.slip_probability / 2
        directions_with_probability = [
            (1.0 - self.slip_probability, action),
            (side_probability, first_side),
            (side_probability, second_side),
        ]
        return [
            Outcome(probability, move(state, direction), self.living_reward, False)
            for probability, direction in directions_with_probability
            if probability > 0.0
        ]

    def reset(self, start_state: State = START_STATE) -> State:
        """Start a new episode. Tests may start from another cell; training always uses (0, 0)."""
        if not is_open_cell(start_state):
            raise ValueError(f"{start_state} is not a cell the robot can be in")
        self._current_state = start_state
        self._episode_over = False
        return start_state

    def step(self, action: Action) -> StepResult:
        """Take one action from the current cell and return what happened."""
        if self._episode_over:
            raise RuntimeError("The episode is over. Call reset() to start a new one.")
        outcome = self._sample_outcome(self.transition_outcomes(self._current_state, action))
        self._current_state = outcome.next_state
        self._episode_over = outcome.done
        return StepResult(outcome.next_state, outcome.reward, outcome.done)

    def _sample_outcome(self, outcomes: list[Outcome]) -> Outcome:
        random_value = self._random_generator.random()
        cumulative_probability = 0.0
        for outcome in outcomes:
            cumulative_probability += outcome.probability
            if random_value < cumulative_probability:
                return outcome
        # Rounding can leave the total a hair under 1.0
        return outcomes[-1]
