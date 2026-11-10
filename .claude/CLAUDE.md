# Project Context

CS440 group repo for the vacuum robot labs.
- Lab 4: context engineering setup (strategy 1: persistent memory, strategy 4: sub-agents)
- Lab 5: Q-learning for the vacuum robot, using hooks, sub-agents and skills

## Architecture Overview
- Python 3, standard library only. Tests use `unittest`.
- `scripts/context-switcher.sh`: UserPromptSubmit hook that loads `.claude/context/*.md` when relevant
- `scripts/run-tests-hook.sh`: PostToolUse hook that runs the unit tests after any Write or Edit to a
  `.py` file in `qlearning/` or `tests/`. Failures come back to Claude, so fix them before moving on.
- `.claude/context/`: longer notes that are loaded only on demand (see "Context loading" below)
- `.claude/agents/rl-code-reviewer.md`: read-only sub-agent that reviews the Q-learning code against
  these rules. Run it after changing `qlearning/` or `tests/`, and before a commit.
- `qlearning/`: the Lab 5 Q-learning code. Run it with `python3 -m qlearning` (`--help` lists the options)
  - `environment.py`: the grid, slips and rewards, plus the State/Action names and constants. It owns T and R.
  - `agent.py`: the Q-table, the update and epsilon-greedy. It imports nothing from the project.
  - `training.py`: `Hyperparameters` (the defaults are the chosen values), the training loop,
    policy extraction and evaluation, and the printed grids
  - `value_iteration.py`: for checking only. It reads T and R to find the true optimal policy.
  - `__main__.py`: the command line; one option per `Hyperparameters` field (e.g. `--episode-count`)
  - `sweep.py`: hyperparameter sweep (`python3 -m qlearning.sweep --help`), same option names
- `tests/`: unittest tests, one file per module
- `.claude/skills/tune-hyperparameters/`: skill that chooses the hyperparameters with a staged sweep.
  Use it to re-check the values after changing the environment or the agent. The evidence for the
  current values is in `docs/hyperparameter-tuning.md`.

## Vacuum Environment (business rules)
Grid cells are (row, column), counted from the top-left corner.

| row \ col | 0     | 1  | 2 | 3  |
|-----------|-------|----|---|----|
| 0         | start | 0  | 0 | 0  |
| 1         | 0     | X  | 0 | 0  |
| 2         | 0     | 0  | 0 | 0  |
| 3         | 0     | -1 | 0 | 10 |

- The robot starts at (0, 0).
- X at (1, 1) is a wall. Moving into the wall or off the grid leaves the robot where it is.
- (3, 1) = -1 and (3, 3) = +10 are terminal. As in the RL slides, the only action in a terminal
  cell is `exit`, which collects that cell's reward and ends the episode. Arriving in a terminal
  cell does not end the episode by itself.
- Actions: up, down, left, right (and `exit` in a terminal cell).
- Moves slip: the robot goes in the chosen direction 80% of the time and in each of the two
  perpendicular directions 10% of the time (e.g. `up` goes left or right 10% each). If that move hits
  the wall or an edge, the robot stays put. `exit` always works.
- The robot is not given T (the transition model) or R (the reward function). It learns only from experience.

## Coding Standards
- Type hints on every function, including the return type
- Descriptive variable names, no abbreviations
- Unit tests for all business logic (environment rules, the Q-value update, policy extraction)
- Seed all randomness so results can be reproduced
- Keep the agent separate from the environment: the agent only sees what the environment's `step()` returns

## Context Loading
`scripts/context-switcher.sh` adds these files to the conversation when they are needed, at most once per session:
- `.claude/context/qlearning.md`: on a `qlearning*` or `lab5*` branch, or when the prompt mentions Q-learning terms
- `.claude/context/testing.md`: when the prompt mentions tests
