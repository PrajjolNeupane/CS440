# Testing Notes (loaded on demand)

- Framework: `unittest` from the standard library, so nothing needs to be installed.
- Run all tests: `python3 -m unittest discover -s tests -v`
- The PostToolUse hook (`scripts/run-tests-hook.sh`) already runs them after every edit to a `.py` file
  in `qlearning/` or `tests/`, and sends the last 60 lines back to Claude when they fail.
- One test file per module: `tests/test_<module>.py`. Test names say what they check,
  for example `test_moving_into_wall_keeps_position`.
- Business logic that must have tests:
  - environment moves: walls, grid edges, terminal cells, rewards
  - the Q-value update, for both terminal and non-terminal steps
  - epsilon-greedy action choice at epsilon = 0 and epsilon = 1
  - extracting the greedy policy from the Q-table
- Fix the random seed in any test that uses randomness.
