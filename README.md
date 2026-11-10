# CS440: Vacuum Robot Q-learning (Labs 4 and 5)

A vacuum robot learns to cross a 4x4 grid with Q-learning, without being told how its moves work
(T) or what they are worth (R). The repository also holds the Claude Code setup used to build it:

- **Lab 4**: context engineering strategy 1 (persistent memory) and strategy 4 (sub-agents)
- **Lab 5**: the Q-learning solution, built with three Claude Code features: hooks, sub-agents
  and skills

## How to run

Requirements: Python 3.9 or later. Only the standard library is used, so there is nothing to
install. The Claude Code hooks also need `bash` and `jq`.

```bash
python3 -m qlearning                       # train with the chosen values and print the result
python3 -m qlearning --help                # one option per hyperparameter
python3 -m qlearning --discount-factor 0.9 --episode-count 2000 --seed 3
python3 -m unittest discover -s tests -v   # 85 unit tests, about 1 second
python3 -m qlearning.sweep --help          # the hyperparameter sweep
```

Run the commands from the repository root. Every run is seeded, so the same options always give
the same output.

Output of `python3 -m qlearning`:

```
Learned policy
   →      →      ↓      ↓
   ↑      X      ↓      ↓
   ↑      ↑      →      ↓
   ←      -1     →     +10

Learned values, V(s) = max over a of Q(s, a)
   9.62   9.69   9.75   9.79
   9.55      X   9.81   9.86
   9.48   9.43   9.87   9.93
   8.99  -1.00   9.93  10.00

Average return over the last 100 training episodes: 9.667
Greedy policy over 1000 test episodes: average return 9.690, reached +10 in 100.0%, average 8.8 steps

Check against value iteration (it uses T and R, which the robot never sees)
  optimal action in 15 of 15 cells
```

## The problem

| row \ col | 0     | 1  | 2 | 3  |
|-----------|-------|----|---|----|
| 0         | start | 0  | 0 | 0  |
| 1         | 0     | X  | 0 | 0  |
| 2         | 0     | 0  | 0 | 0  |
| 3         | 0     | -1 | 0 | 10 |

- The robot starts at (0, 0), as (row, column). X at (1, 1) is a wall.
- Actions are up, down, left and right. Moving into the wall or off the grid leaves the robot
  where it is.
- Moves slip: 80% of the time the robot goes the way it chose, and 10% of the time it goes to
  each side (never backwards). The slides don't give a slip model for this grid, so we used the
  standard 80/10/10.
- (3, 3) = +10 and (3, 1) = -1 are terminal. As in the RL slides, the only action there is
  `exit`, which collects the reward and ends the episode.
- Every move costs the living reward.

## How it works

The robot keeps a table of Q-values, all starting at 0. After every step it updates the value for
the cell and action it just used, following the RL slides:

```
sample  = reward + gamma * max over a' of Q(next cell, a')     (just the reward after exit)
Q(s, a) = (1 - alpha) * Q(s, a) + alpha * sample
```

- It picks actions epsilon-greedily: with probability epsilon a random move, otherwise the best
  one.
- Epsilon and alpha both fall in a straight line over the episodes. The slides say to lower
  epsilon over time and that alpha must eventually be small.
- The learned policy is the best action in each cell, and V(s) is the largest Q-value there.
- **Value iteration is used only to check the result.** It needs T and R, so the robot never uses
  it. The tests and the report use it to confirm that Q-learning found the optimal policy.

Two arrows look odd but are optimal: up into the wall at (2, 1) and left into the edge at (3, 0).
The robot waits there until a slip carries it sideways, because the normal moves from those cells
have a 10% chance of slipping into -1.

## Hyperparameters

Chosen with the `tune-hyperparameters` skill (below). Every command, table and decision is in
[docs/hyperparameter-tuning.md](docs/hyperparameter-tuning.md).

| Hyperparameter | Value | Why |
|---|---|---|
| Living reward | -0.04 | Well inside the range where the optimal policy always reaches +10, and close to 0, so it hurries the robot without changing its goal |
| Discount factor (gamma) | 0.999 | Near 1, the robot maximizes the reward it actually collects and avoids the -1 risk (100% vs 98.9% reaching +10 at gamma 0.9). It was also the easiest to learn |
| Learning rate (alpha) | 0.1, falling to 0.01 | Best in the sweep. Moves slip, so alpha has to end small, but shrinking it to nothing stopped learning too early |
| Exploration rate (epsilon) | 1.0, falling to 0.05 | Starting fully random was clearly best, because some cells are only reached by slips |
| Episodes | 10,000 | The smallest count where every tuning seed learned the optimal action in all 15 cells |

On 20 fresh seeds the final values average 14.7 of 15 optimal cells, against 13.8 for our first
guess (gamma 0.9, alpha 0.5 to 0.05, 5,000 episodes). The only miss is a near-tie at (0, 2) that
costs at most 0.007 expected return.

## Project structure

```
qlearning/
  environment.py      the grid, slips and rewards: the only code that knows T and R
  agent.py            Q-table, update, epsilon-greedy; imports nothing from the project
  training.py         Hyperparameters, training loop, policy extraction and evaluation, grids
  value_iteration.py  optimal policy from T and R, for checking only
  sweep.py            hyperparameter sweep used by the skill
  __main__.py         the command line
tests/                unittest tests, one file per module
docs/
  hyperparameter-tuning.md   evidence for the chosen values
scripts/
  context-switcher.sh   hook: loads .claude/context/ notes when they are relevant
  run-tests-hook.sh     hook: runs the tests after Claude edits Python code
.claude/
  CLAUDE.md             project memory: grid rules, coding standards, file map
  settings.json         hook configuration
  context/              notes loaded on demand: qlearning.md, testing.md
  agents/rl-code-reviewer.md          sub-agent
  skills/tune-hyperparameters/SKILL.md  skill
```

## Claude Code setup

### Lab 4, strategy 1: persistent memory ("Write")

- **Project memory**: [.claude/CLAUDE.md](.claude/CLAUDE.md), loaded in every session. It holds
  the shared rules: the grid and its business rules, the coding standards (type hints, descriptive
  names, unit tests, seeded randomness, keeping the agent apart from the environment) and a map of
  the files.
- **User memory**: each group member keeps personal preferences in their own
  `~/.claude/CLAUDE.md`. That file is not in the repository. We moved the preferences the whole
  group shares (explicit return types, descriptive names) into the project file.
- **Dynamic imports**: longer notes live in `.claude/context/` and are loaded only when needed.
  `scripts/context-switcher.sh` runs on every prompt (a UserPromptSubmit hook). It picks notes by
  git branch (`qlearning*`, `lab5*`) and by words in the prompt (Q-learning terms such as
  "epsilon" load `qlearning.md`, and "test" loads `testing.md`). It loads each note at most once
  per session, so the same text doesn't pile up.

### Lab 4, strategy 4: sub-agents ("Isolate")

[.claude/agents/rl-code-reviewer.md](.claude/agents/rl-code-reviewer.md) reviews `qlearning/` and
`tests/` in its own context window, and only its report comes back to the main conversation. It
follows the four traits from the slides:

- **Specific description**, with "Use proactively after changing qlearning/ or tests/".
- **Limited tools**: Glob, Grep, Read and Bash. It has no Edit or Write, so it reports problems but
  never fixes them.
- **Structured output**: Summary, Critical / Major / Minor Issues, Recommendations, Test Results
  and Approval Status. Every issue gives the file and line, the rule it breaks, and a fix.
- **Obstacles Encountered**: reports anything unclear. In a test run it noticed that CLAUDE.md
  never said that arriving in a terminal cell doesn't end the episode, and we added that.

Before the real code existed, we tested it on a sample with 8 planted bugs (a backwards slip, a
SARSA-style update, the agent reading the grid, and others). It caught all 8.

### Lab 5: hooks, sub-agents and skills

| Feature | Where | What it does for this project |
|---|---|---|
| Hooks | `.claude/settings.json` | UserPromptSubmit runs the context switcher above. PostToolUse runs the unit tests after every Write or Edit to a `.py` file in `qlearning/` or `tests/`, and sends any failures straight back to Claude, so it sees them and fixes them. |
| Sub-agents | `.claude/agents/` | The code reviewer above. It ran after the code was written and again after a refactor. |
| Skills | `.claude/skills/tune-hyperparameters/` | A step-by-step procedure for choosing the four hyperparameters plus the episode count. It runs `python3 -m qlearning.sweep` in stages, decides each value with fixed rules (ties, extending a range when the best value sits at its edge, confirming on fresh seeds), and writes the evidence to `docs/`. |

**Why these three:**
- **Hooks** make the checks automatic. Claude can't forget to run the tests, and it gets the right
  notes without being asked.
- **The test hook replaces a test-runner sub-agent.** The slides say test-runner sub-agents
  "performed worst" because they hide the output you need. The hook puts the failure output in the
  main conversation instead.
- **The sub-agent** suits review work: it only needs to return a verdict, which matches the
  slides' rule for when to delegate.
- **The skill** turns choosing hyperparameters, which the lab asks for, into a procedure anyone in
  the group can repeat.
- **We didn't use MCP.** The project talks to no outside service, and everything runs locally on
  the standard library.
