---
name: tune-hyperparameters
description: Choose the vacuum robot's Q-learning hyperparameters (living reward, discount factor, learning rate, exploration rate, episode count) with a staged, seeded sweep, then set them as the defaults and record the evidence in docs/hyperparameter-tuning.md. Use when asked to tune, pick, justify or re-check the hyperparameters, or after a change to the environment or the agent that could change the best values.
argument-hint: "[optional: one hyperparameter to re-check, e.g. exploration rate]"
allowed-tools: Bash(python3 -m qlearning *), Bash(python3 -m qlearning.sweep *), Bash(python3 -m unittest *), Read, Edit, Write
---

# Tune the Q-learning hyperparameters

Pick each hyperparameter from evidence, one stage at a time, and leave a record a teammate can
check. The tool is `python3 -m qlearning.sweep` (see `python3 -m qlearning.sweep --help`). Every
hyperparameter option takes a comma-separated list; the sweep trains one agent per combination
and seed and ranks them.

If the user named one hyperparameter, run only that stage, using the current defaults in
`qlearning/training.py` for everything else, and update its part of the record.

## What the sweep measures

- **optimal cells**: cells (out of 15) where the learned action is optimal, checked against value
  iteration for the same living reward and discount factor. The main score.
- **worst seed**: the same count for the unluckiest seed. Shows how reliable a setting is.
- **V(start) error**: how far the learned V(start) is from V*(start). Shows whether the Q-values
  themselves were learned, not just their order.
- **goal rate / avg return**: following the learned policy greedily in fresh test episodes.

The living reward and the discount factor change the problem itself (what counts as a good
policy). The learning rate, exploration rate and episode count only change how well the robot
learns it. That is why stages 1-2 look at value iteration's policies first.

## Stages

Run each stage, read the table, decide, and use the decision as the base for the next stage by
passing it explicitly (for example `--living-reward -0.04`). Use seeds 0-9 (the default) while
tuning. Commands can take a minute or two; run long ones in the background.

1. **Living reward (the problem).** No learning:
   `python3 -m qlearning.sweep --optimal-policies --living-reward 1,0.1,0,-0.04,-0.1,-0.4,-1,-2`
   Find the range where the optimal policy heads for +10 by a short route and does not take risks
   next to -1. Too high (positive) and the robot is paid to wander; too low and ending the episode
   at -1 starts to look attractive. Pick a value inside the good range, not at its edge, and close
   to 0 so it nudges the robot along without changing what it is trying to do.
2. **Discount factor.** First the problem:
   `--optimal-policies --discount-factor 0.5,0.8,0.9,0.95,0.99` with the chosen living reward. Rule
   out values whose optimal policy is not sensible. Then learnability for the ones left:
   `--discount-factor <the sensible ones>`. A discount factor close to 1 makes the +10 reach
   further back but takes more episodes to learn.
3. **Learning rate schedule.** `--learning-rate-start 0.1,0.3,0.5,1.0 --learning-rate-end 0.01,0.05,0.1`.
   The slides: Q-learning converges if the learning rate eventually becomes small, but not too
   quickly. Moves slip, so a rate that stays large keeps chasing the latest sample.
4. **Exploration schedule.** `--exploration-rate-start 0.3,0.6,1.0 --exploration-rate-end 0.01,0.05,0.1,0.2`.
   The slides: explore enough, and lower epsilon over time.
5. **Episode count.** `--episode-count 1000,2000,5000,10000,20000`. Pick the smallest count where
   the scores stop improving.
6. **Confirm on fresh seeds.** Rerun the final values against the old defaults with
   `--first-seed 100 --seeds 20`. Tuning and checking on the same seeds would flatter the result.

## How to decide

- Rank by optimal cells. Treat differences under about 0.3 cells as a tie and break ties by worst
  seed, then V(start) error, then the cheaper or simpler setting (fewer episodes, round numbers).
- If the best value is at the edge of the range tried, extend the range and rerun before deciding.
- Never change `slip_probability` or `max_steps_per_episode` to improve scores: slips are a rule of
  the environment (`.claude/CLAUDE.md`), and the step cap is only a safety limit.
- Say so when a result is noisy or a choice is a judgment call. Do not invent a reason the table
  does not support.

## Finish

1. Set the chosen values as the defaults of `Hyperparameters` in `qlearning/training.py` and remove
   the word "provisional" from its docstring. The PostToolUse hook runs the tests after the edit.
   Two tests in `tests/test_training.py` check that the defaults learn a near-optimal policy; if
   they fail, the new defaults are worse than they looked, so look again before loosening them.
2. Run `python3 -m qlearning` and keep its output for the record.
3. Write or update `docs/hyperparameter-tuning.md` with:
   - a table of the final values, each with a one-line reason
   - for each stage: the exact command, the top rows of its (Markdown) table, and the
     decision with its reason
   - the fresh-seed confirmation table and the final `python3 -m qlearning` output
4. Update the hyperparameter section of `README.md` if it exists (the values and one line each).
5. Report back: the chosen values, the evidence in two or three sentences, anything that was a
   close call.
