# Hyperparameter Tuning

How the Q-learning hyperparameters were chosen. The procedure is the `tune-hyperparameters`
skill (`.claude/skills/tune-hyperparameters/SKILL.md`) and the tool is `python3 -m qlearning.sweep`.
Each sweep trains one agent per combination of values and seed (seeds 0-9 unless stated) and
checks the learned policy against value iteration's optimal policy for the same rewards.

## Final values

| Hyperparameter | Value | Reason |
|---|---|---|
| Living reward | -0.04 | Inside the range (+0.05 to -0.4 at gamma 0.99) where the optimal policy always reaches +10; close to 0, so it hurries the robot without changing its goal |
| Discount factor (gamma) | 0.999 | Near 1, so the robot maximizes the reward it actually collects and avoids the -1 risk; also the easiest to learn (best worst seed and value error) |
| Learning rate (alpha) | 0.1, falling linearly to 0.01 | Best in the sweep; small at the end as the slides require for convergence, but not shrunk to nothing |
| Exploration rate (epsilon) | 1.0, falling linearly to 0.05 | Starting fully random was clearly best; lowered over time as the slides recommend |
| Episodes | 10,000 | Smallest count where every tuning seed learned the optimal action in all 15 cells |

Not tuned, because they are rules of the environment rather than learning choices: slip
probability 0.2 (80/10/10) and the 200-step safety cap per episode.

**Result** (`python3 -m qlearning`, seed 0): optimal action in 15 of 15 cells, +10 reached in
100% of 1000 test episodes, 8.8 steps on average. On 20 fresh seeds (100-119): 14.7 of 15 cells
optimal on average. Every miss is the same near-tie at (0, 2), where `right` gives up at most
0.007 expected return compared with `down`.

## Scores used

- **optimal cells**: cells (out of 15) where the learned action is optimal; the main score
- **worst seed**: the same count for the unluckiest seed
- **V(start) error**: |learned V(start) - V*(start)|, whether the values themselves were learned
- **goal rate / avg return**: following the learned policy in fresh test episodes

Differences under about 0.3 optimal cells were treated as ties.

## Stage 1: living reward

`python3 -m qlearning.sweep --optimal-policies --living-reward 1,0.1,0,-0.04,-0.1,-0.4,-1,-2`
(no learning, discount factor 0.9), followed by a finer scan of where the optimal policy changes:

| Discount factor | Optimal policy changes at living reward |
|---|---|
| 0.9 | same policy from +0.2 down to -1.3; at +0.5 or more it never exits; at -1.6 or less it steps into -1 from (3, 0) |
| 0.99 | +0.1 or more: never exits. +0.05 to -0.07: hugs walls to avoid -1. -0.07 to -0.7: shorter routes past -1. -2: steps into -1 |

**Decision**: with gamma 0.9 the living reward barely matters, because discounting already makes
the robot hurry. With gamma near 1 it controls the trade-off between safety and speed, as in the
slides' gridworld examples. Settled together with the discount factor below.

## Stage 2: discount factor

The optimal policy for each discount factor at living reward -0.04
(`--optimal-policies --discount-factor 0.5,0.8,0.95,0.99`), following it for 2000 episodes:

| Discount factor | Reaches +10 | Average steps | Average return |
|---|---|---|---|
| 0.5 / 0.8 / 0.9 | 98.9% | 8.4 | 9.58 |
| 0.95 / 0.99 | 100% | 8.7 | 9.69 |

With a low discount, the +10 is worth so little after a few steps that a 10% risk of the -1 cell
is acceptable. Near 1, the robot takes 0.3 more steps and never ends at -1.

Learnability, `--discount-factor 0.9,0.95,0.99,0.999 --living-reward -0.01,-0.04,-0.1,-0.4`
(16 combinations, top rows; old learning-rate and exploration defaults):

| living_reward | discount_factor | optimal cells | worst seed | V(start) error | goal rate | avg return |
|---|---|---|---|---|---|---|
| -0.04 | 0.999 | 14.5/15 | 14/15 | 0.011 | 100.0% | 9.686 |
| -0.01 | 0.999 | 14.4/15 | 13/15 | 0.002 | 100.0% | 9.922 |
| -0.4 | 0.9 | 14.3/15 | 14/15 | 0.063 | 98.4% | 6.826 |
| -0.4 | 0.95 | 14.2/15 | 13/15 | 0.054 | 98.9% | 6.843 |
| -0.01 | 0.99 | 14.1/15 | 14/15 | 0.019 | 100.0% | 9.921 |
| -0.04 | 0.99 | 13.9/15 | 12/15 | 0.021 | 100.0% | 9.687 |
| -0.04 | 0.9 | 13.7/15 | 12/15 | 0.050 | 98.6% | 9.544 |

0.999 was at the edge, so the range was extended
(`--discount-factor 0.99,0.999,1.0 --living-reward -0.01,-0.04`):

| living_reward | discount_factor | optimal cells | worst seed | V(start) error | goal rate | avg return |
|---|---|---|---|---|---|---|
| -0.04 | 0.999 | 14.5/15 | 14/15 | 0.011 | 100.0% | 9.686 |
| -0.01 | 0.999 | 14.4/15 | 13/15 | 0.002 | 100.0% | 9.922 |
| -0.01 | 1.0 | 14.2/15 | 13/15 | 0.002 | 100.0% | 9.921 |
| -0.04 | 1.0 | 14.1/15 | 13/15 | 0.005 | 100.0% | 9.685 |

**Decision**: gamma 0.999 and living reward -0.04. A discount near 1 is learned best: with little
discounting the gap between a safe and a risky action is large, so the order of the Q-values is
learned reliably. -0.04 and -0.01 tie on optimal cells; -0.04 has the better worst seed and gives
the robot a clearer reason not to waste steps.

## Stage 3: learning rate

`--learning-rate-start 0.1,0.3,0.5,1.0 --learning-rate-end 0.01,0.05,0.1` (top rows):

| learning_rate_start | learning_rate_end | optimal cells | worst seed | V(start) error |
|---|---|---|---|---|
| 0.1 | 0.01 | 14.8/15 | 14/15 | 0.003 |
| 0.3 | 0.01 | 14.7/15 | 14/15 | 0.007 |
| 0.5 | 0.1 | 14.6/15 | 14/15 | 0.009 |
| 1.0 | 0.1 | 13.5/15 | 12/15 | 0.015 |

Both best values were at the edge, so: `--learning-rate-start 0.05,0.1,0.2 --learning-rate-end 0.001,0.005,0.01`:

| learning_rate_start | learning_rate_end | optimal cells | worst seed | V(start) error |
|---|---|---|---|---|
| 0.1 | 0.001 | 14.9/15 | 14/15 | 0.004 |
| 0.1 | 0.005 | 14.8/15 | 14/15 | 0.003 |
| 0.1 | 0.01 | 14.8/15 | 14/15 | 0.003 |
| 0.2 | 0.001 | 14.8/15 | 14/15 | 0.004 |
| 0.05 | 0.005 | 14.7/15 | 14/15 | 0.004 |

**Decision**: start at 0.1 (0.05 and 0.2 were both worse). The three end values tie, so 0.01, the
largest: the slides say the learning rate must become small, but not too quickly. Because moves
slip, a learning rate that stays large keeps chasing the latest random outcome (1.0 to 0.1 was
the worst row).

## Stage 4: exploration rate

`--exploration-rate-start 0.3,0.6,1.0 --exploration-rate-end 0.01,0.05,0.1,0.2` (selected rows):

| exploration_rate_start | exploration_rate_end | optimal cells | worst seed | V(start) error |
|---|---|---|---|---|
| 1.0 | 0.1 | 14.8/15 | 14/15 | 0.003 |
| 1.0 | 0.2 | 14.8/15 | 14/15 | 0.003 |
| 1.0 | 0.05 | 14.8/15 | 14/15 | 0.003 |
| 1.0 | 0.01 | 14.8/15 | 14/15 | 0.007 |
| 0.6 | 0.05 | 13.9/15 | 13/15 | 0.003 |
| 0.3 | 0.05 | 11.7/15 | 10/15 | 0.009 |

**Decision**: start at 1.0. Cells like (2, 1) and (3, 0) are only reached by slips, so the robot
needs fully random early episodes to learn them. The end values 0.05 to 0.2 tie; 0.05 is kept.
Because Q-learning is off-policy, exploring during training does not hurt the learned policy.

## Stage 5: episodes

`--episode-count 1000,2000,5000,10000,20000`:

| episode_count | optimal cells | worst seed | V(start) error |
|---|---|---|---|
| 20000 | 15.0/15 | 15/15 | 0.004 |
| 10000 | 15.0/15 | 15/15 | 0.004 |
| 5000 | 14.8/15 | 14/15 | 0.003 |
| 2000 | 13.8/15 | 13/15 | 0.005 |
| 1000 | 11.4/15 | 11/15 | 0.011 |

**Decision**: 10,000, the smallest count where every seed was optimal in all 15 cells (about 0.6 s
per training run).

## Stage 6: confirmation on fresh seeds 100-119

| Setting | optimal cells | worst seed | V(start) error | goal rate | avg return |
|---|---|---|---|---|---|
| Final values (gamma 0.999) | 14.7/15 | 14/15 | 0.005 | 100.0% | 9.689 |
| Final values with gamma 0.99 | 14.6/15 | 13/15 | 0.014 | 100.0% | 9.690 |
| Old provisional defaults | 13.8/15 | 12/15 | 0.089 | 99.1% | 9.601 |

The final values do slightly worse on fresh seeds than on the tuning seeds (14.7 instead of
15.0): some of the tuning gain was luck with seeds 0-9. They still clearly beat the old defaults.
The only miss, in 7 of 20 seeds, is the near-tie at (0, 2) described above.

## Final run

```
$ python3 -m qlearning
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

Two arrows look odd but are optimal. At (2, 1) the robot moves up into the wall, and at (3, 0)
left into the edge. Either way it usually stays put, and it leaves only when a slip carries it
sideways, which can never be into -1. The normal moves are riskier: `right` from (2, 1) and `up`
from (3, 0) each have a 10% slip into -1. With almost no discounting, waiting a few extra steps
costs much less than that risk. These cells are only reached after a slip, so the main route
from the start (along the top row, then down the right side) is unaffected.
