# Q-learning Notes (loaded on demand)

Q-learning is model-free: the agent never reads T or R. It learns from each sampled step
(state, action, reward, next_state, done) that the environment returns.

## Update Rule (RL slides: L1 p33, L2 p5)
sample = R(s, a, s') + gamma * max over a' of Q(s', a')
Q(s, a) <- (1 - alpha) * Q(s, a) + alpha * sample
- Q starts at 0 for every (state, action) pair.
- After `exit` the episode is over, so the sample is just the reward (there is no future term).

## Rewards
- Terminal rewards use the slides' exit convention: in (3, 3) the only action is `exit`, worth +10,
  and in (3, 1) it is `exit`, worth -1. Taking `exit` ends the episode.
- Every move (up, down, left, right) gives the living reward, a small negative number. It pushes
  the robot toward +10 quickly, but it must stay small enough that a path near -1 is not worth the risk.

## Hyperparameters
| Name             | Symbol  | Role                                                  |
|------------------|---------|-------------------------------------------------------|
| Living reward    | r_live  | Reward for each step that does not end the episode     |
| Discount factor  | gamma   | How much future reward counts, between 0 and 1         |
| Learning rate    | alpha   | How far each update moves Q(s, a) toward the target    |
| Exploration rate | epsilon | Chance of taking a random action (epsilon-greedy)      |

What the slides say:
- Q-learning converges to the optimal policy even while acting suboptimally (off-policy), as long as
  you explore enough and eventually make alpha small, but not too quickly (L1 p34, L2 p6).
- Plain epsilon-greedy keeps thrashing once learning is done; one fix is to lower epsilon over time (L2 p9).
- The crawler demo uses gamma 0.9, alpha 1.0 and epsilon 0.8 or 0.1. The slides give no living reward
  or slip model for this grid.

## Slip Model (group decision)
The slides don't give one, so we use the usual Berkeley noise of 0.2: 80% the chosen direction,
10% each perpendicular direction, and bumping into the wall or an edge means staying put.
Because moves are random, a single fixed alpha never fully settles; this is why alpha should shrink
over time, or stay small.

Chosen values (from the tune-hyperparameters skill; evidence in docs/hyperparameter-tuning.md):
living reward -0.04, gamma 0.999, alpha 0.1 falling linearly to 0.01, epsilon 1.0 falling linearly
to 0.05, 10,000 episodes. They are the defaults of `Hyperparameters` in qlearning/training.py.
Re-run the skill instead of hand-editing them.

## Conventions
- Q-table: a dict mapping (state, action) to a float, starting at 0.0
- Epsilon-greedy: with probability epsilon take a random action, otherwise the best one; break ties at random
- Report the learned policy as a grid of arrows, plus V(s) = max over a of Q(s, a) for each cell
