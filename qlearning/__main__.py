"""Command line: train the robot and print what it learned.

    python3 -m qlearning
    python3 -m qlearning --episode-count 20000 --discount-factor 0.95 --seed 3
    python3 -m qlearning --help
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields

from qlearning.environment import VacuumEnvironment
from qlearning.training import (
    Hyperparameters,
    evaluate_policy,
    extract_policy,
    extract_state_values,
    format_policy_grid,
    format_value_grid,
    train,
)
from qlearning.value_iteration import OPTIMAL_LOSS_TOLERANCE, compute_policy_losses

EVALUATION_EPISODE_COUNT = 1000
RECENT_EPISODE_COUNT = 100


def build_argument_parser() -> argparse.ArgumentParser:
    """One option per Hyperparameters field (episode_count becomes --episode-count), with the
    field's default value. The sweep uses the same option names."""
    defaults = Hyperparameters()
    parser = argparse.ArgumentParser(
        prog="python3 -m qlearning",
        description="Train the vacuum robot with Q-learning and print the learned policy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    for hyperparameter_field in fields(Hyperparameters):
        default_value = getattr(defaults, hyperparameter_field.name)
        parser.add_argument(
            "--" + hyperparameter_field.name.replace("_", "-"),
            type=type(default_value),
            default=default_value,
        )
    return parser


def main(command_line_arguments: list[str] | None = None) -> int:
    parser = build_argument_parser()
    arguments = parser.parse_args(command_line_arguments)
    try:
        # argparse turns --episode-count back into episode_count, so the names match the fields
        hyperparameters = Hyperparameters(**vars(arguments))
    except ValueError as error:
        parser.error(str(error))

    result = train(hyperparameters)
    policy = extract_policy(result.agent, result.environment)
    state_values = extract_state_values(result.agent, result.environment)

    # A separate environment with its own seed, so testing the policy does not reuse training's slips
    evaluation_environment = VacuumEnvironment(
        living_reward=hyperparameters.living_reward,
        slip_probability=hyperparameters.slip_probability,
        seed=hyperparameters.seed + 1,
    )
    evaluation = evaluate_policy(
        policy,
        evaluation_environment,
        EVALUATION_EPISODE_COUNT,
        hyperparameters.max_steps_per_episode,
    )
    policy_losses = compute_policy_losses(
        result.environment, policy, hyperparameters.discount_factor
    )
    suboptimal_cells = {
        state: loss for state, loss in policy_losses.items() if loss > OPTIMAL_LOSS_TOLERANCE
    }
    recent_returns = result.episode_returns[-RECENT_EPISODE_COUNT:]

    print("Hyperparameters")
    for name, value in asdict(hyperparameters).items():
        print(f"  {name}: {value}")
    print("\nLearned policy")
    print(format_policy_grid(policy))
    print("\nLearned values, V(s) = max over a of Q(s, a)")
    print(format_value_grid(state_values))
    print(
        f"\nAverage return over the last {len(recent_returns)} training episodes: "
        f"{sum(recent_returns) / len(recent_returns):.3f}"
    )
    print(
        f"Greedy policy over {EVALUATION_EPISODE_COUNT} test episodes: "
        f"average return {evaluation.average_return:.3f}, "
        f"reached +10 in {evaluation.goal_rate:.1%}, "
        f"average {evaluation.average_step_count:.1f} steps"
    )
    print("\nCheck against value iteration (it uses T and R, which the robot never sees)")
    print(f"  optimal action in {len(policy) - len(suboptimal_cells)} of {len(policy)} cells")
    for state, loss in sorted(suboptimal_cells.items()):
        print(f"  {state}: {policy[state]} gives up {loss:.4f} expected return")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
