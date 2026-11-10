"""Hyperparameter sweep, used by the tune-hyperparameters skill (.claude/skills/).

Every hyperparameter option takes a comma-separated list of values. The sweep trains one agent
per combination and seed, and scores the learned policy against value iteration's optimal
policy for the same living reward and discount factor.

    python3 -m qlearning.sweep --learning-rate-start 0.1,0.5,1.0 --seeds 10
    python3 -m qlearning.sweep --optimal-policies --living-reward 0,-0.04,-0.5,-2

--optimal-policies skips learning. It prints value iteration's policy for each living reward and
discount factor, which shows what each choice asks the robot to do.
"""

from __future__ import annotations

import argparse
import itertools
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, fields, replace

from qlearning.environment import START_STATE, VacuumEnvironment
from qlearning.training import (
    Hyperparameters,
    evaluate_policy,
    extract_policy,
    extract_state_values,
    format_policy_grid,
    train,
)
from qlearning.value_iteration import (
    OPTIMAL_LOSS_TOLERANCE,
    compute_optimal_policy,
    compute_optimal_state_values,
    compute_policy_losses,
)
# Evaluation environments use seeds far from the training seeds
EVALUATION_SEED_OFFSET = 10_000
SWEPT_FIELD_NAMES: tuple[str, ...] = tuple(
    field.name for field in fields(Hyperparameters) if field.name != "seed"
)


@dataclass(frozen=True)
class SweepRow:
    """Scores for one combination of hyperparameters, averaged over the seeds.

    mean_optimal_cell_count: cells (out of 15) where the learned action is optimal
    worst_optimal_cell_count: the same count for the worst seed
    mean_start_value_error: |V(start) learned - V*(start)|, how well the values were learned
    mean_goal_rate: fraction of greedy test episodes that exited at +10
    mean_average_return: total reward per greedy test episode
    """

    hyperparameters: Hyperparameters
    mean_optimal_cell_count: float
    worst_optimal_cell_count: int
    mean_start_value_error: float
    mean_goal_rate: float
    mean_average_return: float


def parse_number_list(text: str, number_type: Callable[[str], float]) -> list[float]:
    """'0.1, 0.5,1' -> [0.1, 0.5, 1.0]. Raises ValueError for anything that is not a number."""
    values = [number_type(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError(f"no values in {text!r}")
    return values


def build_grid(
    base_hyperparameters: Hyperparameters, values_by_field: dict[str, Sequence[float]]
) -> list[Hyperparameters]:
    """Every combination of the listed values. Fields that are not listed keep their base value."""
    field_names = list(values_by_field)
    return [
        replace(base_hyperparameters, **dict(zip(field_names, combination)))
        for combination in itertools.product(*(values_by_field[name] for name in field_names))
    ]


def evaluate_configuration(
    hyperparameters: Hyperparameters,
    seed_count: int,
    evaluation_episode_count: int,
    first_seed: int = 0,
) -> SweepRow:
    """Train once per seed and score each learned policy against the optimal one."""
    checking_environment = VacuumEnvironment(
        living_reward=hyperparameters.living_reward,
        slip_probability=hyperparameters.slip_probability,
    )
    optimal_state_values = compute_optimal_state_values(
        checking_environment, hyperparameters.discount_factor
    )
    optimal_cell_counts = []
    start_value_errors = []
    goal_rates = []
    average_returns = []
    for seed in range(first_seed, first_seed + seed_count):
        result = train(replace(hyperparameters, seed=seed))
        policy = extract_policy(result.agent, result.environment)
        losses = compute_policy_losses(
            checking_environment, policy, hyperparameters.discount_factor, optimal_state_values
        )
        optimal_cell_counts.append(
            sum(1 for loss in losses.values() if loss <= OPTIMAL_LOSS_TOLERANCE)
        )
        learned_start_value = extract_state_values(result.agent, result.environment)[START_STATE]
        start_value_errors.append(abs(learned_start_value - optimal_state_values[START_STATE]))
        evaluation = evaluate_policy(
            policy,
            VacuumEnvironment(
                living_reward=hyperparameters.living_reward,
                slip_probability=hyperparameters.slip_probability,
                seed=seed + EVALUATION_SEED_OFFSET,
            ),
            evaluation_episode_count,
            hyperparameters.max_steps_per_episode,
        )
        goal_rates.append(evaluation.goal_rate)
        average_returns.append(evaluation.average_return)
    return SweepRow(
        hyperparameters=hyperparameters,
        mean_optimal_cell_count=sum(optimal_cell_counts) / seed_count,
        worst_optimal_cell_count=min(optimal_cell_counts),
        mean_start_value_error=sum(start_value_errors) / seed_count,
        mean_goal_rate=sum(goal_rates) / seed_count,
        mean_average_return=sum(average_returns) / seed_count,
    )


def rank_rows(rows: list[SweepRow]) -> list[SweepRow]:
    """Best first: most optimal cells, then the best worst seed, then the smallest value error."""
    return sorted(
        rows,
        key=lambda row: (
            -row.mean_optimal_cell_count,
            -row.worst_optimal_cell_count,
            row.mean_start_value_error,
        ),
    )


def format_table(rows: list[SweepRow], varied_field_names: list[str]) -> str:
    """A Markdown table with one line per row, readable in the terminal and ready to paste into
    docs/hyperparameter-tuning.md. Only the hyperparameters that were varied get a column."""
    headers = varied_field_names + [
        "optimal cells",
        "worst seed",
        "V(start) error",
        "goal rate",
        "avg return",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in rows:
        cells = [str(asdict(row.hyperparameters)[name]) for name in varied_field_names] + [
            f"{row.mean_optimal_cell_count:.1f}/15",
            f"{row.worst_optimal_cell_count}/15",
            f"{row.mean_start_value_error:.3f}",
            f"{row.mean_goal_rate:.1%}",
            f"{row.mean_average_return:.3f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def describe_optimal_policies(
    base_hyperparameters: Hyperparameters,
    living_rewards: Sequence[float],
    discount_factors: Sequence[float],
    evaluation_episode_count: int,
) -> str:
    """Value iteration's policy for each living reward and discount factor, and how it does."""
    sections = []
    for living_reward, discount_factor in itertools.product(living_rewards, discount_factors):
        environment = VacuumEnvironment(
            living_reward=living_reward,
            slip_probability=base_hyperparameters.slip_probability,
            seed=EVALUATION_SEED_OFFSET,
        )
        policy = compute_optimal_policy(environment, discount_factor)
        start_value = compute_optimal_state_values(environment, discount_factor)[START_STATE]
        evaluation = evaluate_policy(
            policy,
            environment,
            evaluation_episode_count,
            base_hyperparameters.max_steps_per_episode,
        )
        sections.append(
            f"living_reward {living_reward}, discount_factor {discount_factor}\n"
            f"{format_policy_grid(policy)}\n"
            f"V*(start) {start_value:.3f}; following it: reached +10 in {evaluation.goal_rate:.1%}, "
            f"average {evaluation.average_step_count:.1f} steps, "
            f"average return {evaluation.average_return:.3f}"
        )
    return "\n\n".join(sections)


def build_argument_parser() -> argparse.ArgumentParser:
    """One comma-separated list option per hyperparameter, plus options for the sweep itself."""
    defaults = Hyperparameters()
    parser = argparse.ArgumentParser(
        prog="python3 -m qlearning.sweep",
        description="Train once per combination of values and seed, and rank the results.",
    )
    for field_name in SWEPT_FIELD_NAMES:
        parser.add_argument(
            "--" + field_name.replace("_", "-"),
            default=None,
            metavar="VALUES",
            help=f"comma-separated values (default: {getattr(defaults, field_name)})",
        )
    parser.add_argument("--seeds", type=int, default=10, help="training runs per combination")
    parser.add_argument("--first-seed", type=int, default=0, help="seeds are first..first+seeds-1")
    parser.add_argument("--evaluation-episodes", type=int, default=500)
    parser.add_argument(
        "--optimal-policies",
        action="store_true",
        help="print value iteration's policy per living reward and discount factor; no learning",
    )
    return parser


def main(command_line_arguments: list[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(command_line_arguments)
    base_hyperparameters = Hyperparameters()
    values_by_field: dict[str, list[float]] = {}
    for field_name in SWEPT_FIELD_NAMES:
        option_text = getattr(arguments, field_name)
        if option_text is not None:
            number_type = type(getattr(base_hyperparameters, field_name))
            values_by_field[field_name] = parse_number_list(option_text, number_type)

    if arguments.optimal_policies:
        print(
            describe_optimal_policies(
                base_hyperparameters,
                values_by_field.get("living_reward", [base_hyperparameters.living_reward]),
                values_by_field.get("discount_factor", [base_hyperparameters.discount_factor]),
                arguments.evaluation_episodes,
            )
        )
        return 0

    configurations = build_grid(base_hyperparameters, values_by_field)
    rows = [
        evaluate_configuration(
            configuration, arguments.seeds, arguments.evaluation_episodes, arguments.first_seed
        )
        for configuration in configurations
    ]
    varied_field_names = [name for name, values in values_by_field.items() if len(values) > 1]
    print(
        f"{len(configurations)} combination(s) x {arguments.seeds} seeds "
        f"(seeds {arguments.first_seed}-{arguments.first_seed + arguments.seeds - 1})"
    )
    print(format_table(rank_rows(rows), varied_field_names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
