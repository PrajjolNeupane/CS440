"""Value iteration (RL slides L1 p32). FOR CHECKING ONLY.

Value iteration needs T and R, which the robot is not given, so the Q-learning agent never uses
this module. The tests and the command-line report use it to find the true optimal policy and
check what Q-learning learned against it.
"""

from __future__ import annotations

from qlearning.environment import Action, State, VacuumEnvironment

# A cell counts as optimal when its action gives up less than this much expected return
OPTIMAL_LOSS_TOLERANCE = 1e-6


def q_value_from_state_values(
    environment: VacuumEnvironment,
    state_values: dict[State, float],
    state: State,
    action: Action,
    discount_factor: float,
) -> float:
    """Q(s, a) = sum over s' of T(s, a, s') * [R(s, a, s') + discount_factor * V(s')]."""
    total = 0.0
    for outcome in environment.transition_outcomes(state, action):
        future_value = 0.0 if outcome.done else discount_factor * state_values[outcome.next_state]
        total += outcome.probability * (outcome.reward + future_value)
    return total


def compute_optimal_state_values(
    environment: VacuumEnvironment,
    discount_factor: float,
    tolerance: float = 1e-10,
    max_iterations: int = 100_000,
) -> dict[State, float]:
    """V*(s) for every open cell, repeating the Bellman update until the values stop changing."""
    state_values = {state: 0.0 for state in environment.open_states()}
    for _ in range(max_iterations):
        new_state_values = {
            state: max(
                q_value_from_state_values(environment, state_values, state, action, discount_factor)
                for action in environment.legal_actions(state)
            )
            for state in state_values
        }
        largest_change = max(
            abs(new_state_values[state] - state_values[state]) for state in state_values
        )
        state_values = new_state_values
        if largest_change < tolerance:
            break
    return state_values


def compute_policy_losses(
    environment: VacuumEnvironment,
    policy: dict[State, Action],
    discount_factor: float,
    optimal_state_values: dict[State, float] | None = None,
) -> dict[State, float]:
    """For each cell, how much expected return the policy's action gives up: V*(s) - Q*(s, pi(s)).

    0 means the action is optimal. Comparing values instead of action names means a cell with
    two equally good actions counts as correct whichever one the policy picked.
    Pass optimal_state_values when checking many policies for the same problem, to skip
    recomputing them.
    """
    if optimal_state_values is None:
        optimal_state_values = compute_optimal_state_values(environment, discount_factor)
    return {
        state: optimal_state_values[state]
        - q_value_from_state_values(
            environment, optimal_state_values, state, action, discount_factor
        )
        for state, action in policy.items()
    }


def compute_optimal_policy(
    environment: VacuumEnvironment, discount_factor: float
) -> dict[State, Action]:
    """pi*(s) = argmax over a of Q*(s, a). Ties go to the first legal action."""
    optimal_state_values = compute_optimal_state_values(environment, discount_factor)
    return {
        state: max(
            environment.legal_actions(state),
            key=lambda action: q_value_from_state_values(
                environment, optimal_state_values, state, action, discount_factor
            ),
        )
        for state in environment.open_states()
    }
