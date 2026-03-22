from dataclasses import dataclass
from typing import Dict, List, Optional

from .mdp import FiniteMDP, State
from .policy import action_value, greedy_policy_from_value


@dataclass
class ValueIterationResult:
    """
    Store the output of value iteration.
    """
    values: Dict[State, float]
    policy: Dict[State, Optional[object]]
    iterations: int
    final_delta: float
    deltas: List[float]


def bellman_optimality_update(
    mdp: FiniteMDP,
    state: State,
    values: Dict[State, float],
) -> float:
    """
    Compute one Bellman optimality update:

        (T V)(s) = max_a Q(s, a)

    Terminal states are fixed at 0.
    """
    if mdp.is_terminal(state):
        return 0.0

    best_value: Optional[float] = None

    for action in mdp.available_actions(state):
        current_value = action_value(mdp, state, action, values)

        if best_value is None or current_value > best_value:
            best_value = current_value

    return float(best_value)


def value_iteration(
    mdp: FiniteMDP,
    epsilon: float = 1e-10,
    max_iterations: int = 10000,
    initial_values: Optional[Dict[State, float]] = None,
) -> ValueIterationResult:
    """
    Run value iteration for a finite discounted MDP.

    Stopping rule:
        stop when max_s |V_new(s) - V_old(s)| < epsilon
    """
    if epsilon <= 0.0:
        raise ValueError("epsilon must be strictly positive.")

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive.")

    if initial_values is None:
        values_old = {}
        for state in mdp.states:
            values_old[state] = 0.0
    else:
        values_old = {}
        for state in mdp.states:
            values_old[state] = float(initial_values.get(state, 0.0))

    deltas: List[float] = []

    for iteration in range(1, max_iterations + 1):
        values_new: Dict[State, float] = {}
        delta = 0.0

        for state in mdp.states:
            values_new[state] = bellman_optimality_update(mdp, state, values_old)

            difference = abs(values_new[state] - values_old[state])
            if difference > delta:
                delta = difference

        deltas.append(delta)
        values_old = values_new

        if delta < epsilon:
            policy = greedy_policy_from_value(mdp, values_old)

            return ValueIterationResult(
                values=values_old,
                policy=policy,
                iterations=iteration,
                final_delta=delta,
                deltas=deltas,
            )

    raise RuntimeError(
        "Value iteration did not converge within max_iterations. "
        "Try increasing max_iterations or check the MDP specification."
    )