from typing import Dict, Optional

from .mdp import Action, FiniteMDP, State


def action_value(
    mdp: FiniteMDP,
    state: State,
    action: Action,
    values: Dict[State, float],
) -> float:
    """
    Compute the action-value for one state-action pair:

        Q(s, a) = sum_{s'} P(s'|s,a) * [ R(s'|s,a) + gamma * V(s') ]
    """
    total = 0.0

    for transition in mdp.transitions[state][action]:
        total += transition.probability * (
            transition.reward + mdp.gamma * values[transition.next_state]
        )

    return total


def greedy_policy_from_value(
    mdp: FiniteMDP,
    values: Dict[State, float],
) -> Dict[State, Optional[Action]]:
    """
    Extract a greedy policy from a value function.

    For terminal states, the policy is None.
    """
    policy: Dict[State, Optional[Action]] = {}

    for state in mdp.states:
        if mdp.is_terminal(state):
            policy[state] = None
            continue

        best_action: Optional[Action] = None
        best_value: Optional[float] = None

        for action in mdp.available_actions(state):
            current_value = action_value(mdp, state, action, values)

            if best_value is None or current_value > best_value:
                best_value = current_value
                best_action = action

        policy[state] = best_action

    return policy