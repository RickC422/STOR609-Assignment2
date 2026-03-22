from dataclasses import dataclass
from typing import Dict, Hashable, Sequence, Set


State = Hashable
Action = Hashable


@dataclass(frozen=True)
class Transition:
    """
    One stochastic transition:
        state --action--> next_state
    with a probability and an immediate reward.
    """
    next_state: State
    probability: float
    reward: float


@dataclass
class FiniteMDP:
    """
    A finite discounted Markov Decision Process.

    Attributes
    ----------
    states
        All states in the MDP.
    actions
        A dictionary mapping each state to its available actions.
    transitions
        A nested dictionary such that transitions[s][a] is a list of
        Transition objects.
    gamma
        Discount factor, must satisfy 0 <= gamma < 1.
    terminal_states
        A set of terminal states.
    """
    states: Sequence[State]
    actions: Dict[State, Sequence[Action]]
    transitions: Dict[State, Dict[Action, Sequence[Transition]]]
    gamma: float
    terminal_states: Set[State]

    def __post_init__(self) -> None:
        if not (0.0 <= self.gamma < 1.0):
            raise ValueError("gamma must satisfy 0 <= gamma < 1.")

        state_set = set(self.states)

        for state in self.terminal_states:
            if state not in state_set:
                raise ValueError(f"Terminal state {state!r} is not in states.")

        for state in self.states:
            if state not in self.actions:
                raise ValueError(f"Missing actions for state {state!r}.")

        for state in self.states:
            available_actions = self.actions[state]

            if state in self.terminal_states:
                if len(available_actions) != 0:
                    raise ValueError(
                        f"Terminal state {state!r} should have no available actions."
                    )
                continue

            if len(available_actions) == 0:
                raise ValueError(
                    f"Non-terminal state {state!r} must have at least one action."
                )

            if state not in self.transitions:
                raise ValueError(
                    f"Missing transition dictionary for non-terminal state {state!r}."
                )

            for action in available_actions:
                if action not in self.transitions[state]:
                    raise ValueError(
                        f"Missing transitions for state {state!r}, action {action!r}."
                    )

                transition_list = self.transitions[state][action]

                if len(transition_list) == 0:
                    raise ValueError(
                        f"Empty transition list for state {state!r}, action {action!r}."
                    )

                probability_sum = 0.0

                for transition in transition_list:
                    if transition.next_state not in state_set:
                        raise ValueError(
                            f"Unknown next state {transition.next_state!r} "
                            f"from state {state!r}, action {action!r}."
                        )

                    if transition.probability < 0.0:
                        raise ValueError(
                            f"Negative probability from state {state!r}, action {action!r}."
                        )

                    probability_sum += transition.probability

                if abs(probability_sum - 1.0) > 1e-10:
                    raise ValueError(
                        f"Transition probabilities for state {state!r}, action {action!r} "
                        f"sum to {probability_sum}, not 1."
                    )

    def is_terminal(self, state: State) -> bool:
        return state in self.terminal_states

    def available_actions(self, state: State) -> Sequence[Action]:
        return self.actions[state]