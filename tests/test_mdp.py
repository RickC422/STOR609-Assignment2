import pytest

from src.value_iteration import FiniteMDP, Transition


"""
Basic tests for the FiniteMDP data structure.

These tests check that a valid small MDP can be created, and that
some invalid inputs raise errors. In particular, they check transition
probabilities and terminal-state action rules.
"""

def make_simple_mdp():
    states = ["A", "B", "T"]

    actions = {
        "A": ["go", "stay"],
        "B": ["finish"],
        "T": [],
    }

    transitions = {
        "A": {
            "go": [
                Transition(next_state="B", probability=1.0, reward=5.0),
            ],
            "stay": [
                Transition(next_state="A", probability=1.0, reward=1.0),
            ],
        },
        "B": {
            "finish": [
                Transition(next_state="T", probability=1.0, reward=2.0),
            ],
        },
    }

    terminal_states = {"T"}

    return FiniteMDP(
        states=states,
        actions=actions,
        transitions=transitions,
        gamma=0.5,
        terminal_states=terminal_states,
    )


def test_valid_mdp_is_created():
    mdp = make_simple_mdp()

    assert set(mdp.states) == {"A", "B", "T"}
    assert mdp.is_terminal("T") is True
    assert mdp.is_terminal("A") is False
    assert list(mdp.available_actions("A")) == ["go", "stay"]


def test_probabilities_must_sum_to_one():
    states = ["A", "B", "T"]

    actions = {
        "A": ["go"],
        "B": ["finish"],
        "T": [],
    }

    transitions = {
        "A": {
            "go": [
                Transition(next_state="B", probability=0.8, reward=5.0),
            ],
        },
        "B": {
            "finish": [
                Transition(next_state="T", probability=1.0, reward=2.0),
            ],
        },
    }

    terminal_states = {"T"}

    with pytest.raises(ValueError):
        FiniteMDP(
            states=states,
            actions=actions,
            transitions=transitions,
            gamma=0.5,
            terminal_states=terminal_states,
        )


def test_terminal_state_must_have_no_actions():
    states = ["A", "T"]

    actions = {
        "A": ["go"],
        "T": ["loop"],
    }

    transitions = {
        "A": {
            "go": [
                Transition(next_state="T", probability=1.0, reward=1.0),
            ],
        },
        "T": {
            "loop": [
                Transition(next_state="T", probability=1.0, reward=0.0),
            ],
        },
    }

    terminal_states = {"T"}

    with pytest.raises(ValueError):
        FiniteMDP(
            states=states,
            actions=actions,
            transitions=transitions,
            gamma=0.5,
            terminal_states=terminal_states,
        )