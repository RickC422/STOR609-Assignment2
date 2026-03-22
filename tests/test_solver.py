import pytest

from value_iteration import (
    FiniteMDP,
    Transition,
    bellman_optimality_update,
    value_iteration,
)

"""
Basic tests for the value iteration solver.

These tests check that the Bellman update works as expected for a
terminal state, that value iteration returns reasonable values and
policy on a small MDP, and that invalid solver inputs raise errors.
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


def test_bellman_update_for_terminal_state_is_zero():
    mdp = make_simple_mdp()

    values = {
        "A": 0.0,
        "B": 0.0,
        "T": 0.0,
    }

    new_value = bellman_optimality_update(mdp, "T", values)

    assert new_value == 0.0


def test_value_iteration_returns_expected_values_and_policy():
    mdp = make_simple_mdp()

    result = value_iteration(mdp, epsilon=1e-12, max_iterations=1000)

    assert result.values["T"] == pytest.approx(0.0, abs=1e-10)
    assert result.values["B"] == pytest.approx(2.0, abs=1e-8)
    assert result.values["A"] == pytest.approx(6.0, abs=1e-8)

    assert result.policy["A"] == "go"
    assert result.policy["B"] == "finish"
    assert result.policy["T"] is None

    assert result.iterations > 0
    assert result.final_delta < 1e-12


def test_value_iteration_rejects_non_positive_epsilon():
    mdp = make_simple_mdp()

    with pytest.raises(ValueError):
        value_iteration(mdp, epsilon=0.0)


def test_value_iteration_rejects_non_positive_max_iterations():
    mdp = make_simple_mdp()

    with pytest.raises(ValueError):
        value_iteration(mdp, max_iterations=0)