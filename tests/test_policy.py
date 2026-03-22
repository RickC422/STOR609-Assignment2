import pytest

from value_iteration import FiniteMDP, Transition, action_value, greedy_policy_from_value

"""
Basic tests for policy-related functions.

These tests check that action values are computed correctly for a small
example MDP, and that a greedy policy is extracted correctly from a
given value function.
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


def test_action_value_is_computed_correctly():
    mdp = make_simple_mdp()

    values = {
        "A": 6.0,
        "B": 2.0,
        "T": 0.0,
    }

    q_go = action_value(mdp, "A", "go", values)
    q_stay = action_value(mdp, "A", "stay", values)
    q_finish = action_value(mdp, "B", "finish", values)

    assert q_go == pytest.approx(6.0, abs=1e-8)
    assert q_stay == pytest.approx(4.0, abs=1e-8)
    assert q_finish == pytest.approx(2.0, abs=1e-8)


def test_greedy_policy_from_value():
    mdp = make_simple_mdp()

    values = {
        "A": 6.0,
        "B": 2.0,
        "T": 0.0,
    }

    policy = greedy_policy_from_value(mdp, values)

    assert policy["A"] == "go"
    assert policy["B"] == "finish"
    assert policy["T"] is None