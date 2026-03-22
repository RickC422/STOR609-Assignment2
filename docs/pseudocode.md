Algorithm: Value Iteration for a Finite Discounted MDP

Input:
    states S
    actions A(s)
    transition model P(s' | s, a)
    reward model R(s' | s, a)
    discount factor gamma
    terminal states T
    tolerance epsilon > 0
    maximum number of iterations max_iterations
    optional initial values

Output:
    approximate optimal value function V
    greedy policy pi
    number of iterations
    final delta
    list of delta values

Step 1: Initialise
    If no initial values are given:
        For each state s in S:
            V_old(s) <- 0
    Otherwise:
        Use the given initial values
        If some state is missing, use 0 for that state

Step 2: Repeat value updates
    For iteration = 1, 2, ..., max_iterations:

        delta <- 0

        For each state s in S:

            If s is a terminal state:
                V_new(s) <- 0

            Else:
                best_value <- None

                For each action a in A(s):
                    q <- 0

                    For each transition from (s, a):
                        q <- q + P(s' | s, a) * [ R(s' | s, a) + gamma * V_old(s') ]

                    If best_value is None or q > best_value:
                        best_value <- q

                V_new(s) <- best_value

            change <- |V_new(s) - V_old(s)|

            If change > delta:
                delta <- change

        Store delta in the delta list
        Replace V_old by V_new

        If delta < epsilon:
            Stop the iteration

Step 3: Extract a greedy policy
    For each state s in S:

        If s is a terminal state:
            pi(s) <- None

        Else:
            best_action <- None
            best_action_value <- None

            For each action a in A(s):
                q <- sum over s' of
                     P(s' | s, a) * [ R(s' | s, a) + gamma * V_old(s') ]

                If best_action_value is None or q > best_action_value:
                    best_action_value <- q
                    best_action <- a

            pi(s) <- best_action

Step 4: Return results
    If delta < epsilon:
        Return the final values V_old, the greedy policy pi, the number of iterations,
        the final delta, and the list of all delta values

    Otherwise:
        Raise an error indicating that value iteration did not converge within
        max_iterations
