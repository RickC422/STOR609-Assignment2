# Value Iteration and the Textbook Algorithm

This note compares my value iteration design with the textbook pseudocode in in figure 9.16 in section 9.5.2 of Artificial Intelligence: Foundations and Computational Agents 2nd edition.

My design follows the same main idea as the book. I start from an initial value function, update the value of each non-terminal state by the Bellman optimality rule, and stop when the change becomes small. After that, I extract the policy by choosing the action with the largest expected return under the final value function. So the core method is the same.

The main differences come from implementation needs. I use transition-based rewards, written as \(R(s' \mid s,a)\), instead of state-action rewards \(R(s,a)\). I do this because the coursework gives the grid world as a transition table. Each row already contains the next state, probability, and reward, so this form is more direct in code and avoids extra conversion.

My version makes terminal states explicit, while the textbook pseudocode leaves this modelling choice implicit.

Another practical change is how I store values during iteration. The textbook writes the sequence \(V_0, V_1, V_2, \dots\). In code, I only keep `V_old` and `V_new`. This is enough for the algorithm and makes the implementation simpler.

The textbook says to repeat the update “until termination”. In my design, I make this precise by using  
`delta = max_s |V_new(s) - V_old(s)|`  
and stopping when `delta < epsilon`. I added this because a real program needs a clear stopping rule.

I also include some simple input checks. For example, I check that `0 <= gamma < 1`, that terminal states belong to the state set, and that transition probabilities sum to `1`. These checks are not part of the mathematics, but they make the program safer and easier to debug.

Overall, my design stays close to the textbook algorithm. The Bellman update and the policy extraction are unchanged. The differences are only practical changes to make the method easier to implement, easier to explain, and better suited to this coursework.