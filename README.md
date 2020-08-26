# Navigating Turbulent Odor Plumes Using Reinforcement Learning



<figure>
    <img src = "data/plume_fig2.png" width="500" align="left">
</figure>

This repository explores the use of traditional Reinforcement Learning *temporal-difference* methods such as the  [Q-learning](https://en.wikipedia.org/wiki/Q-learning) algorithm to understand how agents can learn to navigate to the source of a 2D turbulent odor plume. Turbulent odor plumes are spatiotemporally complex: odor signals arrive in random, unpredictable bursts, and odor concentration does follow continuous gradients, making plume navigation a difficult task. Surprisingly, there has been no published work on applying RL algorithms to turbulent odor plume navigation. 




## Q-learning algorithm
Q-learning updates a state-action value function, $Q(a, s)$, which quantifies the value of taking action $a$ when in state $s$. This function is updated in time using the Q-learning algorithm. At each timestep, the navigator takes an action $A$ to go from state $S$ to state $S'$, using whatever policy (greedy, $\epsilon$-greedy, etc.), where it receives a reward $R$. Then $Q(A, S)$ is updated using a *greedy* update rule (hence *off-policy*):

$\begin{align}
 Q(A, S) \leftarrow Q(A, S) + \alpha[R + \gamma \max_a Q(a, S')  - Q(A, S)]
\end{align}$

Here, $\alpha < 1$ and $\gamma < 1$ are update step-size and learning rate -- both hyperparameters in the simulation.