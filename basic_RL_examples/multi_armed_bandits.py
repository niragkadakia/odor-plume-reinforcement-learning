"""
Simulate a multi-armed bandit.

Created by Nirag Kadakia at 16:38 10-24-2019
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""


import numpy as np
import matplotlib.pyplot as plt

num_trials = 1000
num_bandits = 2000
num_actions = 10

# Greedy means epsilon = 0. % of time wrong sub-optimal arbitrary chosen
eps = 0

# For each bandits, this will hold the actions (E[reward]) of each action
np.random.seed(0)
actions = np.random.normal(0, 1, (num_actions, num_bandits))

# Holds estimate of action value and number of times that action chosen
Q = np.random.normal(0, 1, (num_actions, num_bandits))
N = np.zeros(Q.shape)

# Holds average rewards on each step and % of time optimal action taken
avg_R = []
avg_opt_A = []

for _ in range(num_trials):
	
	# Action is optimum of estimated rewards, plus non-greedy randomness
	A = np.argmax(Q, axis=0)
	rand_flips = np.random.uniform(0, 1, len(A)) < eps
	A[rand_flips] = np.random.randint(0, 10, np.sum(rand_flips))
	avg_opt_A.append(np.mean(A == np.argmax(actions, axis=0)))
	
	N[A, range(num_bandits)] += 1
	R = np.random.normal(actions[A, range(num_bandits)], 1)
	
	Q[A, range(num_bandits)] += 1./(N[A, range(num_bandits)])*\
	  (R - Q[A, range(num_bandits)]) 
	
	avg_R.append(np.mean(R))
	
plt.plot(avg_R)
plt.plot(avg_opt_A)
plt.show()

