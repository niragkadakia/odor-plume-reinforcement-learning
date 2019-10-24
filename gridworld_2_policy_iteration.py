"""

"""

import numpy as np
import matplotlib.pyplot as plt

gamma = 1

num_s0 = 16
num_s1 = 16
num_actions = 4

prob = np.zeros((num_s0, num_s1, num_actions))
reward = -np.ones((num_s0, num_s1, num_actions))
action = np.zeros((num_s0, num_actions))

# Equal transition probabilities for all directions
action[:] = 0.25

# s, s', a
col = 1
row = 4
u = 0
r = 1
d = 2
l = 3

i = 1
prob[i, i, u] = 1
prob[i, i + col, r] = 1
prob[i, i + row, d] = 1 

# Terminal state
prob[i, 0, l] = 1 

i = 2
prob[i, i, u] = 1
prob[i, i + col, r] = 1
prob[i, i + row, d] = 1 
prob[i, i - col, l] = 1 

i = 3
prob[i, i, u] = 1
prob[i, i, r] = 1
prob[i, i + row, d] = 1
prob[i, i - col, l] = 1

i = 4
# Terminal state
prob[i, 0, u] = 1

prob[i, i + col, r] = 1
prob[i, i + row, d] = 1
prob[i, i, l] = 1

i = 8
prob[i, i - row, u] = 1
prob[i, i + col, r] = 1
prob[i, i + row, d] = 1
prob[i, i, l] = 1

for i in [ 5,  6,  
		   9, 10]:
	prob[i, i - row, u] = 1
	prob[i, i + col, r] = 1
	prob[i, i + row, d] = 1
	prob[i, i - col, l] = 1

i = 7
prob[i, i - row, u] = 1
prob[i, i, r] = 1
prob[i, i + row, d] = 1
prob[i, i - col, l] = 1

i = 11
prob[i, i - row, u] = 1
prob[i, i, r] = 1
prob[i, i - col, l] = 1

# Terminal state
prob[i, 0, d] = 1
	
i = 12
prob[i, i - row, u] = 1
prob[i, i + col, r] = 1
prob[i, i, d] = 1
prob[i, i, l] = 1

i = 13
prob[i, i - row, u] = 1
prob[i, i + col, r] = 1
prob[i, i, d] = 1
prob[i, i - col, l] = 1

i = 14
prob[i, i - row, u] = 1

# Terminal state
prob[i, 0, r] = 1
prob[i, i, d] = 1
prob[i, i - col, l] = 1

V_old = np.zeros(num_s0)
action_old = np.ones((num_s0, num_actions))*0.25
	
for i in range(39):
	
	# Policy evaluation	
	err = 1
	while err > 1e-8:
		V_new = np.zeros(num_s0)
		for iC in range(num_s0):
			for jC in range(num_s1):
				for aC in range(num_actions):
					V_new[iC] += action_old[iC, aC]*prob[iC, jC, aC]*\
							(reward[iC, jC, aC] + gamma*V_old[jC])
		err = max(V_old - V_new)
		V_old[:] = V_new[:]
		
	# Policy iteration
	action_new = np.zeros(action_old.shape)
	for iC in range(num_s0):
		value_iC = np.zeros(num_actions)
		for aC in range(num_actions):
			for jC in range(num_s1):
				value_iC[aC] += prob[iC, jC, aC]*\
						(reward[iC, jC, aC] + gamma*V_new[jC])
		opt_actions = np.where(value_iC == np.amax(value_iC))[0]
		for iO in opt_actions:
			action_new[iC, iO] = 1./len(opt_actions)
		
	action_old[:] = action_new[:]
	
print (V_new)
print (action_new)
	