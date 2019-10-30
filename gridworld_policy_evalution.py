"""
Update value function in RL using iteration.

Created by Nirag Kadakia at 16:38 10-24-2019
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""


import numpy as np
import matplotlib.pyplot as plt

gamma = 0.9

num_s0 = 25
num_s1 = 25
num_actions = 4

prob = np.zeros((num_s0, num_s1, num_actions))
reward = np.zeros((num_s0, num_s1, num_actions))
action = np.zeros((num_s0, num_actions))

# Equal transition probabilities for all directions
action[:] = 0.25

# s, s', a
col = 1
row = 5
u = 0
r = 1
d = 2
l = 3

i = 0
prob[i, i, u] = 1
prob[i, i + col, r] = 1
prob[i, i + row, d] = 1
prob[i, i, l] = 1

for i in [1, 2, 3]:
	prob[i, i, u] = 1
	prob[i, i + col, r] = 1
	prob[i, i + row, d] = 1 
	prob[i, i - col, l] = 1 

i = 4
prob[i, i, u] = 1
prob[i, i, r] = 1
prob[i, i + row, d] = 1
prob[i, i - col, l] = 1

for i in [5, 10, 15]:
	prob[i, i - row, u] = 1
	prob[i, i + col, r] = 1
	prob[i, i + row, d] = 1
	prob[i, i, l] = 1

for i in [ 6,  7,  8, 
		  11, 12, 13, 
		  16, 17, 18]:
	prob[i, i - row, u] = 1
	prob[i, i + col, r] = 1
	prob[i, i + row, d] = 1
	prob[i, i - col, l] = 1

for i in [9, 14, 19]:
	prob[i, i - row, u] = 1
	prob[i, i, r] = 1
	prob[i, i + row, d] = 1
	prob[i, i - col, l] = 1

i = 20
prob[i, i - row, u] = 1
prob[i, i + col, r] = 1
prob[i, i, d] = 1
prob[i, i, l] = 1

for i in [21, 22, 23]:
	prob[i, i - row, u] = 1
	prob[i, i + col, r] = 1
	prob[i, i, d] = 1
	prob[i, i - col, l] = 1

i = 24
prob[i, i - row, u] = 1
prob[i, i, r] = 1
prob[i, i, d] = 1
prob[i, i - col, l] = 1

# Special states
prob[1, :, :] = 0
prob[3, :, :] = 0
for a in range(4):
	prob[1, 21, a] = 1
	prob[3, 13, a] = 1

# Rewards

for i in range(5):
	reward[i, i, u] =  -1
for i in range(0, 25, 5):
	reward[i, i, l] =  -1
for i in range(4, 25, 5):
	reward[i, i, r] =  -1
for i in range(20, 25):
	reward[i, i, d] =  -1

# Special states
reward[1, :, :] = 0
reward[3, :, :] = 0
for a in range(4):
	reward[1, 21, a] = 10.
	reward[3, 13, a] = 5.

# Get value function by iterative update
V_old = np.zeros(num_s0)	
for i in range(100):
	V_new = np.zeros(num_s0)
	for iC in range(num_s0):
		for jC in range(num_s1):
			for aC in range(num_actions):
				V_new[iC] += action[iC, aC]*prob[iC, jC, aC]*\
						(reward[iC, jC, aC] + gamma*V_old[jC])
	V_old[:] = V_new[:]
	
print (V_new)