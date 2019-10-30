"""
Monte Carlo evaluation of value function in Blackjack

Created by Nirag Kadakia at 16:38 10-30-2019
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import numpy as np
import matplotlib.pyplot as plt

# 10 player and dealer states, ace/no ace; values are 0 (hit), 1(stick)
policy = np.zeros((10, 12, 2))
values = np.zeros((10, 12, 2, 2))
returns = np.zeros((10, 12, 2, 2), dtype=object)
for i in range(10):
	for j in range(12):
		for k in range(2):
			for l in range(2):
				returns[i, j, k, l] = []

policy[8, :, :] = 1
policy[9, :, :] = 1

num_hands = 10000
player_hit_val = 19
dealer_hit_val = 17

np.random.seed(0)
player_cards = np.random.randint(2, 15, (num_hands, 2))
dealer_cards = np.random.randint(2, 15, (num_hands, 2))
hits = np.random.randint(2, 15, num_hands*50)
hit_card = -1

for iT in range(num_hands):
	print (iT)
	states_in_hand = []

	d_cards = dealer_cards[iT]
	d_vals = np.zeros(2)
	d_vals[:] = d_cards[:]
	d_vals[(d_vals >= 10)*(d_vals < 14)] = 10
	d_vals[d_vals == 14] = 11
	d_state = int(d_vals[0])
	
	p_cards = player_cards[iT]
	p_vals = np.zeros(2)
	p_vals[:] = p_cards[:]
	p_vals[(p_vals >= 10)*(p_vals < 14)] = 10
	p_vals[p_vals == 14] = 11
	num_p_aces = 1*(len(np.where(p_vals == 11)[0]) > 0)
	p_state = int(np.sum(p_vals) - 12)

	
	if p_state < 10:
		action = policy[p_state, d_state, num_p_aces]
		states_in_hand.append([p_state, d_state, num_p_aces])
	while action == 0:
		hit_card += 1
		next_card = hits[hit_card]
		val_next_card = next_card*(next_card < 10) + \
						10*(next_card >= 10)*(next_card < 14) + \
						11*(next_card == 14)
		p_vals = np.append(p_vals, val_next_card)
		num_p_aces = 1*(len(np.where(p_vals == 11)[0]) > 0)
		p_state = int(np.sum(p_vals) - 12)
		if p_state < 10:
			states_in_hand.append([p_state, d_state, num_p_aces])
			action = policy[p_state, d_state, num_p_aces]
		else:
			action = 1
			
	player_sum = np.sum(p_vals)
	
	if np.sum(d_vals) == 22:
		d_vals[0] = 1
	
	while np.sum(d_vals) <= dealer_hit_val:
		hit_card += 1
		next_card = hits[hit_card]
		val_next_card = next_card*(next_card < 10) + \
						10*(next_card >= 10)*(next_card < 14) + \
						11*(next_card == 14)
		d_vals = np.append(d_vals, [val_next_card])
		
		# Demote aces if above 21
		d_aces = np.where(d_vals == 11)[0]
		if (len(d_aces) > 0) and (np.sum(d_vals) > 21):
			d_vals[np.where(d_vals == 11)[0][0]] = 1
	
	dealer_sum = np.sum(d_vals)
	
	# Hit on first N - 1
	for iS, state in enumerate(states_in_hand):
		
		if iS == len(states_in_hand) - 1:
			action = 1
		else: 
			action = 0
		
		state_int = (int(state[0]), int(state[1]), int(state[2]), action)
		if player_sum > 21:
			returns[state_int].append(-1)
		elif dealer_sum > 21:
			returns[state_int].append(1)
		elif player_sum == dealer_sum:
			returns[state_int].append(0)
		elif player_sum > dealer_sum:
			returns[state_int].append(1)
		elif player_sum < dealer_sum:
			returns[state_int].append(-1)
	
	# Update value
	for i in range(10):
		for j in range(12):
			for k in range(2):
				for l in range(2):
					if len(returns[i, j, k, l]) != 0:
						values[i, j, k, l] = np.mean(returns[i, j, k, l])
						
	# Update policy
	for i in range(10):
		for j in range(12):
			for k in range(2):
				policy[i, j, k] = np.argmax(values[i, j, k, :])
				
plt.imshow(policy[:, :, 0].T)
plt.show()
		
plt.imshow(policy[:, :, 1].T)
plt.show()