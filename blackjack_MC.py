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

# States are hand sum, dealer's showing card, and usable ace?
states = np.zeros((10, 12, 2), dtype=object)
for idx in range(10):
	for idy in range(12):
		for idz in range(2):
			states[idx, idy, idz] = []

num_hands = 100000
hit_val = 20

np.random.seed(0)
player_cards = np.random.randint(2, 15, (num_hands, 2))
dealer_cards = np.random.randint(2, 15, (num_hands, 2))
player_hits = np.random.randint(2, 15, num_hands*10)
hit_card = -1
for iT in range(num_hands):
	d_cards = dealer_cards[iT]*(dealer_cards[iT] < 10) + \
						10*(dealer_cards[iT] >= 10)*(dealer_cards[iT] < 14) + \
						11*(dealer_cards[iT] == 14)
	dealer_showing = d_cards[0]
	cards = player_cards[iT]
	vals = np.zeros(2)
	vals[:] = cards[:]
	vals[(vals >= 10)*(vals < 14)] = 10
	vals[vals == 14] = 11
	aces = np.where(vals == 11)[0]
		
	# States to be visited during this hand
	# Player sums are 12 to 21 -- if smaller than 12, ignore
	hand_states = []
	
	# Two aces; special case
	if np.sum(vals) == 22:
		vals[0] = 1
	
	# Hit up to hit_val, reduce aces if needed
	if np.sum(vals) >= hit_val:
		if np.sum(vals) - 12 >= 0:	
			hand_states.append([np.sum(vals) - 12, dealer_showing, len(aces) > 0])
	else:
		while np.sum(vals) < hit_val:
			if np.sum(vals) - 12 >= 0:	
				hand_states.append([np.sum(vals) - 12, dealer_showing, len(aces) > 0])
			hit_card += 1
			next_card = player_hits[hit_card]
			val_next_card = next_card*(next_card < 10) + \
							10*(next_card >= 10)*(next_card < 14) + \
							11*(next_card == 14)
			vals = np.append(vals, [val_next_card])
			
			# Demote aces if above 21
			aces = np.where(vals == 11)[0]
			if (len(aces) > 0) and (np.sum(vals) > 21):
				vals[np.where(vals == 11)[0][0]] = 1
	
	player_sum = np.sum(vals)
	dealer_sum = np.sum(d_cards)
	for state in hand_states:
		state_int = (int(state[0]), int(state[1]), int(state[2]))
		if (player_sum > dealer_sum)*(player_sum <= 21):
			states[state_int].append(1)
		elif (player_sum == dealer_sum):
			states[state_int].append(0)
		else:
			states[state_int].append(-1)
	
avgs = np.zeros((10, 12, 2))
for idx in range(10):
	for idy in range(12):
		for idz in range(2):
			avgs[idx, idy, idz] = np.mean(states[idx, idy, idz])
plt.imshow(avgs[:, :, 0])
plt.show()
		