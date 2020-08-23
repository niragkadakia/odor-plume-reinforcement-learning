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
values = np.zeros((10, 12, 2), dtype=object)
for idx in range(10):
	for idy in range(12):
		for idz in range(2):
			values[idx, idy, idz] = []

num_hands = 500000
player_hit_val = 19

np.random.seed(0)
player_cards = np.random.randint(2, 15, (num_hands, 2))
dealer_cards = np.random.randint(2, 15, (num_hands, 2))
hits = np.random.randint(2, 15, num_hands*10)
hit_card = -1
for iT in range(num_hands):
	d_cards = dealer_cards[iT]
	d_vals = np.zeros(2)
	d_vals[:] = d_cards[:]
	d_vals[(d_vals >= 10)*(d_vals < 14)] = 10
	d_vals[d_vals == 14] = 11
	dealer_showing = d_vals[0]
	
	p_cards = player_cards[iT]
	p_vals = np.zeros(2)
	p_vals[:] = p_cards[:]
	p_vals[(p_vals >= 10)*(p_vals < 14)] = 10
	p_vals[p_vals == 14] = 11
	p_aces = np.where(p_vals == 11)[0]
	

	# Player:
		
	# States to be visited during this hand
	# Player sums are 12 to 21 -- if smaller than 12, ignore
	hand_states = []
	
	# Two aces; special case
	if np.sum(p_vals) == 22:
		p_vals[0] = 1
	
	# Hit up to player_hit_val, reduce aces if needed
	if np.sum(p_vals) > player_hit_val:
		if np.sum(p_vals) - 12 >= 0:	
			hand_states.append([np.sum(p_vals) - 12, dealer_showing, len(p_aces) > 0])
	else:
		while np.sum(p_vals) <= player_hit_val:
			if np.sum(p_vals) - 12 >= 0:	
				hand_states.append([np.sum(p_vals) - 12, dealer_showing, len(p_aces) > 0])
			hit_card += 1
			next_card = hits[hit_card]
			val_next_card = next_card*(next_card < 10) + \
							10*(next_card >= 10)*(next_card < 14) + \
							11*(next_card == 14)
			p_vals = np.append(p_vals, [val_next_card])
			
			# Demote p_aces if above 21
			p_aces = np.where(p_vals == 11)[0]
			if (len(p_aces) > 0) and (np.sum(p_vals) > 21):
				p_vals[np.where(p_vals == 11)[0][0]] = 1
	
	player_sum = np.sum(p_vals)
	
	
	# Dealer:
	
	if np.sum(d_vals) == 22:
		d_vals[0] = 1
	dealer_hit_val = 17
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
	
	for state in hand_states:
		state_int = (int(state[0]), int(state[1]), int(state[2]))
		if player_sum > 21:
			values[state_int].append(-1)
		elif dealer_sum > 21:
			values[state_int].append(1)
		elif player_sum == dealer_sum:
			values[state_int].append(0)
		elif player_sum > dealer_sum:
			values[state_int].append(1)
		elif player_sum < dealer_sum:
			values[state_int].append(-1)

values = values[:, 2:, :]
values = np.roll(values, 1, axis=1)
avgs = np.zeros((10, 10, 2))
for idx in range(10):
	for idy in range(10):
		for idz in range(2):
			avgs[idx, idy, idz] = np.mean(values[idx, idy, idz])
plt.imshow(avgs[:, :, 0].T)
plt.show()
		
plt.imshow(avgs[:, :, 1].T)
plt.show()
