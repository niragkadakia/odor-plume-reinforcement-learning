"""
RL models

Created by Nirag Kadakia at 21:37 08-23-2020
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import sys
import numpy as np
import cv2


class centerline_upwind(object, num_walkers=100, alpha=0.9, gamma=0.9, 
						epsilon=0.05):
	"""
	Reinforced learning simulation where navigators can i) go straight, 
	ii) go upwind, iii) turn randomly, or iv), turn toward the 
	plume centerline.
	"""

	def __init__(self):
		"""
		Define video file, video, and RL parameters.
		"""

		self.vid_file = r'data/intermittent_smoke.avi'
		self.beg_frm = 500
		self.max_frms = 5000
		self.xy_step = 2
		self.num_steps = 3000
		self.x0_min = 500
		self.x0_max = 1400
		self.max_x = 1500
		self.max_y = 840
		self.y_center = 450
		self.y0_spread = 200

		# Number of parallel agents
		self.num_walkers = num_walkers
		self.alpha = alpha
		self.gamma = gamma

		# epsilon-greedy action selection
		self.epsilon = epsilon

	def def_states(self, odor_threshold=100, odor_vec_len=500, 
				   freq_bins=[-0.01, 2, 100], tau=2):
		"""
		States are delineated by odor frequency, which is defined
		by an exponential timescale
		"""

		self.odor_threshold = odor_threshold
		self.odor_vec_len = odor_vec_len
		self.freq_bins = freq_bins
		self.tau = tau
		self._num_states = len(self.freq_bins) - 1

	def def_actions(self):
		"""
		Actions defined by motion `v' in run script
		"""
		self._num_actions = 4

	def update_actions(self, Q, states, iS):
		"""
		Get optimal actions from Q function
		"""

		# Q for state and action at last timestep
		last_Q = Q[iS, states[iS], :, np.arange(self.num_walkers)].T

		# Binary array of size (num_actions, num_walkers) for previous state. 
		# Equals `1' if that action is optimal (there can be degeneracies).
		opt_action_arr = 1.*(last_Q == np.amax(last_Q, axis=0))

		# From the optimal actions, choose randomly by random shifts 
		# around `1' + sorting
		opt_action_arr *= np.random.normal(1, 0.01, opt_action_arr.shape)
		actions = np.argmax(opt_action_arr, axis=0)

		# Epsilon-greedy action selection randomizes actions with prob=epsilon
		actions_to_flip = np.random.uniform(0, 1, self.num_walkers) > \
						 (1. - self.epsilon)
		random_actions = np.random.randint(0, self._num_actions, 
										   self.num_walkers)
		actions[actions_to_flip] = random_actions[actions_to_flip]

		return actions

	def update_xys(self, xs, ys, vs, actions, iS):
		"""
		Update navigator positions given selected actions
		"""

		_xs = xs[iS]
		_ys = ys[iS]

		_xs[_xs > self.max_x] = self.max_x
		_ys[_ys > self.max_y] = self.max_y
		_xs[_xs < 0] = 0
		_ys[_ys < 0] = 0

		# Actions give change in velocity v, 0-4; 0=right (downwind), 
		# 1=up (crosswind), 2=left (upwind), 3=down (crosswind). 
		# Actions: 
		#	0: go upwind (_v = 2)
		#	1: keep same direction (_v = v[iS])
		#	2: randomize direction (_v = random from 0 to 3)
		#	3: go toward centerline (_v = 1 or 3 depending on y)
		_vs = np.empty(self.num_walkers)*np.nan
		_vs[actions == 0] = 2
		_vs[actions == 1] = vs[iS, actions == 1]
		_vs[actions == 2] = np.random.randint(0, 4, np.sum(actions == 2))
		_vs[actions == 3] = (ys[iS, actions == 3] > self.y_center)*3 + \
							(ys[iS, actions == 3] <= self.y_center)*1

		_xs[_vs == 0] += self.xy_step
		_ys[_vs == 1] += self.xy_step
		_xs[_vs == 2] -= self.xy_step
		_ys[_vs == 3] -= self.xy_step

		xs[iS + 1] = _xs
		ys[iS + 1] = _ys
		vs[iS + 1] = _vs

		return xs, ys, vs

	def update_states(self, xs, ys, odors, states, frm, fps, iS):
		"""
		Update the odor vector and the next state based on odor signal
		"""

		# Whiff is binary; 1 if greater than odor threshold
		whfs = 1.*(frm[(ys[iS], xs[iS])] > self.odor_threshold)

		# Append new signal to odor vector (odors in recent past)
		odors = np.roll(odors, -1, axis=0)
		odors[-1] = whfs

		# Freq of odor whiffs by discounting recent odor hits (diff of odors)
		# with a 2s decaying exponential
		whf_freqs = np.sum((np.diff(odors, axis=0) > 0).T*
						  np.exp(-np.arange(odors.shape[0] - 1).T
						  /self.tau/fps)[::-1], axis=1)

		states[iS + 1] = np.digitize(whf_freqs, self.freq_bins, right=True) - 1

		return odors, states

	def run(self, seed=0):
		"""
		Do 1 run
		"""

		np.random.seed(seed)

		cap = cv2.VideoCapture(self.vid_file)
		cap.set(cv2.CAP_PROP_POS_FRAMES, self.beg_frm)
		fps = cap.get(cv2.CAP_PROP_FPS)

		xs = np.zeros((self.num_steps, self.num_walkers), dtype=int)
		ys = np.zeros((self.num_steps, self.num_walkers), dtype=int)
		vs = np.zeros((self.num_steps, self.num_walkers), dtype=int)
		states = np.zeros((self.num_steps, self.num_walkers), dtype=int)
		Q = np.ones((self.num_steps, self._num_states, 
					 self._num_actions, self.num_walkers))

		# Binary vectory indicating if path reached source
		ends = np.zeros(self.num_walkers)

		# Length of trajectory from initial point to source in timesteps
		path_lengths = np.ones(self.num_walkers)*self.num_steps

		# Initial x,y are random; initial motion is upwind (v = 2)
		xs[0] = np.random.randint(self.x0_min, 
								  self.x0_max, 
								  self.num_walkers)
		ys[0] = np.random.randint(self.y_center - self.y0_spread, 
								  self.y_center + self.y0_spread, 
								  self.num_walkers)
		vs[0] = np.ones(self.num_walkers)*2

		# Holds odor vector to define states
		odors = np.zeros((self.odor_vec_len, self.num_walkers))

		for iS in range(self.num_steps - 1):
			sys.stdout.write('\r')
			sys.stdout.write("{:.1f}%".format((100/(self.num_steps - 1)*iS)))
			sys.stdout.flush()

			# Update actions and positions based on optimal actions
			actions = self.update_actions(Q, states, iS)
			xs, ys, vs = self.update_xys(xs, ys, vs, actions, iS)

			# Mark those that reached the source and how long to do so
			_ends = (xs[iS] < 150)*(abs(ys[iS] - self.y_center) < 50)
			path_lengths[(ends == 0)*_ends] = iS
			ends[_ends] = 1

			# Grab odor signal from current frame; cycle video if reached end
			if iS > self.max_frms:
				cap = cv2.VideoCapture(vid_file)
				cap.set(cv2.CAP_PROP_POS_FRAMES, self.beg_frm)
			ret, frm = cap.read()
			frm = frm[:, :, 2]

			# Update odor vector, states, and rewards
			odors, states = self.update_states(xs, ys, odors, states, 
											   frm, fps, iS)
			rewards = (odors[-1] - odors[-2]) > 0

			# All Qs at this timestep are the same as last, except the current
			# (state, action) pair, which is being updated
			Q[iS + 1] = Q[iS]
			Q[iS + 1, states[iS], actions, np.arange(self.num_walkers)] = \
				Q[iS, states[iS], actions, np.arange(self.num_walkers)] + \
				self.alpha*(rewards + self.gamma*np.amax(Q[iS, states[iS + 1], 
				:, np.arange(self.num_walkers)].T, axis=0) - \
				Q[iS, states[iS], actions, np.arange(self.num_walkers)])

		return xs, ys, Q, path_lengths


class centerline_upwind_downwind_turns(object, num_walkers=100, alpha=0.9, 
									   gamma=0.9, epsilon=0.05):
	"""
	Reinforced learning simulation where navigators can i) go straight, 
	ii) turn upwind, iii) turn downwind, iv) possibly turn to centerline 
	"""

	def __init__(self):
		"""
		Define video file, video, and RL parameters.
		"""

		self.vid_file = r'data/intermittent_smoke.avi'
		self.beg_frm = 500
		self.max_frms = 5000
		self.xy_step = 2
		self.num_steps = 3000
		self.x0_min = 500
		self.x0_max = 1400
		self.max_x = 1500
		self.max_y = 840
		self.y_center = 450
		self.y0_spread = 200

		# Number of parallel agents
		self.num_walkers = num_walkers
		self.alpha = alpha
		self.gamma = gamma

		# epsilon-greedy action selection
		self.epsilon = epsilon

	def def_states(self, odor_threshold=100, odor_vec_len=500, 
				   freq_bins=[-0.01, 2, 100], tau=2):
		"""
		States are delineated by odor frequency, which is defined
		by an exponential timescale
		"""

		self.odor_threshold = odor_threshold
		self.odor_vec_len = odor_vec_len
		self.freq_bins = freq_bins
		self.tau = tau
		self._num_states = len(self.freq_bins) - 1

	def def_actions(self, centerline=True):
		"""
		Actions defined by motion `v' in run script
		"""
		
		if centerline == True:
			self._num_actions = 4
		else:
			self._num_actions = 3

	def update_actions(self, Q, states, iS):
		"""
		Get optimal actions from Q function
		"""

		# Q for state and action at last timestep
		last_Q = Q[iS, states[iS], :, np.arange(self.num_walkers)].T

		# Binary array of size (num_actions, num_walkers) for previous state. 
		# Equals `1' if that action is optimal (there can be degeneracies).
		opt_action_arr = 1.*(last_Q == np.amax(last_Q, axis=0))

		# From the optimal actions, choose randomly by random shifts 
		# around `1' + sorting
		opt_action_arr *= np.random.normal(1, 0.01, opt_action_arr.shape)
		actions = np.argmax(opt_action_arr, axis=0)

		# Epsilon-greedy action selection randomizes actions with prob=epsilon
		actions_to_flip = np.random.uniform(0, 1, self.num_walkers) > \
						 (1. - self.epsilon)
		random_actions = np.random.randint(0, self._num_actions, 
										   self.num_walkers)
		actions[actions_to_flip] = random_actions[actions_to_flip]

		return actions

	def update_xys(self, xs, ys, vs, actions, iS):
		"""
		Update navigator positions given selected actions
		"""

		_xs = xs[iS]
		_ys = ys[iS]

		_xs[_xs > self.max_x] = self.max_x
		_ys[_ys > self.max_y] = self.max_y
		_xs[_xs < 0] = 0
		_ys[_ys < 0] = 0

		# Actions give change in velocity v, 0-4; 0=right (downwind), 
		# 1=up (crosswind), 2=left (upwind), 3=down (crosswind). 
		# Actions: 
		#	0: go straight
		#	1: turn upwind (_v += 1 if < 2, _v-=1 if > 2)
		#	2: turn downwind (opposite to 0)
		#	3: turn toward centerline 
		_vs = np.empty(self.num_walkers)*np.nan
		rnd = np.random.choice([1, 3])
		_vs[actions == 0] = vs[iS, actions == 0]
		_vs[actions == 1] = 2*(vs[iS, actions == 1] == 1) + \
							2*(vs[iS, actions == 1] == 3) + \
							2*(vs[iS, actions == 1] == 2) + \
							rnd*(vs[iS, actions == 1] == 0)
		_vs[actions == 2] = 0*(vs[iS, actions == 2] == 1) + \
							0*(vs[iS, actions == 2] == 3) + \
							0*(vs[iS, actions == 2] == 0) + \
							rnd*(vs[iS, actions == 2] == 2)
		_vs[actions == 3] = (ys[iS, actions == 3] > self.y_center)*3 + \
							(ys[iS, actions == 3] <= self.y_center)*1
		_vs[actions == 1] = _vs[actions == 1] % 4
		_vs[actions == 2] = _vs[actions == 2] % 4
		
		_xs[_vs == 0] += self.xy_step
		_ys[_vs == 1] += self.xy_step
		_xs[_vs == 2] -= self.xy_step
		_ys[_vs == 3] -= self.xy_step

		xs[iS + 1] = _xs
		ys[iS + 1] = _ys
		vs[iS + 1] = _vs

		return xs, ys, vs

	def update_states(self, xs, ys, odors, states, frm, fps, iS):
		"""
		Update the odor vector and the next state based on odor signal
		"""

		# Whiff is binary; 1 if greater than odor threshold
		whfs = 1.*(frm[(ys[iS], xs[iS])] > self.odor_threshold)

		# Append new signal to odor vector (odors in recent past)
		odors = np.roll(odors, -1, axis=0)
		odors[-1] = whfs

		# Freq of odor whiffs by discounting recent odor hits (diff of odors)
		# with a 2s decaying exponential
		whf_freqs = np.sum((np.diff(odors, axis=0) > 0).T*
						  np.exp(-np.arange(odors.shape[0] - 1).T
						  /self.tau/fps)[::-1], axis=1)

		states[iS + 1] = np.digitize(whf_freqs, self.freq_bins, right=True) - 1

		return whfs, odors, states

	def run(self, seed=0):
		"""
		Do 1 run
		"""

		np.random.seed(seed)

		cap = cv2.VideoCapture(self.vid_file)
		cap.set(cv2.CAP_PROP_POS_FRAMES, self.beg_frm)
		fps = cap.get(cv2.CAP_PROP_FPS)

		xs = np.zeros((self.num_steps, self.num_walkers), dtype=int)
		ys = np.zeros((self.num_steps, self.num_walkers), dtype=int)
		vs = np.zeros((self.num_steps, self.num_walkers), dtype=int)
		states = np.zeros((self.num_steps, self.num_walkers), dtype=int)
		Q = np.ones((self.num_steps, self._num_states, 
					 self._num_actions, self.num_walkers))

		# Binary vectory indicating if path reached source
		ends = np.zeros(self.num_walkers)

		# Length of trajectory from initial point to source in timesteps
		path_lengths = np.ones(self.num_walkers)*self.num_steps

		# Initial x,y are random; initial motion is upwind (v = 2)
		xs[0] = np.random.randint(self.x0_min, 
								  self.x0_max, 
								  self.num_walkers)
		ys[0] = np.random.randint(self.y_center - self.y0_spread, 
								  self.y_center + self.y0_spread, 
								  self.num_walkers)
		vs[0] = np.ones(self.num_walkers)*2

		# Holds odor vector to define states
		odors = np.zeros((self.odor_vec_len, self.num_walkers))

		for iS in range(self.num_steps - 1):
			sys.stdout.write('\r')
			sys.stdout.write("{:.1f}%".format((100/(self.num_steps - 1)*iS)))
			sys.stdout.flush()

			# Update actions and positions based on optimal actions
			actions = self.update_actions(Q, states, iS)
			xs, ys, vs = self.update_xys(xs, ys, vs, actions, iS)

			# Mark those that reached the source and how long to do so
			_ends = (xs[iS] < 150)*(abs(ys[iS] - self.y_center) < 50)
			path_lengths[(ends == 0)*_ends] = iS
			ends[_ends] = 1

			# Grab odor signal from current frame; cycle video if reached end
			if iS > self.max_frms:
				cap = cv2.VideoCapture(vid_file)
				cap.set(cv2.CAP_PROP_POS_FRAMES, self.beg_frm)
			ret, frm = cap.read()
			frm = frm[:, :, 2]

			# Update odor vector, states, and rewards
			whfs, odors, states = self.update_states(xs, ys, odors, states, 
											   frm, fps, iS)
			rewards = (odors[-1] - odors[-2]) > 0

			# All Qs at this timestep are the same as last, except the current
			# (state, action) pair, which is being updated
			Q[iS + 1] = Q[iS]
			Q[iS + 1, states[iS], actions, np.arange(self.num_walkers)] = \
				Q[iS, states[iS], actions, np.arange(self.num_walkers)] + \
				self.alpha*(rewards + self.gamma*np.amax(Q[iS, states[iS + 1], 
				:, np.arange(self.num_walkers)].T, axis=0) - \
				Q[iS, states[iS], actions, np.arange(self.num_walkers)])

		return xs, ys, Q, path_lengths