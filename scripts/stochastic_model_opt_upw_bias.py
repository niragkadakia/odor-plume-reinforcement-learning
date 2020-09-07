"""
Run stochastic turning model over suite of upwind biases

Created by Nirag Kadakia at 21:37 08-23-2020
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append('../src')
from utils import gen_plot
from models import stochastic_turning

avgs = []
taus = [0.5, 1, 1.5, 2, 2.5, 3.5, 5, 6, 7, 8, 9, 10, 13, 16, 19]
for tau in taus:
	a = stochastic_turning()
	a.vid_file = r'../data/intermittent_smoke.avi'		
	a.x0_min = 1000
	a.xy_step = 5
	a.num_walkers = 1000
	a.def_actions()
	a.num_steps = 5000
	
	freq_bins = [-1, 0.5, 2, 100]
	a.def_states(freq_bins=freq_bins, tau=tau, odor_vec_len=2000)

	xs, ys, Q, Q_terminal, path_lengths = a.run()

	print (np.mean(path_lengths < a.num_steps))
	avgs.append(np.mean(path_lengths < a.num_steps))
	
plt.plot(taus, avgs)
plt.show()
quit()

fig, ax = gen_plot(2, 4)
colors = plt.cm.Reds(np.linspace(0.3, 1.0, a._num_states))
for iS in range(a._num_states):
    plt.plot(Q_terminal[iS], color=colors[iS], lw=2, 
            label='$W_{freq}$ > %.1f' % freq_bins[iS])

plt.xlabel('Actions', fontsize=15)
plt.ylabel('Q-value', fontsize=15)
plt.xticks(range(a._num_actions), fontsize=12)
plt.yticks(np.arange(600, 1600, 250), fontsize=12)
ax.set_xticklabels(['Upwind', 'Downwind'], rotation=45)
plt.legend(bbox_to_anchor=(1.1, 1.05), fontsize=13)
plt.show()	
temp = 125

Q_norm = (np.exp(Q_terminal.T/temp)/np.sum(np.exp(Q_terminal/temp), axis=-1)).T

fig, ax = gen_plot(2, 4)
colors = plt.cm.Reds(np.linspace(0.3, 1.0, a._num_states))
for iS in range(a._num_states):
    avg_freq = max((freq_bins[iS] + freq_bins[iS + 1])/2, 0)
    plt.plot(Q_norm[iS], color=colors[iS], lw=2, 
             label='$W_{freq}$ > %.1f' % freq_bins[iS])

plt.xlabel('Actions', fontsize=15)
plt.ylabel(r'$p_{T=%s}($action|states)' % temp, fontsize=15)
plt.xticks(range(a._num_actions), fontsize=12)
plt.yticks([0, 0.25, 0.5, 0.75], fontsize=12)
ax.set_xticklabels(['Upwind', 'Downwind'], rotation=45)
plt.legend(bbox_to_anchor=(1.1, 1.05), fontsize=13)
plt.show()