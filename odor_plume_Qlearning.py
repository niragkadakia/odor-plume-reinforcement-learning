"""
Q-learning for a plume with simplified action space

Created by Nirag Kadakia at 12:40 06-23-2020
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, visit 
http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import numpy as np
import matplotlib.pyplot as plt
import pkl

vid_file = r'C:\\Users\\nk479\\Dropbox (emonetlab)\\users\\nirag_kadakia'\
            '\\data\\optogenetic-assay\\assay\\stim_videos\\IS_1.avi'		

# Number of actins
num_actions = 4

# Odor signal threshold
thresh = 100

# Duration threshold
freq_bins = [-0.01, 2.5, 100]

# Motion step size
xy_step = 2

# Step size
alpha = 0.9

# Learning rate
gamma = 0.9

def run():
    num_points = 2000
    xs = [np.random.randint(1000, 1150)]
    ys = [np.random.randint(450-200, 450+200)]
    
    vs = [2]
    states = [0]
    
    # States are below and above 10% duration in last 2 seconds
    # Q is a function of n states and 4 actions
    Q = np.zeros((len(freq_bins) - 1, num_actions))

    # To hold odor signal in last bit of time
    odors = np.zeros(300)

    cap = cv2.VideoCapture(vid_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
    for idx in range(num_points - 1):

        # Actions: get all optimal ones and choose randomly from those
        opt_actions = np.argwhere(Q[states[-1]] == np.amax(Q[states[-1]])).T[0]
        action = np.random.choice(opt_actions)
        randomize = np.random.uniform(0, 1)
        if randomize > 0.95:
            action = np.random.randint(num_actions)

        # Update position based on action
        x = xs[-1]
        y = ys[-1]
        
        if action == 0:
            v = 2
        elif action == 1:
            v = vs[-1]
        elif action == 2:
            v = np.random.choice(4)
        elif action == 3:
            if ys[-1] > 450:
                v = 3
            elif ys[-1] <= 450:
                v = 1
        
        if vs[-1] == 0:
            x += xy_step
        elif vs[-1] == 1:
            y += xy_step
        elif vs[-1] == 2:
            x -= xy_step
        elif vs[-1] == 3:
            y -= xy_step
        
        x = max(min(x, 1200), 100)
        y = max(min(y, 450+250), 450-250)
        xs.append(x)
        ys.append(y)
        vs.append(v)
        
        if idx > 5000:
            cap = cv2.VideoCapture(vid_file)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
        ret, frm = cap.read()

        whf = 1.*(frm[ys[-1], xs[-1], 2] > thresh)

        # Update recent reward vector
        prev_odor_mean = np.mean(odors)
        odors = np.roll(odors, -1)
        odors[-1] = whf

        # State is 1 if mean odor duration is enough
        freq_mean = np.sum((np.diff(odors) > 0)*
                           np.exp(-np.arange(len(odors) - 1)/100)[::-1])
        #print (freq_mean)
        new_state = np.digitize(freq_mean, freq_bins, right=True) - 1

        # Reward is 1 if got whiff otherwise 0
        reward = whf  

        # Q-learning step
        Q[states[-1], action] += alpha*(reward + gamma*max(Q[new_state]) 
										- Q[states[-1], action])
        states.append(new_state)
        print (new_state, end = " ", flush=True)

        if (x < 150)*(abs(y - 450) < 50):
            break

    xs.extend([np.nan]*(num_points - len(xs)))
    ys.extend([np.nan]*(num_points - len(ys)))
    
    return np.asarray(xs), np.asarray(ys), Q
	
# Run 100 models
data_to_save = None
for idx in range(100):
    print (idx)
    xs, ys, Qs = run()
    if data_to_save is None:
        data_to_save = dict()
        data_to_save['x'] = xs
        data_to_save['y'] = ys
        data_to_save['Q0'] = Qs[0]
        data_to_save['Q1'] = Qs[1]
    else:
        data_to_save['x'] = np.vstack((data_to_save['x'].T, xs.T)).T
        data_to_save['y'] = np.vstack((data_to_save['y'].T, ys.T)).T
        data_to_save['Q0'] = np.vstack((data_to_save['Q0'].T, Qs[0].T)).T
        data_to_save['Q1'] = np.vstack((data_to_save['Q1'].T, Qs[1].T)).T
    Q_all += Qs
	
# Plot the trajectories
fig, ax = gen_plot(5, 3)
skip = 1
for idx in range(data_to_save['Q0'].shape[1]):
    color_num = np.arange(np.sum(np.isfinite(data_to_save['x'][::skip, idx])))
    color_num = color_num/color_num[-1]
    colors = plt.cm.plasma_r(color_num)
    plt.scatter(data_to_save['x'][::skip, idx], data_to_save['y'][::skip, idx], 
                c=colors, s=0.5)
plt.xticks(np.arange(-1000, 1000, 65))
plt.yticks(np.arange(-1000, 2000, 65))
plt.ylim(450 - 300, 450 + 300)
plt.xlim(200, 1100)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_aspect('equal', 'box')
ax.spines['right'].set_visible(True)
ax.spines['top'].set_visible(True)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(0.5)
#plt.savefig('RL_trajs.png', dpi=600)
plt.show()

# Plot the optimal actions for each of the states
hi_freq = np.mean(data_to_save['Q1'], axis=1)
hi_freq = np.exp(hi_freq)
hi_freq /= np.sum(hi_freq)
lo_freq = np.mean(data_to_save['Q0'], axis=1)
lo_freq = np.exp(lo_freq)
lo_freq /= np.sum(lo_freq)
x = range(num_actions)

fig, ax = gen_plot(1.5, 2.5)
plt.scatter(x, lo_freq, s=55, color='k')
plt.plot(np.arange(num_actions), lo_freq, color='k')
plt.scatter(np.arange(num_actions), hi_freq, s=55, color='r')
plt.plot(np.arange(num_actions), hi_freq, color='r')
plt.xticks(range(num_actions))
ax.set_xticklabels([])
plt.ylim(0, 0.75)
plt.savefig('RL_Q_functions.svg')