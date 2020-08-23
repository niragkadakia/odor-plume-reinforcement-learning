"""
Helper functions

Created by Nirag Kadakia at 18:12 08-21-2020
This work is licensed under the 
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 
International License. 
To view a copy of this license, 
visit http://creativecommons.org/licenses/by-nc-sa/4.0/.
"""

import matplotlib.pyplot as plt

def gen_plot(width, height):
    """
    Generic plot for all figures to get right linewidth, font sizes, 
    and bounding boxes
    """

    fig = plt.figure(figsize=(width, height))
    ax = plt.subplot(111)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.tick_params(direction='in', length=3, width=0.5)
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(0.5)

    return fig, ax