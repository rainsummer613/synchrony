import matplotlib.pyplot as plt
import os
import sys
from matplotlib.colors import ListedColormap

paths = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
         os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'src')]
sys.path.extend(paths)

from src.draw import draw_angle_stimulus, draw_stimulus_proximity

experiment_params = {'continuity': (90, 112, 135, 157, 180),
                     'proximity': (4, 3, 2, 1, 0),
                     'similarity': (90, 112, 135, 157, 180)
                     }
def get_matrices(exp='similarity', height=20, width=20):
    stimuli = {}
    for param in experiment_params[exp]:

        if exp == 'similarity':
            stimuli[180-param] = draw_angle_stimulus(angle=param, strength=1, width=width, height=height, half=True)['img']

        elif exp == 'proximity':
            stimuli[param] = draw_stimulus_proximity(distance=param, width=width, height=height)['img']

        elif exp == 'continuity':
            stimuli[180-param] = draw_angle_stimulus(angle=param, strength=1, width=width, height=height, half=False)['img']

    return stimuli

if __name__ == '__main__':

    experiments = ('proximity', 'similarity', 'continuity')
    exp = experiments[2]
    stimuli = get_matrices(exp=exp)

    bg_color = 'black'
    neutral_color = 'lightgray'
    cmap = ListedColormap([bg_color, neutral_color])

    exp_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'plots', exp)
    plot_dir = os.path.join(exp_dir, 'stimuli')
    os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots(nrows=len(stimuli), ncols=1, figsize=(20, 20))

    for i, (k, v) in enumerate(stimuli.items()):
        im = ax[i].imshow(stimuli[k], cmap=cmap)
        #ax[i].axis('off')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_ylabel(k, fontsize=24, labelpad=10)
        ax[i].grid(False)

    plot_path = os.path.join(plot_dir, 'all.png')
    fig.savefig(plot_path, bbox_inches='tight')
    plt.show()
