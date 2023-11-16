import matplotlib.pyplot as plt
import os
import sys
from matplotlib.colors import ListedColormap

paths = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
	os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'src')]
sys.path.extend(paths)

from src.draw import draw_angle_stimulus, draw_stimulus_proximity

def plot(**kwargs):

    exp = kwargs.get('exp', 'similarity')
    height, width = kwargs.get('height', 20), kwargs.get('width', 20)
    segment = kwargs.get('segment', 'cont')

    bg_color = 'black'
    neutral_color = 'lightgray'
    rsync_high_color = '#ff9772' #'darksalmon'
    rsync_low_color = '#9ad1c3' #'teal'
    cmap = ListedColormap([bg_color, rsync_low_color])

    if exp == 'similarity':
        angle = kwargs.get('angle', 90)
        stimulus = draw_angle_stimulus(angle, strength=1, width=width, height=height, half=True)
        if angle == 180:
            cmap = ListedColormap([bg_color, rsync_high_color])

    elif exp == 'proximity':
        distance = kwargs.get('distance', 0)
        stimulus = draw_stimulus_proximity(width=width, strength=1, height=height, distance=distance)
        if distance == 0:
            cmap = ListedColormap([bg_color, rsync_high_color])
 
    elif exp == 'continuity':
        angle = kwargs.get('angle', 90)
        stimulus = draw_angle_stimulus(angle=angle, strength=1, width=width, height=height, half=False)

        if segment == 'cont':
            stimulus['img'][stimulus['coords']['y_top'], stimulus['coords']['x_top']] = 1
            stimulus['img'][stimulus['coords']['y_bottom'], stimulus['coords']['x_bottom']] = 1
            stimulus['img'][stimulus['coords']['y_left'], stimulus['coords']['x_left']] = 2
            stimulus['img'][stimulus['coords']['y_right'], stimulus['coords']['x_right']] = 2

        elif segment == 'diff':
            stimulus['img'][stimulus['coords']['y_right'], stimulus['coords']['x_right']] = 1
            stimulus['img'][stimulus['coords']['y_bottom'], stimulus['coords']['x_bottom']] = 1
            stimulus['img'][stimulus['coords']['y_top'], stimulus['coords']['x_top']] = 2
            stimulus['img'][stimulus['coords']['y_left'], stimulus['coords']['x_left']] = 2

        colors = [bg_color, neutral_color, rsync_low_color]
        if segment == 'cont':
            colors = [bg_color, neutral_color, rsync_high_color]
        cmap = ListedColormap(colors)
    
    return {'stim': stimulus['img'], 'cmap': cmap}

if __name__ == '__main__':

    height, width = 20, 20
    data_dir = 	 os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data')
    fig_dir = os.path.join(data_dir, 'figures')

    # proximity
    proximity_4 = plot(exp='proximity', distance=4, height=height, width=width)
    proximity_0 = plot(exp='proximity', distance=0, height=height, width=width)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout()
    #fig.suptitle('Stimuli', fontsize=27)
    ax1.imshow(proximity_4['stim'], cmap=proximity_4['cmap'])
    ax2.imshow(proximity_0['stim'], cmap=proximity_0['cmap'])

    ax1.axis('off')
    ax2.axis('off')
    fig.savefig(f'{fig_dir}/proximity/stimuli.svg', bbox_inches='tight')
    plt.show()

    # similarity
    similarity_90 = plot(exp='similarity', angle=90, height=height, width=width)
    similarity_180 = plot(exp='similarity', angle=180, height=height, width=width)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.tight_layout()

    ax1.imshow(similarity_90['stim'], cmap=similarity_90['cmap'])
    ax2.imshow(similarity_180['stim'], cmap=similarity_180['cmap'])

    ax1.axis('off')
    ax2.axis('off')
    fig.savefig(f'{fig_dir}/similarity/stimuli.svg', bbox_inches='tight')
    plt.show()

    # continuity
    continuity_90_cont = plot(exp='continuity', angle=90, height=height, width=width, segment='cont')
    continuity_157_cont = plot(exp='continuity', angle=157, height=height, width=width, segment='cont')
    continuity_90_diff = plot(exp='continuity', angle=90, height=height, width=width, segment='diff')
    continuity_157_diff = plot(exp='continuity', angle=157, height=height, width=width, segment='diff')

    fig, ax = plt.subplots(2, 2)
    ax[0,0].imshow(continuity_90_cont['stim'], cmap=continuity_90_cont['cmap'])
    ax[0,1].imshow(continuity_157_cont['stim'], cmap=continuity_157_cont['cmap'])
    ax[1,0].imshow(continuity_90_diff['stim'], cmap=continuity_90_diff['cmap'])
    ax[1,1].imshow(continuity_157_diff['stim'], cmap=continuity_157_diff['cmap'])

    ax[0,0].axis('off')
    ax[0,1].axis('off')
    ax[1,0].axis('off')
    ax[1,1].axis('off')

    fig.tight_layout()
    fig.savefig(f'{fig_dir}/continuity/stimuli.svg', bbox_inches='tight')
    plt.show()
    

    
