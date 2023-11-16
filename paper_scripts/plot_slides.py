import matplotlib.pyplot as plt
import os
import sys
from matplotlib.colors import ListedColormap

paths = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
         os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'src')]
sys.path.extend(paths)

from src.draw import draw_angle_stimulus, draw_stimulus_proximity

if __name__ == "__main__":
    bg_color = 'black'
    neutral_color = 'lightgray'
    green = '#6CC3BE'
    red = '#E78759'
    # colors = ['#6CC3BE', '#C1E3E0', '#F0DCD3', '#ECB196', '#E78759']
    # colors = [(108, 195, 190, 1), (193, 227, 224, 1), "#41b6c4", "#2c7fb8", "#253494"]

    width, height = 22, 22

    """
    params = [90, 112, 135, 157]
    #params = [4, 3, 2, 1, 0]
    param_names = [90, 68, 45, 23]
    ncols = len(params)
    fig, ax = plt.subplots(nrows=2, ncols=ncols, figsize=(12, 12))

    for i in range(ncols):
        #stimulus = draw_stimulus_proximity(distance=params[i], width=width, height=height)
        #stimulus = draw_angle_stimulus(angle=params[i], width=width, height=height, half=True)
        stimulus = draw_angle_stimulus(angle=params[i], width=width, height=height, half=False)
        img = stimulus["img"].copy()

        colors = ['black', 'lightgray', red]
        cmap = ListedColormap(colors)

        for x,y in zip(stimulus["coords"]["x_left"], stimulus["coords"]["y_left"]):
            img[y, x] = 2
        for x, y in zip(stimulus["coords"]["x_right"], stimulus["coords"]["y_right"]):
            img[y, x] = 2

        ax[1, i].imshow(img, cmap=cmap)
        ax[1, i].set_xticks([])
        ax[1, i].set_yticks([])
        ax[1, i].grid(False)

        img = stimulus["img"].copy()
        colors = ['black', 'lightgray', green]
        cmap = ListedColormap(colors)

        for x, y in zip(stimulus["coords"]["x_left"], stimulus["coords"]["y_left"]):
            img[y, x] = 2
        for x, y in zip(stimulus["coords"]["x_top"], stimulus["coords"]["y_top"]):
            img[y, x] = 2

        ax[0, i].imshow(img, cmap=cmap)
        ax[0, i].set_xticks([])
        ax[0, i].set_yticks([])
        ax[0, i].grid(False)
        ax[0, i].set_title(f"Angle diff\n {param_names[i]}", fontsize=22)
        fig.tight_layout()

    fig.savefig("C://Users/vzemliak/Desktop/slides/cont", bbox_inches='tight')

    
    red_color = '#ff9772'
    green_color = '#61deaf'
    cmap = ListedColormap([bg_color, neutral_color, red_color])

    width, height = 10, 10
    #stimulus = draw_stimulus_proximity(distance=1, width=width, height=height)
    stimulus = draw_angle_stimulus(angle=90, width=width, height=height, half=True)
    img = stimulus["img"].copy()

    for i in (1, 3):
        x1, y1 = stimulus["coords"]["x_left"][i], stimulus["coords"]["y_left"][i]
        img[y1, x1] = 2
    for i in (1, 4):
        x2, y2 = stimulus["coords"]["x_right"][i], stimulus["coords"]["y_right"][i]
        img[y2, x2] = 2

    x0, y0 = stimulus["coords"]["x_left"][0], stimulus["coords"]["y_left"][0]
    img[y0, x0] = 1

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(img, cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    #ax[0, i].set_title(f"Similarity", fontsize=22)
    fig.savefig("C://Users/vzemliak/Desktop/slides/sim_simple", bbox_inches='tight')
    # plt.show()

    cmap = ListedColormap([bg_color, neutral_color, red_color, green_color])
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

    for i_row in range(2):
        for i_col in range(2):
            img1 = stimulus["img"].copy()
            x1, y1 = stimulus["coords"]["x_left"][0], stimulus["coords"]["y_left"][0]
            img1[y1, x1] = 2

            if i_row == 0:
                x2, y2 = stimulus["coords"]["x_left"][i_col+2], stimulus["coords"]["y_left"][i_col+2]
            else:
                x2, y2 = stimulus["coords"]["x_right"][i_col+2], stimulus["coords"]["y_right"][i_col+2]

            img1[y1, x1] = 2
            img1[y2, x2] = 3

            ax[i_col, i_row].imshow(img1, cmap=cmap)
            ax[i_col, i_row].set_xticks([])
            ax[i_col, i_row].set_yticks([])
            ax[i_col, i_row].grid(False)
    fig.tight_layout()
    fig.savefig("plot1.png", bbox_inches='tight')
    """

    red_color = '#ff9772'
    green_color = '#61deaf'
    cmap = ListedColormap([bg_color, neutral_color, red_color])

    width, height = 10, 10
    # stimulus = draw_stimulus_proximity(distance=1, width=width, height=height)
    stimulus = draw_angle_stimulus(angle=90, width=width, height=height, half=False)
    img = stimulus["img"].copy()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

    # top
    for i in (1, 3):
        x1, y1 = stimulus["coords"]["x_left"][i], stimulus["coords"]["y_left"][i]
        img[y1, x1] = 2
    for i in (1, 4):
        x2, y2 = stimulus["coords"]["x_right"][i], stimulus["coords"]["y_right"][i]
        img[y2, x2] = 2

    x0, y0 = stimulus["coords"]["x_left"][0], stimulus["coords"]["y_left"][0]
    img[y0, x0] = 1

    im = ax[0].imshow(img, cmap=cmap)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].grid(False)

    img = stimulus["img"].copy()
    # bottom
    for i in (1, 3):
        x1, y1 = stimulus["coords"]["x_left"][i], stimulus["coords"]["y_left"][i]
        img[y1, x1] = 2
    for i in (1, 4):
        x2, y2 = stimulus["coords"]["x_top"][i], stimulus["coords"]["y_top"][i]
        img[y2, x2] = 2

    x0, y0 = stimulus["coords"]["x_left"][0], stimulus["coords"]["y_left"][0]
    img[y0, x0] = 1

    im = ax[1].imshow(img, cmap=cmap)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].grid(False)

    fig.tight_layout()
    fig.savefig("C://Users/vzemliak/Desktop/slides/cont_simple", bbox_inches='tight')
    # plt.show()



