import matplotlib.pyplot as plt
import numpy as np
from skimage import draw

def line_22(x_start, y_start, width):
    y_list = []
    x_list = []
    
    for i in range(0,100,2):
        if x_start-i-1 > 0:
            x_i = list(range(x_start-i-1, x_start-i+1))
            y_i = [y_start-i//2] * len(x_i)
            
            x_list.extend(x_i)
            y_list.extend(y_i)
        else:
            break
    return x_list, y_list

def line_45(x_start, y_start, width):
    y_list = []
    x_list = []
    
    for i in range(0,100,1):
        if x_start-i-2 > 0:
            x_i = list(range(x_start-i-1, x_start-i))
            y_i = [y_start-i] * len(x_i)
                       
            x_list.extend(x_i)
            y_list.extend(y_i)                       
        else:
            break
    return x_list, y_list

def line_67(x_start, y_start, width):
    y_list = []
    x_list = []

    for i in range(0,100,2):
        if x_start-i-1 > 0:
            y_i = list(range(y_start-i-1, y_start-i+1))
            x_i = [x_start-i//2] * len(y_i)
                       
            x_list.extend(x_i)
            y_list.extend(y_i)    
        else:
            break
    return x_list, y_list

def line_90(x_start, y_start, width):
    y_list = []
    x_list = []
    
    for i in range(0,100,1):
        if y_start-i > 0:            
            x_list.append(x_start)
            y_list.append(y_start-i)    
        else:
            break
    return x_list, y_list

def line_112(x_start, y_start, width):
    y_list = []
    x_list = []
    
    for i in range(0,100,2):
        if y_start-i-1 >= 1:
            y_i = list(range(y_start-i-1, y_start-i+1))
            x_i = [x_start+i//2] * len(y_i)
            
            x_list.extend(x_i)
            y_list.extend(y_i)          
        else:
            break
    return x_list, y_list
            
def line_135(x_start, y_start, width):
    y_list = []
    x_list = []
    
    x_start += 1
    
    for i in range(0, 100, 1):
        if y_start-i >= 2:
            x_i = list(range(x_start+i-1, x_start+i))
            y_i = [y_start-i] * len(x_i)
            
            x_list.extend(x_i)
            y_list.extend(y_i)          
        else:
            break
    return x_list, y_list

def line_157(x_start, y_start, width):
    y_list = []
    x_list = []
    y_step = 0

    for i in range(0, 100, 2):
        if x_start+i+2 < width:
            x_i = list(range(x_start+i, x_start+i+2))
            y_i = [y_start-y_step] * 2
            y_step += 1

            x_list.extend(x_i)
            y_list.extend(y_i)
        else:
            break
    return x_list, y_list

def line_180(x_start, y_start, width):
    x_list = list(range(x_start, x_start+width//2))
    y_list = [y_start] * len(x_list)
    return x_list, y_list
    
angles_draw = {
            180: line_180,
            22: line_22,
            45: line_45,
            67: line_67,
            90: line_90,
            112: line_112,
            135: line_135,
            157: line_157
        }

def draw_angle_stimulus(angle, strength=1, width=20, height=20, half=True):
    img = np.zeros(shape=(height, width))

    middle_x = width // 2
    middle_y = height // 2

    start_y, start_x = middle_y, 1
    end_y, end_x = middle_y, middle_x

    # left horizontal line
    y_left, x_left = draw.line_nd((start_y, start_x), (end_y, end_x))
    right_y_modifier = 0
    if angle < 180:
        right_y_modifier = 1

    if half == True:
        # right line
        x_right, y_right = angles_draw[angle](x_start=end_x, y_start=end_y-right_y_modifier, width=width)
        min_len = min(len(x_right), len(x_left))
        if len(x_right) > min_len:
            x_right, y_right = x_right[:min_len], y_right[:min_len]
        elif len(x_left) > min_len:
            x_left, y_left = x_left[-min_len:], y_left[-min_len:]

        img[y_left, x_left] = strength
        img[y_right, x_right] = strength

        return {'img': img,
                'coords': {
                    'y_left': y_left,
                    'x_left': x_left,
                    'y_right': y_right,
                    'x_right': x_right
                }}
    else:
        # top line
        x_top, y_top = angles_draw[angle](x_start=end_x, y_start=end_y, width=width)

        # right horizontal line
        y_right = y_left
        x_right = [el + len(x_left) for el in x_left]

        # bottom line
        y_bottom = [el + (max(y_top) - min(y_top)) for el in y_top]
        x_bottom = [el - (max(x_top) - min(x_top)) for el in x_top]

        if angle == 112:
            x_bottom = [el - 1 for el in x_bottom]
            y_top = [el - 1 for el in y_top]
            pass
        elif angle == 157:
            #y_bottom = [el + 2 for el in y_bottom]
            x_bottom = [el + 1 for el in x_bottom]
            #x_top = [el + 1 for el in x_top]
            pass

        min_len = min(len(x_right), len(x_left), len(x_top), len(x_bottom))
        if len(x_right) > min_len:
            x_right, y_right = x_right[:min_len], y_right[:min_len]
        if len(x_top) > min_len:
            x_top, y_top = x_top[:min_len], y_top[:min_len]
        if len(x_left) > min_len:
            x_left, y_left = x_left[-min_len:], y_left[-min_len:]
        if len(x_bottom) > min_len:
            x_bottom, y_bottom = x_bottom[-min_len:], y_bottom[-min_len:]

        # plot all lines
        img[y_left, x_left] = strength
        img[y_top, x_top] = strength
        img[y_right, x_right] = strength
        img[y_bottom, x_bottom] = strength

        return {'img': img,
                'coords': {
                    'y_left': y_left,
                    'x_left': x_left,
                    'y_right': y_right,
                    'x_right': x_right,
                    'y_top': y_top,
                    'x_top': x_top,
                    'y_bottom': y_bottom,
                    'x_bottom': x_bottom
                },
                }

def draw_stimulus_proximity(width=20, height=20, distance=0):
    img = np.zeros(shape=(height, width))

    if (width - distance) % 2 != 0:
        segment_len = (width - distance) // 2 - 1
        x_left_start = 1
    else:
        segment_len = (width - distance) // 2
        x_left_start = 0

    x_left_end = x_left_start + segment_len
    x_right_start = x_left_end + int(distance) + 1
    x_right_end = x_right_start + segment_len
    y = height // 2

    x_left = list(range(x_left_start, x_left_end + 1))
    x_right = list(range(x_right_start, x_right_end + 1))
    y_left = [y] * len(x_left)
    y_right = [y] * len(x_right)
    y_left = list(y_left)
    y_right = list(y_right)

    img[y, x_left_start:x_left_end] = 1
    img[y, x_right_start:x_right_end] = 1

    return {'img': img,
            'coords': {
                'y_left': y_left,
                'x_left': x_left,
                'y_right': y_right,
                'x_right': x_right
            },
            }

if __name__ == '__main__':

    par = 157
    img = draw_angle_stimulus(angle=par, strength=1, width=20, height=20, half=True)
    #img = draw_stimulus_proximity(width=22, height=22, distance=par)
    plt.imshow(img['img'])
    plt.savefig(f'sim{par}.png')

