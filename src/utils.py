import logging
import os 
import sys
import random
import matplotlib.colors as mcolors
import numpy as np
import string
import time
import yaml

from collections import OrderedDict
from functools import singledispatch, update_wrapper

angle_vals = [180, 157, 135, 112, 90]
distance_vals = [0, 1, 2, 3, 4]
exp_params = OrderedDict({
                  "proximity": ("distance", distance_vals),
                  "similarity": ("angle", angle_vals),
                  "continuity": ("angle", angle_vals[1:])
                })

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()

def create_logger(logger_name):
    # Gets or creates a logger
    logger = logging.getLogger(__name__)  

    # set log level
    logger.setLevel(logging.INFO)

    # define file handler and set formatter
    file_handler = logging.FileHandler(logger_name)
    formatter    = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # add file handler to logger
    logger.addHandler(file_handler)
    return logger

def d1_to_d2(coord, width):
    x = coord % width
    y = coord // width
    return (int(x), int(y))
    
def d2_to_d1(x, y, width):
    return x + width * y

def save_npy(logs_dir_path, data, plot_type, param, trial):
    npy_dir_path = os.path.join(logs_dir_path, str(param), plot_type)
    os.makedirs(npy_dir_path, exist_ok=True)

    files = [int(f.split('.')[0]) for f in os.listdir(npy_dir_path)]
    if len(files) == 0:
        trial = 1
    else:
        for i in range(min(files), max(files)+2):
            file_path = os.path.join(npy_dir_path, f"{i}.npy")
            if not os.path.isfile(file_path):
                trial = i
                break

    file_path = os.path.join(npy_dir_path, f"{trial}.npy")
    np.save(file_path, data) 
    print(f"FILE {file_path} saved")
    
def generate_pattern(width, height, pattern_distance_prob_drop, pattern_distance_prob_cutoff):    
    pattern = np.zeros((height, width))
    margin = 2
    center = np.random.randint(margin, height-margin), np.random.randint(margin, width-margin)
            
    for i in range(height):
        for j in range(width):
                
            p_on = max(1.0 / (pattern_distance_prob_drop * np.sqrt((center[0]-i)**2 + (center[1]-j)**2)) - pattern_distance_prob_cutoff, 0) if (i,j) != center else 1
            if np.random.random() < p_on:
                pattern[i,j] = 1
    return pattern

def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

def find_transient(spikes):
    """
    Finds a time step from which it makes sense to interpret spiking results.
    Before this time step there is a transient period:
        1) neurons don't spike at all, then all spike at the same time,
        2) then spiking becomes meaningful, activity stabilizes. The end of transient period

    Args:
        spikes (numpy array): spike trains
    """
    zero_after_transient = 0
    transient_start = -1
    transient_end = 0

    for i, s in enumerate(spikes.T):
        if s.sum() > 20:
            transient_start = i

        if transient_start > -1 and s.sum() == 0:
            zero_after_transient += 1

        elif transient_start > -1 and s.sum() > 0:
            zero_after_transient = 0

        if zero_after_transient > 10:
            transient_end = i
            break
    return transient_start, transient_end

def downsample_data(firings, downsample):
    time_steps = firings.shape[1]
    spike_indices, spike_times = np.nonzero(firings)
    spikes = np.zeros((firings.shape[0], int(time_steps / downsample)), dtype='bool')
    spikes[spike_indices, (spike_times / downsample).astype(int)] = True
    return spikes

def add_noise(stimulus, noise_type, noise_level):
        """    
        image : ndarray Input image data. Will be converted to float.
        noise_level : float noise level.
        noise_type : str
            One of the following strings, selecting the type of noise to add:

            'gauss'     Gaussian-distributed additive noise.
            'poisson'   Poisson-distributed noise generated from the data.
            'SnP'       Replaces random pixels with 0 or 1.
            'speckle'   Multiplicative noise using out = image + n*image,where
                        n is uniform noise with specified mean & variance.
        """

        if noise_type == "gauss":
            active_input_neurons = stimulus == 1
            noise = np.random.uniform(0, 1, stimulus.shape)
            stimulus = np.clip((stimulus + noise*noise_level), 0, 1.0)
            stimulus[active_input_neurons] = 1
            return stimulus

        if noise_type == "gauss2":
            noise = np.random.uniform(0, 1, stimulus.shape)
            noise = np.clip(noise, 0, noise_level)

        elif noise_type == "SnP":
            row,col,ch = stimulus.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(stimulus)
            # Salt mode
            num_salt = np.ceil(amount * stimulus.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                    for i in stimulus.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount* stimulus.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                    for i in stimulus.shape]
            out[coords] = 0
            return out

        elif noise_type == "poisson":
            vals = len(np.unique(stimulus))
            vals = 2 ** np.ceil(np.log2(vals))
            stimulus_noisy = np.random.poisson(stimulus * vals) / float(vals)

        elif noise_type =="speckle":
            row,col,ch = stimulus.shape
            gauss = np.random.randn(row,col,ch)
            gauss = gauss.reshape(row,col,ch)        
            stimulus_noisy = stimulus + stimulus * gauss
        return stimulus_noisy 

def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.6+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count}", end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

def latticeWrapIdx(index, lattice_shape):
    """returns periodic lattice index 
    for a given iterable index
    
    Required Inputs:
        index :: iterable :: one integer for each axis
        lattice_shape :: the shape of the lattice to index to

    source: 
        https://stackoverflow.com/questions/38066785/np-ndarray-with-periodic-boundary-conditions
    """
    if not hasattr(index, '__iter__'): return index         # handle integer slices
    if len(index) != len(lattice_shape): return index       # must reference a scalar
    if any(type(i) == slice for i in index): return index   # slices not supported
    if len(index) == len(lattice_shape):                    # periodic indexing of scalars
        mod_index = tuple(( (i%s + s)%s for i,s in zip(index, lattice_shape)))
        return mod_index
    raise ValueError('Unexpected index: {}'.format(index))

def mkdir_experiment_setup(config_file_path):

    # import experiment configurations
    with open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), config_file_path)) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    if config["experiment"]["save_experiment"]: 

        base_dir_path = os.path.join(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))), 'experiments/')

        if not os.path.isdir(base_dir_path): os.mkdir(base_dir_path)

        experiment_ID = time.strftime("%Y%m%d_%H%M%S_")
        experiment_ID += ''.join(random.choices(string.ascii_lowercase + string.digits, k=5))
        experiment_path = base_dir_path + str(experiment_ID)
        os.mkdir(experiment_path)

        config["experiment"]["experiment_ID"] = experiment_ID
        config["experiment"]["experiment_path"] = experiment_path

        with open(os.path.join(experiment_path, 'config.yaml'), 'w') as file:
            yaml.dump(config, file)

        print(f"Set up experiment '{experiment_ID}' directory")

    return config, experiment_ID

def singledispatch_instance_method(func):
    """Small wrapper function to allow for singledispatch of instance methods"""

    dispatcher = singledispatch(func)
    def wrapper(*args, **kw):
        return dispatcher.dispatch(args[1].__class__)(*args, **kw)
    wrapper.register = dispatcher.register
    update_wrapper(wrapper, dispatcher)
    return wrapper

class Observable:

    def __init__(self):
        self._observers = []

    def subscribe(self, observer):
        self._observers.append(observer)

    def notify(self, *args, **kwargs):
        for obs in self._observers:
            obs.update(self, *args, **kwargs)

    def unsubscribe(self, observer):
        self._observers.remove(observer)

class Observer:

    def __init__(self, observables):
        for observable in observables: observable.subscribe(self)

    def update(self, observable, *args, **kwargs):
        raise NotImplementedError
