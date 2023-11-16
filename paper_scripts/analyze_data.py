import argparse
import multiprocessing as mp
import numpy as np
import os
import sys
import yaml

paths = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
	os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')]
sys.path.extend(paths)

from src.draw import draw_stimulus_proximity, draw_angle_stimulus
from src.measure import Measure
from src.simulation import SimulationExperiment
from src.utils import exp_params, create_logger, find_transient, angle_vals, downsample_data

def jitter_firings(neuron_firing, jitter_value=60):
    spike_times = np.nonzero(neuron_firing)[0]
    spike_times_jittered = spike_times + np.random.randint(low=-jitter_value, high=jitter_value, size=spike_times.shape)
    spike_times_jittered[spike_times_jittered < 0] = 0
    spike_times_jittered[spike_times_jittered >= len(neuron_firing)] = len(neuron_firing) - 1
    # print('counts', np.unique(spike_times_jittered, return_counts=True))
    neuron_firing_jittered = np.zeros(neuron_firing.shape)
    neuron_firing_jittered[spike_times_jittered] = 1
    return neuron_firing_jittered

def get_coords_proximity(stimulus, thalamic_input):
    height, width = stimulus['img'].shape
    # all coordinates of the left stimulus part
    all_xy_left = [y * width + x for y, x in zip(stimulus['coords']['y_left'], stimulus['coords']['x_left'])
                   if thalamic_input[y * width + x] == 1]
    # all coordinates of the right stimulus part
    all_xy_right = [y * width + x for y, x in zip(stimulus['coords']['y_right'], stimulus['coords']['x_right'])
                    if thalamic_input[y * width + x] == 1]
    res = {}
    for n in range(1, 7):
        random_left = list(np.random.choice(all_xy_left, n, replace=False))
        random_right = list(np.random.choice(all_xy_right, n, replace=False))
        res[n] = random_left + random_right
        '''
        coord_step = min(len(all_xy_left), len(all_xy_right)) // n
        step_left = all_xy_left[::coord_step]
        step_right = all_xy_right[::-coord_step]
        min_len = min(len(step_left), len(step_right))
        res[n] = step_left[:min_len] + step_right[:min_len]
        '''
    return res

def get_coords_similarity(stimulus, angle, thalamic_input):
    height, width = stimulus['img'].shape
    # find all coords on the left half
    all_xy_left = [y * width + x for y, x in zip(stimulus['coords']['y_left'], stimulus['coords']['x_left'])
                   if thalamic_input[y * width + x] == 1]

    # find all coords on the right half
    i = angle_vals.index(angle)
    all_y, all_x = stimulus['coords']['y_right'], stimulus['coords']['x_right']

    if all_x[0] > all_x[1]:
        all_x = all_x[::-1]
        all_y = all_y[::-1]
    all_xy_right_tuples = [(x, y, (y * width + x) + width * height * i) for x, y in zip(all_x, all_y)
                           if thalamic_input[y * width + x + width * height * i] == 1]
    all_x, all_y, all_xy_right = [list(t) for t in zip(*all_xy_right_tuples)]

    res = {}
    for n in range(1, 7):
        random_left = list(np.random.choice(all_xy_left, n, replace=False))
        random_right = list(np.random.choice(all_xy_right, n, replace=False))
        res[n] = random_left + random_right
        '''
        coord_step = min(len(all_xy_left), len(all_xy_right)) // n
        step_left = all_xy_left[::coord_step]
        step_right = all_xy_right[::-coord_step]
        min_len = min(len(step_left), len(step_right))
        res[n] = step_left[:min_len] + step_right[:min_len]
        '''
    return res

def get_coords_continuity(stimulus, thalamic_input, angle):
    height, width = stimulus['img'].shape
    i = angle_vals.index(angle)
    all_xy_left = [y * width + x for y, x in zip(stimulus['coords']['y_left'], stimulus['coords']['x_left'])
                   if thalamic_input[y * width + x] > 0]
    all_xy_right = [y * width + x for y, x in zip(stimulus['coords']['y_right'], stimulus['coords']['x_right'])
                    if thalamic_input[y * width + x] > 0]
    if stimulus['coords']['x_top'][0] > stimulus['coords']['x_top'][1]:
        stimulus['coords']['x_top'] = stimulus['coords']['x_top'][::-1]
        stimulus['coords']['y_top'] = stimulus['coords']['y_top'][::-1]

    if stimulus['coords']['x_bottom'][0] > stimulus['coords']['x_bottom'][1]:
        stimulus['coords']['x_bottom'] = stimulus['coords']['x_bottom'][::-1]
        stimulus['coords']['y_bottom'] = stimulus['coords']['y_bottom'][::-1]

    all_xy_top = [(y * width + x) + width * height * i for y, x in
                  zip(stimulus['coords']['y_top'], stimulus['coords']['x_top'])
                  if thalamic_input[y * width + x + width * height * i] > 0]
    all_xy_bottom = [(y * width + x) + width * height * i for y, x in
                     zip(stimulus['coords']['y_bottom'], stimulus['coords']['x_bottom'])
                     if thalamic_input[y * width + x + width * height * i] > 0]
    res = {}
    for n in range(1, 7):
        res[n] = {}
        res[n]["left"] = list(np.random.choice(all_xy_left, n, replace=False))
        res[n]["right"] = list(np.random.choice(all_xy_right, n, replace=False))
        res[n]["top"] = list(np.random.choice(all_xy_top, n, replace=False))
        res[n]["bottom"] = list(np.random.choice(all_xy_bottom, n, replace=False))
        '''
        res[n] = {}
        coord_step = min(len(all_xy_left), len(all_xy_right), len(all_xy_bottom), len(all_xy_top)) // n
        step_left = all_xy_left[::coord_step]
        step_right = all_xy_right[::-coord_step]
        step_top = all_xy_top[::coord_step]
        step_bottom = all_xy_bottom[::-coord_step]
        min_len = min(len(step_left), len(step_right), len(step_top), len(step_bottom))
        res[n]["left"] = step_left[:min_len]
        res[n]["right"] = step_right[:min_len]
        res[n]["top"] = step_top[:min_len]
        res[n]["bottom"] = step_bottom[:min_len]
        '''
    return res

def measure_rsync_proximity_similarity(firings, rsync_kwargs):
    measure = Measure(firings, 'rsync')
    rsync = round(measure.similarity(**rsync_kwargs), 4)
    return rsync

def update_rsync_continuity(firings, coords, n_coords, rsync_kwargs):
    result = {}
    # between left and top coords
    measure = Measure(firings[coords["left"] + coords["top"]], 'rsync')
    result[f"rsync_left_top_{n_coords}"] = round(measure.similarity(**rsync_kwargs), 4)

    # between left and right coords
    measure = Measure(firings[coords['left'] + coords['right']], 'rsync')
    result[f"rsync_left_right_{n_coords}"] = round(measure.similarity(**rsync_kwargs), 4)
    return result

def analyze_exp_param(exp, param, log_dir, config_file_path, overwrite):
    with open(config_file_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    width, height = config["stimulus"]["width"], config["stimulus"]["height"]
    rsync_kwargs = {"downsample": config["simulation"]["downsample"], "delta_t": config["model"]["delta_t"]}
    npy_dir_path = os.path.join(log_dir, str(param), "firings")

    if os.path.isdir(npy_dir_path):
        log_file_path = os.path.join(log_dir, f"{exp_params[exp][0]}_{param}.log")
        if os.path.isfile(log_file_path) and overwrite is True:
            os.remove(log_file_path)
        logger = create_logger(log_file_path)
        print(f"Logger {log_file_path} created")

        if exp == "similarity":
            stimulus = draw_angle_stimulus(angle=param, strength=1, width=width, height=height, half=True)
            simulation = SimulationExperiment(stimulus=stimulus, config_file_path=config_file_path)
            coords_all = get_coords_similarity(stimulus=stimulus, angle=param, thalamic_input=simulation.thalamic_input)
        elif exp == "continuity":
            stimulus = draw_angle_stimulus(angle=param, strength=1, width=width, height=height, half=False)
            simulation = SimulationExperiment(stimulus=stimulus, config_file_path=config_file_path)
            coords_all = get_coords_continuity(stimulus=stimulus, angle=param, thalamic_input=simulation.thalamic_input)
        elif exp == "proximity":
            stimulus = draw_stimulus_proximity(distance=param, width=width, height=height)
            simulation = SimulationExperiment(stimulus=stimulus, config_file_path=config_file_path)
            coords_all = get_coords_proximity(stimulus=stimulus, thalamic_input=simulation.thalamic_input)

        for f in os.listdir(npy_dir_path):
            print(f)
            file_path = os.path.join(npy_dir_path, f)
            firings = np.load(file_path)
            firings = downsample_data(firings, config["simulation"]["downsample"])
            transient_start, transient_end = find_transient(firings)
            firings = firings[:, transient_end:]

            # measure rsync for continuity data
            if exp == "continuity":
                result = {}
                for n_coords in coords_all:
                    coords = coords_all[n_coords]

                    # measure rsync for regular firings between selected coords
                    result_no_jitter = update_rsync_continuity(firings, coords, n_coords, rsync_kwargs)
                    for k, v in result_no_jitter.items():
                        result[k] = v

                    # measure rsync for jittered firings
                    spikes_jittered = np.apply_along_axis(func1d=jitter_firings, axis=1, arr=firings)
                    result_jitter = update_rsync_continuity(spikes_jittered, coords, n_coords, rsync_kwargs)
                    for k, v in result_jitter.items():
                        k += '_jitter'
                        result[k] = v

            # measure rsync for proximity and similarity data
            else:
                result = {}
                for n_coords in coords_all:
                    coords = coords_all[n_coords]

                    # measure rsync for regular firings between selected coords
                    result[f"rsync_{n_coords}"] = measure_rsync_proximity_similarity(firings[coords], rsync_kwargs)

                    # measure rsync for jittered firings between selected coords
                    spikes_jittered = np.apply_along_axis(func1d=jitter_firings, axis=1, arr=firings[coords])
                    result[f'rsync_{n_coords}_jitter'] = measure_rsync_proximity_similarity(spikes_jittered, rsync_kwargs)

            logger_row = ''
            for k, v in result.items():
                logger_row += f'{k}:{v} '
            logger.info(logger_row[:-1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--overwrite', type=int, default=0,
                        help="overwrite log files with rsync values. 0: no, 1: yes")
    parser.add_argument('-e', '--experiment', type=int, default=0,
                        help="analyze data for the experiment. 0: proximity, 1: similarity, 2: continuity")
    parser.add_argument('-m', '--multiproc', type=int, default=1,
                        help="use multiprocesing for running the analysis. 0: no, 1: yes")
    args = parser.parse_args()
    overwrite = bool(args.overwrite)
    multiproc = bool(args.multiproc)
    exp = list(exp_params.keys())[int(args.experiment)]

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    config_file_path = os.path.join(data_dir, "config.yaml")
    log_dir = os.path.join(data_dir, "logs", exp)

    # if the log dir for specific experiment exists
    if os.path.isdir(log_dir):
        if multiproc is False:
            for param in exp_params[exp][1]:
                analyze_exp_param(exp=exp, param=param, log_dir=log_dir, config_file_path=config_file_path, overwrite=overwrite)
        else:
            procs = []
            for param in exp_params[exp][1]:
                print(f'Start {exp} analysis for parameter {param}')
                proc = mp.Process(target=analyze_exp_param, args=(exp, param, log_dir, config_file_path, overwrite),
                                  name=f'{exp}_{param}')
                procs.append(proc)
                proc.start()

            # complete the processes
            for proc in procs:
                proc.join()

            for proc in procs:
                proc.terminate()














