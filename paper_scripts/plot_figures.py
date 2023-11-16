import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import scipy as sp
import seaborn as sns
import sys

paths = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
         os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'src'),
         os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'paper_scripts')]
sys.path.extend(paths)

from src.draw import draw_angle_stimulus, draw_stimulus_proximity
from src.simulation import SimulationExperiment
from src.measure import Measure
from src.utils import angle_vals, get_continuous_cmap

from paper_scripts.experiment_similarity import get_coords_similarity
from paper_scripts.experiment_proximity import get_coords_proximity
from paper_scripts.experiment_continuity import get_coords_continuity

sns.set(font_scale=1.5)

data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'data')
logs_dir = os.path.join(data_dir, 'logs')
plots_dir = os.path.join(data_dir, 'plots')
stats_dir = os.path.join(data_dir, 'stats')

hex_codes_pairwise = ['#000000', '#75d4bc', '#fa7c4f', '#FFF2E8'] #black bg
#hex_codes_pairwise = ['#ffffff', '#6ccdc0', '#f9a567', '#bf3800'] #white bg
#hex_codes_pairwise = ['#f5f5f5', '#149084', '#f08745', '#8b1500'] #gray bg
hex_codes_matrix = [ '#000000', '#6ccdc0', '#eb7341', '#ffffff']

cmap_pairwise = get_continuous_cmap(hex_codes_pairwise)
cmap_matrix = get_continuous_cmap(hex_codes_matrix)

experiment_params = {
                     'proximity': (0, 1, 2, 3, 4),
                     'similarity': (90, 112, 135, 157, 180),
                     'continuity': (90, 112, 135, 157)
                     }

experiment_param_names = {
    'proximity': 'distance',
    'similarity': 'angle',
    'continuity': 'angle'
}
def init_heatmap_dicts(width, height, param_list):
    heatmaps_rsync = {}
    heatmaps_pair_nums = {}
    heatmaps_rate = {}

    for param_val in param_list:
        heatmaps_rsync[param_val] = {'in': np.zeros((width, height)), 'out': np.zeros((width, height)),
                                     'all': np.zeros((width, height))}
        heatmaps_pair_nums[param_val] = {'in': np.zeros((width, height)), 'out': np.zeros((width, height)),
                                         'all': np.zeros((width, height))}
        heatmaps_rate[param_val] = np.zeros((width, height))
    return heatmaps_rsync, heatmaps_pair_nums, heatmaps_rate

def get_coords_all_angles(width, height, stimulus, angle_vals):
    coords_all_angles = {}
    for i in range(len(angle_vals)):

        for segment in ('left', 'right', 'top', 'bottom'):
            if f'x_{segment}' in stimulus['coords'] and f'y_{segment}' in stimulus['coords']:

                for y, x in zip(stimulus['coords'][f'y_{segment}'], stimulus['coords'][f'x_{segment}']):
                    coords_all_angles[y * width + x + (width * height) * i] = (y, x)
    return coords_all_angles

if __name__ == '__main__':
    max_rsync = 0
    min_rsync = 1.0

    config_file_path = os.path.join(data_dir, 'config_angles.yaml')
    spatial_jitter = 0

    results = {}
    for experiment in experiment_params:

        if experiment == 'similarity':
            width, height = 20, 20
        elif experiment == 'proximity':
            width, height = 22, 22
        elif experiment == 'continuity':
            width, height = 24, 24

        experiment_results = {'connect_matrix': {}, 'rsync_matrix': {},
                        'pearson': {}, 'pearson_p': {}, 'pearson_std': {},
                        'spearman': {}, 'spearman_p': {}, 'spearman_std': {}}
        heatmaps_rsync, heatmaps_pair_nums, heatmaps_rate = init_heatmap_dicts(width, height,
                                                                               experiment_params[experiment])
        for param in experiment_params[experiment]:

            print(experiment, param)
            if experiment == 'similarity':
                angle = param
                stimulus = draw_angle_stimulus(angle=param, strength=1, width=width, height=height, half=True)

            elif experiment == 'proximity':
                angle = 180
                stimulus = draw_stimulus_proximity(width=width, height=height, distance=param)

            elif experiment == 'continuity':
                stimulus = draw_angle_stimulus(angle=param, strength=1, width=width, height=height, half=False)

            # initialize the simulation
            simulation = SimulationExperiment(stimulus_file_path=stimulus['img'],
                                              config_file_path=config_file_path)
            connectivity = simulation.model.connectivity_matrix

            coords_all_angles = get_coords_all_angles(width=width, height=height, stimulus=stimulus,
                                                      angle_vals=angle_vals)
            n_coords = max([len(stimulus['coords'][segment]) for segment in stimulus['coords']])

            if experiment == 'similarity':
                selected_coords, stimulus_coords_dict = get_coords_similarity(stimulus=stimulus,
                                                                              thalamic_input=simulation.thalamic_input,
                                                                              n_coords=n_coords, jitter=spatial_jitter,
                                                                              angle=angle, angle_vals=angle_vals)
            elif experiment == 'proximity':
                selected_coords, stimulus_coords_dict = get_coords_proximity(stimulus=stimulus,
                                                                             n_coords=n_coords, max_distance=param,
                                                                             jitter=spatial_jitter)
                stimulus_coords_dict['left'] = stimulus_coords_dict['left'][:-1]
                stimulus_coords_dict['right'] = stimulus_coords_dict['right'][:-2]

            elif experiment == 'continuity':
                selected_coords, stimulus_coords_dict = get_coords_continuity(stimulus=stimulus,
                                                                              thalamic_input=simulation.thalamic_input,
                                                                              angle=param, angle_vals=angle_vals,
                                                                              n_coords=n_coords,
                                                                              jitter=spatial_jitter)
                stimulus_coords_dict['left'] = stimulus_coords_dict['left'] + stimulus_coords_dict['right']
                stimulus_coords_dict['right'] = stimulus_coords_dict['top'] + stimulus_coords_dict['bottom']

            stimulus_coords = sorted(list(set(sum(list(stimulus_coords_dict.values()), []))))

            num_trials_max = 100

            # fill a connectivity matrix
            connect_matrix = connectivity[stimulus_coords][:, stimulus_coords]
            connect_matrix_1 = connect_matrix.copy()
            np.fill_diagonal(connect_matrix_1, connect_matrix_1.max())
            rsync_matrix = np.zeros(list(connect_matrix.shape) + [num_trials_max])

            real_trials = []
            for trial in range(num_trials_max):
                param_name = experiment_param_names[experiment]
                firings_path = f'C://Users/vzemliak/work/git/synchrony/data/logs/{experiment}/{param_name}_{param}/firings/{trial}.npy'

                if os.path.isfile(firings_path):
                    real_trials.append(trial)
                    firings = np.load(firings_path).astype(int)
                    firings_selected = firings[stimulus_coords]

                    rsync_kwargs = {'downsample': simulation.downsample, 'delta_t': simulation.delta_t}

                    for i in range(firings_selected.shape[0]):
                        for j in range(firings_selected.shape[0]):

                            rsync = 0
                            if firings_selected[[i, j]].sum() > 0:
                                measure = Measure(firings_selected[[i, j]], 'rsync')
                                rsync = measure.similarity(**rsync_kwargs)

                            rsync_matrix[i, j, trial] = rsync

                    # fill heatmaps
                    for i in stimulus_coords_dict['left']:

                        y, x = coords_all_angles[i]
                        heatmaps_rate[param][y, x] += firings[i].sum()

                        for j in stimulus_coords_dict['left']:

                            if not np.any(firings[[i, j]].sum(1) == 0):
                                measure = Measure(firings[[i, j]], 'rsync')
                                rsync = measure.similarity(**rsync_kwargs)
                                heatmaps_rsync[param]['in'][y, x] += rsync
                                heatmaps_pair_nums[param]['in'][y, x] += 1

                                heatmaps_rsync[param]['all'][y, x] += rsync
                                heatmaps_pair_nums[param]['all'][y, x] += 1

                        for j in stimulus_coords_dict['right']:

                            if not np.any(firings[[i, j]].sum(1) == 0):
                                measure = Measure(firings[[i, j]], 'rsync')
                                rsync = measure.similarity(**rsync_kwargs)

                                heatmaps_rsync[param]['out'][y, x] += rsync
                                heatmaps_pair_nums[param]['out'][y, x] += 1

                                heatmaps_rsync[param]['all'][y, x] += rsync
                                heatmaps_pair_nums[param]['all'][y, x] += 1

                    for i in stimulus_coords_dict['right']:

                        y, x = coords_all_angles[i]
                        heatmaps_rate[param][y, x] += firings[i].sum()

                        for j in stimulus_coords_dict['right']:

                            if not np.any(firings[[i, j]].sum(1) == 0):
                                measure = Measure(firings[[i, j]], 'rsync')
                                rsync = measure.similarity(**rsync_kwargs)

                                heatmaps_rsync[param]['in'][y, x] += rsync
                                heatmaps_pair_nums[param]['in'][y, x] += 1

                                heatmaps_rsync[param]['all'][y, x] += rsync
                                heatmaps_pair_nums[param]['all'][y, x] += 1

                        for j in stimulus_coords_dict['left']:

                            if not np.any(firings[[i, j]].sum(1) == 0):
                                measure = Measure(firings[[i, j]], 'rsync')
                                rsync = measure.similarity(**rsync_kwargs)

                                heatmaps_rsync[param]['out'][y, x] += rsync
                                heatmaps_pair_nums[param]['out'][y, x] += 1

                                heatmaps_rsync[param]['all'][y, x] += rsync
                                heatmaps_pair_nums[param]['all'][y, x] += 1

            pearson = []
            pearson_p = []
            spearman = []
            spearman_p = []

            for trial in real_trials:
                rsync_matrix_1 = rsync_matrix[:, :, trial].copy()
                np.fill_diagonal(rsync_matrix_1, rsync_matrix_1.max())

                pearson_trial = [round(el, 4) for el in sp.stats.pearsonr(connect_matrix_1.flatten(),
                                                                          rsync_matrix_1.flatten())]
                spearman_trial = [round(el, 4) for el in sp.stats.spearmanr(connect_matrix_1.flatten(),
                                                                            rsync_matrix_1.flatten())]
                pearson.append(pearson_trial[0])
                pearson_p.append(pearson_trial[1])
                spearman.append(spearman_trial[0])
                spearman_p.append(spearman_trial[1])

            rsync_matrix = rsync_matrix[:, :, real_trials]
            rsync_matrix = rsync_matrix.mean(2)
            experiment_results['pearson'][param] = np.mean(pearson)
            experiment_results['pearson_p'][param] = np.mean(pearson_p)
            experiment_results['spearman'][param] = np.mean(spearman)
            experiment_results['spearman_p'][param] = np.mean(spearman_p)
            experiment_results['pearson_std'][param] = np.std(pearson)
            experiment_results['spearman_std'][param] = np.std(spearman)

            # rsync_matrix /= len(real_trials)
            heatmaps_rate[param] /= len(real_trials)

            experiment_results['connect_matrix'][param] = connect_matrix
            experiment_results['rsync_matrix'][param] = rsync_matrix
            print('RSYNC MATRIX', rsync_matrix.shape, rsync_matrix.min(), rsync_matrix.mean(),
                  rsync_matrix.max())

            for segment in heatmaps_rsync[param]:
                heatmaps_rsync[param][segment] /= (heatmaps_pair_nums[param][segment] + 1)
                nonzero_mask = heatmaps_rsync[param][segment] > 0

                min_rsync = min(heatmaps_rsync[param][segment][nonzero_mask].min(), min_rsync)
                max_rsync = max(heatmaps_rsync[param][segment].max(), max_rsync)

            print('---')

        experiment_results['heatmaps_rsync'] = heatmaps_rsync
        experiment_results['heatmaps_rate'] = heatmaps_rate

        results[experiment] = experiment_results
        print(experiment, 'DONE\n---')

    print('RSYNC MIN', min_rsync, 'RSYNC MAX', max_rsync)

    for experiment in results:
        print('RESULT', experiment)
        col_pearson = []
        col_pearson_p = []
        col_spearman = []
        col_spearman_p = []
        col_param = []

        matrix_type = 'heatmaps_rsync'
        figures_dir = os.path.join(plots_dir, experiment)
        figures_heatmaps_dir = os.path.join(figures_dir, matrix_type)
        figures_connect_dir = os.path.join(figures_dir, 'connectivity')
        os.makedirs(figures_heatmaps_dir, exist_ok=True)
        os.makedirs(figures_connect_dir, exist_ok=True)

        for param in results[experiment][matrix_type]:
            fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 20))
            im_in = ax[0].imshow(results[experiment][matrix_type][param]['in'], vmin=min_rsync, vmax=max_rsync,
                                 cmap=cmap_pairwise)
            ax[0].title.set_text('Pairwise Rsync INSIDE')
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[0].axis('off')

            im_out = ax[1].imshow(results[experiment][matrix_type][param]['out'], vmin=min_rsync, vmax=max_rsync,
                                  cmap=cmap_pairwise)
            ax[1].title.set_text('Pairwise Rsync OUTSIDE')
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            ax[1].axis('off')

            im_all = ax[2].imshow(results[experiment][matrix_type][param]['all'], vmin=min_rsync, vmax=max_rsync,
                                  cmap=cmap_pairwise)
            ax[2].title.set_text('Pairwise Rsync ALL')
            ax[2].set_xticks([])
            ax[2].set_yticks([])
            ax[2].axis('off')
            cax = fig.add_axes(
                [ax[2].get_position().x1 + 0.05, ax[2].get_position().y0, 0.02, ax[2].get_position().height])
            fig.colorbar(im_all, ax=ax[2], cax=cax)

            plot_path = os.path.join(figures_heatmaps_dir, f'{param}.svg')
            fig.savefig(plot_path, bbox_inches='tight')

            # plot connectivity and pairwise rsync
            connect_matrix = results[experiment]['connect_matrix'][param]
            rsync_matrix = results[experiment]['rsync_matrix'][param]

            if experiment in ('continuity', 'similarity'):
                col_param.append(180-param)
            else:
                col_param.append(param)

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 20))

            im_connect = ax[0].imshow(connect_matrix, vmax=connect_matrix.max(),
                                      cmap=cmap_matrix)
            ax[0].invert_yaxis()
            ax[0].set_xticks([])
            ax[0].set_yticks([])
            ax[0].axis('off')
            ax[0].title.set_text('Connectivity')
            cax = fig.add_axes(
                [ax[0].get_position().x1 + 0.01, ax[0].get_position().y0, 0.02, ax[0].get_position().height])
            fig.colorbar(im_connect, ax=ax[0], cax=cax)

            print('PLOTTING RSYNC with connectivity')
            print('min_rsync', min_rsync, 'max_rsync', max_rsync)
            print('matrix', rsync_matrix.min(), rsync_matrix.mean(), rsync_matrix.max())
            im_rsync = ax[1].imshow(rsync_matrix, vmin=min_rsync, vmax=max_rsync,
                                    cmap=cmap_matrix)
            ax[1].invert_yaxis()
            ax[1].set_xticks([])
            ax[1].set_yticks([])
            ax[1].axis('off')
            ax[1].title.set_text('Rsync')
            cax = fig.add_axes(
                [ax[1].get_position().x1 + 0.01, ax[1].get_position().y0, 0.02, ax[1].get_position().height])
            fig.colorbar(im_rsync, ax=ax[1], cax=cax)

            plot_path = os.path.join(figures_connect_dir, f'{param}.svg')
            fig.savefig(plot_path, bbox_inches='tight')
            print(plot_path, 'SAVED')

        df = pd.DataFrame()
        col_param_name = 'angle difference'
        if experiment == 'proximity':
            col_param_name = 'distance'

        print('PEARSON KEYS', experiment_results['pearson'].keys())

        df[col_param_name] = col_param
        df['Pearson'] = experiment_results['pearson'].values()  # col_pearson
        df['Pearson p'] = experiment_results['pearson_p'].values()  # col_pearson_p
        df['Spearman'] = experiment_results['spearman'].values()  # col_spearman
        df['Spearman p'] = experiment_results['spearman_p'].values()  # col_spearman_p
        df['Pearson std'] = experiment_results['pearson_std'].values()
        df['Spearman std'] = experiment_results['spearman_std'].values()

        stat_path = os.path.join(stats_dir, experiment, 'connections_corr.xlsx')
        df.to_excel(stat_path)