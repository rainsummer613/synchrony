import json
import numpy as np
import os
import yaml
from itertools import combinations, product, permutations

from scipy import spatial

from draw import draw_angle_stimulus
from image_preprocessor import ImagePreprocessor
from result_plotter import ResultPlotter
from utils import Observable, sigmoid, d2_to_d1, d1_to_d2, generate_pattern, np_encoder

class Connectivity(Observable):
    def __init__(self, connectivity_dir):
        '''
        Class for building a matrix of horizontal connections between neurons.
        Args:
            connectivity_dir: path to the connectivity file to save or upload
            filters = convolution filters for angles detection
        '''
        super().__init__()
        
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
        self.connectivity_dir = os.path.join(data_dir, connectivity_dir)

        #check if connectivity directory exists and mkdir otherwise
        if not os.path.isdir(self.connectivity_dir): 
            os.mkdir(self.connectivity_dir)

    def check_if_network_file_exists(self, connectivity_file_path):

        #check if connectivity directory exists and mkdir otherwise
        if not os.path.isdir(os.path.split(connectivity_file_path)[0]): 
            os.mkdir(os.path.split(connectivity_file_path)[0])
        return os.path.isfile(connectivity_file_path)

class AngleSpatialConnectivity(Connectivity):
    '''
    Args:
        filters = convolution filters for angles detection
    '''

    def __init__(self, connectivity_dir, filters, connectivity_params, width, height):
        super().__init__(connectivity_dir)
        self.filters = filters
        self.angle_strength = connectivity_params["angle_strength"]
        self.spatial_strength = connectivity_params["spatial_strength"]
        self.total_strength = connectivity_params["total_strength"]
        self.width = width
        self.height = height

    def get_connectivity_file_path(self):
        connectivity_ID = f"{self.width}x{self.height}" + \
                          f"_{self.angle_strength}vs{self.spatial_strength}vs{self.total_strength}".replace(
                              ".", "-")
        connectivity_dir_exp = os.path.join(self.connectivity_dir, f'{connectivity_ID}')
        connectivity_file_path = os.path.join(connectivity_dir_exp, 'connectivity.npy')
        return connectivity_ID, connectivity_dir_exp, connectivity_file_path

    def spatial_connect(self):
        '''
        Calculating spatial connectivity between all neurons.
        Args:
            width = width of the grid with neurons
            height = height of the grid with neurons
        '''

        # create an array with all neuron coordinates
        neurons_coord = np.array(list(product(range(self.height), range(self.width))), dtype='float32')

        # calculate Chebyshev distance between each pair
        distances = spatial.distance.cdist(neurons_coord, neurons_coord, 'chebyshev')

        # duplicate the results for each group of neurons recognizing a specific angle
        distances = np.tile(distances, (len(self.filters), len(self.filters)))

        distances[distances < 0.5] = 0

        # return inverted distance: the bigger distances, the weaker connections
        return self.spatial_strength / (distances + 1)

    def angle_connect(self):
        '''
        Calculating angular connectivity between all neurons.
        Bigger angle difference -> weaker connection.
        '''

        def angle_diff(x, y):
            '''
            Calculate difference between two angles up to 180 degrees
            Args:
                x = one angle of interest
                y = another angle of interest
            '''
            abs_diff = 90 - abs(abs(x - y) - 90)
            return abs_diff

            # get angle resolution

        angle_res = 22.5  # 180//len(self.filters)

        # list of angles detected, e.g. [0, 45, 90, 135]
        angles = [angle_res * i for i in range(len(self.filters))]

        # number of neurons processing each angle
        vec_len = self.width * self.height

        # create an empty array for angle differences between each pair of neurons
        angle_diffs = np.zeros((vec_len * len(self.filters), vec_len * len(self.filters)), dtype='float32')

        # calculate angle difference between each pair of neurons
        for i in range(len(self.filters)):
            for j in range(len(self.filters)):
                angle_diffs[vec_len * i:vec_len * (i + 1), vec_len * j:vec_len * (j + 1)] = angle_diff(angles[i],
                                                                                                       angles[
                                                                                                        j]) / angle_res

        # return inverted differences: the bigger differences, the weaker connections
        return self.angle_strength / (angle_diffs + 1)

    def _sliding_window_coords(self, array, window_size, step_size=1):
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        if isinstance(step_size, int):
            step_size = (step_size, step_size)

        # Generate the starting and ending indices for each window
        start_coords = [
            np.arange(0, array.shape[i] - window_size[i] + 1, step_size[i])
            for i in range(2)
        ]
        end_coords = [
            np.arange(window_size[i], array.shape[i] + 1, step_size[i])
            for i in range(2)
        ]

        # Create a meshgrid of the starting and ending coordinates
        xx, yy = np.meshgrid(start_coords[0], start_coords[1], indexing="ij")
        start_coords = np.stack((xx, yy), axis=-1).reshape(-1, 2)
        xx, yy = np.meshgrid(end_coords[0], end_coords[1], indexing="ij")
        end_coords = np.stack((xx, yy), axis=-1).reshape(-1, 2)

        coords = list(zip(start_coords, end_coords))
        return coords

    def _get_index_groups(sel, res):
        filter_indices = np.indices((res, res)).T.reshape((res * res, 2))
        index_groups = dict(enumerate(list(combinations(filter_indices, res))))
        bad_groups = []

        for group_num, group in index_groups.items():
            distances = spatial.distance.cdist(group, group, 'chebyshev')
            np.fill_diagonal(distances, res)
            if np.max(np.min(distances, 1)) > 1:
                bad_groups.append(group_num)

        for group1, group2 in combinations(list(index_groups.keys()), 2):
            if group1 not in bad_groups and group2 not in bad_groups:
                diff = (np.array(index_groups[group2]) - np.array(index_groups[group1])).T
                unique = np.count_nonzero(np.diff(np.sort(diff)), axis=1) + 1

                if unique.sum() == 2:
                    bad_groups.append(group2)
        return {key: value for key, value in index_groups.items() if key not in bad_groups}

    def build(self):
        '''
        Build a connection matrix which considers spatial distance and angle difference between all pairs of neurons
        '''

        connectivity_ID, connectivity_dir_exp, connectivity_path = self.get_connectivity_file_path()

        if self.check_if_network_file_exists(connectivity_path):
            print(f"load {connectivity_ID}")
            connectivity_matrix = np.load(connectivity_path)
            self.notify("file_exists",
                        os.path.join(connectivity_dir_exp, "total_connectivity_matrix.png"))

        elif self.total_strength == 0.0:
            print(f"generate new empty {connectivity_ID}")
            connectivity_matrix_dim = self.height * self.width * len(self.filters)
            connectivity_matrix = np.zeros((connectivity_matrix_dim, connectivity_matrix_dim))

            np.save(connectivity_path, connectivity_matrix)
            self.notify(connectivity_matrix, 'total connectivity matrix', (3, 3), len(self.filters),
                        connectivity_dir_exp)
        else:
            print(f"generate new {connectivity_ID}")

            # count spatial connection weights
            spatial_connectivity_matrix = self.spatial_connect()

            # count angle connection weights
            angle_connectivity_matrix = self.angle_connect()

            # count resulting connection weights
            connectivity_matrix = (spatial_connectivity_matrix + angle_connectivity_matrix)
            np.fill_diagonal(connectivity_matrix, 0)  # turn connections of neurons to themselves to zero
            connectivity_matrix *= self.total_strength

            # plot and save connectivity matrices
            np.save(connectivity_path, connectivity_matrix)
            self.notify(connectivity_matrix, 'total connectivity matrix', (5, 5), len(self.filters),
                        connectivity_dir_exp)
            self.notify(spatial_connectivity_matrix, 'spatial connectivity matrix', (5, 5), len(self.filters),
                        connectivity_dir_exp)
            self.notify(angle_connectivity_matrix, 'angle connectivity matrix', (5, 5), len(self.filters),
                        connectivity_dir_exp)
        return connectivity_matrix

class RandomClusterConnectivity(Connectivity):
    '''
    Args:
        filters = convolution filters for angles detection
    '''

    def __init__(self, connectivity_relative_directory_path, cluster_params=None):

        super().__init__(connectivity_relative_directory_path)

    def build(self, width, height,
              total_connect_strength,
              exc_inh_connect_strength, inh_exc_connect_strength,
              num_clusters=10, random_seed=None):

        connectivity_ID, connectivity_dir, connectivity_file_path, clusters_file_path = self.get_connectivity_file_path(
            width, height,
            exc_inh_connect_strength=exc_inh_connect_strength,
            inh_exc_connect_strength=inh_exc_connect_strength,
            angle_connect_strength=0,
            spatial_connect_strength=0,
            total_connect_strength=total_connect_strength)

        self.connectivity_file_path = connectivity_file_path
        self.clusters_file_path = clusters_file_path
        self.connectivity_dir = connectivity_dir

        # Initialize arrays with coordinates of all neurons
        all_y, all_x = np.indices((height, width))
        all_coords = list(zip(np.concatenate(all_y), np.concatenate(all_x)))

        if self.check_if_network_file_exists(connectivity_file_path) and self.check_if_network_file_exists(
                clusters_file_path):

            print(f"load {connectivity_ID}")
            connectivity_matrix = np.load(connectivity_file_path)

            # clusters = np.load(clusters_file_path)

            # self.notify("file_exists", os.path.join(connectivity_dir, "total connectivity matrix".replace(" ", "_")+".png"), connectivity_dir)
            # self.notify('random cluster connectivity matrix', all_x, all_y, connections)

        else:
            # Iitialize empty neuronal grid
            arr = np.zeros((height, width))
            arr_flat = arr.reshape(-1)

            connections_plot = {}
            connectivity_matrix = np.zeros((len(arr_flat), len(arr_flat)))

            # generate patterns by choosing a point on the network and activating a random choice of cells near it
            clusters = np.zeros((height, width, num_clusters))

            np.random.seed() if random_seed == None else np.random.seed(random_seed)

            for pat in range(num_clusters):
                clusters[:, :, pat] = generate_pattern(width=width,
                                                       height=height,
                                                       pattern_distance_prob_drop=1,
                                                       pattern_distance_prob_cutoff=0.2
                                                       )

            all_pairs = [el for el in list(product(all_coords, all_coords)) if el[0] != el[1]]
            for neuron_pair in all_pairs:

                y1, x1 = neuron_pair[0]
                y2, x2 = neuron_pair[1]

                neuron1_id = d2_to_d1(x1, y1, width)
                neuron2_id = d2_to_d1(x2, y2, width)

                # if both nodes participate in the same pattern, make a strong link,
                # with some probability depending on distance
                in_pattern = False

                pat = 0
                while in_pattern == False and pat < clusters.shape[-1]:

                    if clusters[y1, x1, pat] and clusters[y2, x2, pat]:
                        in_pattern = True
                    pat += 1

                p_connect_pattern = max(0, 1 / (spatial.distance.chebyshev((y1, x1), (y2, x2))) - 0.15)
                p_connect_background = max(0, 1 / (spatial.distance.chebyshev((y1, x1), (y2, x2))) - 0.3)

                connection_strength, connection_strength_plot = 0, 0

                if in_pattern and np.random.random() < p_connect_pattern:
                    connection_strength = total_connect_strength
                    connection_strength_plot = sigmoid(connection_strength) / 3

                # fewer and weaker background connections are created where there was no common input.
                elif np.random.random() < p_connect_background:
                    connection_strength = total_connect_strength / 15
                    connection_strength_plot = sigmoid(connection_strength) / 10

                if connection_strength != 0:
                    connections_plot[((y1, x1), (y2, x2))] = connection_strength_plot
                    connectivity_matrix[
                        [int(neuron1_id), int(neuron2_id)], [int(neuron2_id), int(neuron1_id)]] = connection_strength

            # self.notify(connectivity_matrix, 'random cluster connectivity matrix', all_x, all_y, connections_plot, connectivity_dir)
            np.save(connectivity_file_path, connectivity_matrix)
            np.save(clusters_file_path, clusters)

        return connectivity_matrix