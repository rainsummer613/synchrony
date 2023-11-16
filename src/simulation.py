import os
import numpy as np

from draw import draw_angle_stimulus
from result_plotter import ResultPlotter
from utils import mkdir_experiment_setup, Observable, downsample_data
from network import AngleSpatialConnectivity, RandomClusterConnectivity
from model import IzhikevichKorndoerfer
from image_preprocessor import ImagePreprocessor

class SimulationExperiment(Observable):

    def __init__(self, stimulus, config_file_path):
        """Builds the connections network and runs the simulation of a given length.

        Args:
            config_file_path = file path to the yaml config file
            stimulus = file path to the input stimulus, or the stimulus itself
        """

        super().__init__()
        print('Initialize the simulation')
        config, self.exp_id = mkdir_experiment_setup(config_file_path)
        self.delta_t = config["model"]["delta_t"]
        self.downsample = config["simulation"]["downsample"]

        # preprocessor instantiation and preprocessing of image
        preprocessor = ImagePreprocessor(config)

        # initialize the Connectivity class with or without filters
        if config["stimulus"]["detect_angles"] == True:
            connectivity = AngleSpatialConnectivity(config["experiment"]["connectivity_dir"],
                                                    preprocessor.angle_filters, config["connectivity"],
                                                    config["stimulus"]["width"], config["stimulus"]["height"])
        else:
            connectivity = RandomClusterConnectivity(config["experiment"]["connectivity_dir"])

        # initialize the Plotter instance
        if config["experiment"]["save_experiment"]:
            plotter = ResultPlotter(config["experiment"]["experiment_path"],
                config["experiment"]["show_substeps"], [preprocessor, connectivity, self])

        # transform input image to angle detection maps
        image_input = preprocessor.preprocess(stimulus,
                                    detect_angles=config["stimulus"]["detect_angles"])
        print('input shape', image_input.shape)

        # initialize connectivity matrix
        self.connectivity_matrix = connectivity.build()
        print("connectivity shape", self.connectivity_matrix.shape)

        # reshape image input for convenience
        image_input = image_input.reshape(-1)

        # full thalamic input
        n_neurons = len(self.connectivity_matrix)
        self.thalamic_input = np.zeros(n_neurons,)
        self.thalamic_input[:len(image_input)] = image_input
        print("thalamic input shape", self.thalamic_input.shape)

        # create an instance of the Izhikevich neural model
        self.model = IzhikevichKorndoerfer(connectivity_matrix=self.connectivity_matrix,
                                                 input_strength=config["input"]["strength"],
                                                 voltage_noise_amount=config["model"]["voltage_noise_amount"],
                                                 input_firing_rate=config["input"]["firing_rate"],
                                                 input_noise_level=config["input"]["noise_level"],
                                                 input_noise_type=config["input"]["noise_type"],
                                                 random_seed=config["experiment"]["random_seed"],
                                                 delta_t=config["model"]["delta_t"])

    def run(self, length=1000, verbose=False):
        time_steps = int(length / self.delta_t)

        print(f'Simulation of {time_steps} time steps started')
        voltage, firings = self.model.simulate(self.thalamic_input, time_steps, verbose)

        # downsample and plot results
        voltage = np.array(voltage[:, 0:time_steps:self.downsample], dtype='float32')
        spikes = downsample_data(firings, self.downsample)
        self.notify(voltage, "result voltage")
        self.notify(spikes, "result spikes")

        return voltage, firings

if __name__ == '__main__':

    # simulation parameters
    width, height = 10, 10
    angle = 180
    stimulus = draw_angle_stimulus(angle, strength=1, width=width, height=height, half=True)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    config_file_path = os.path.join(data_dir, 'config.yaml')

    # run simulation
    stimulus = draw_angle_stimulus(angle=angle, strength=1, width=width, height=height, half=True)
    simulation = SimulationExperiment(stimulus_file_path=stimulus['img'], config_file_path=config_file_path)
    voltage, firings = simulation.run(length=400, verbose=True)

