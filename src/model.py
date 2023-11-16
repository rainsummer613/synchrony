import numpy as np
import os
import time

from utils import mkdir_experiment_setup, add_noise

class IzhikevichKorndoerfer:

    def __init__(self,
                 random_seed, connectivity_matrix, voltage_noise_amount,
                 input_strength, input_firing_rate,
                 input_noise_level, input_noise_type,
                 delta_t):

        self.connectivity_matrix = connectivity_matrix
        self.delta_t = delta_t
        self.n_neurons = connectivity_matrix.shape[0]

        self.voltage_noise_amount = voltage_noise_amount  # 0.4 from the code
        self.input_firing_rate = input_firing_rate  # 40 from paper
        self.input_strength = input_strength  # 2.5 from the code
        self.input_noise_type = input_noise_type
        self.input_noise_level = input_noise_level

        np.random.seed() if random_seed == None else np.random.seed(random_seed)

        self.voltage = 30.0  # from code
        self.recov = 30.0  # from code

        # Izhikevich parameters
        self.threshold = 30.0  # from paper
        self.izhi_a = 0.01  # from paper
        self.izhi_b = -0.1  # from paper
        self.izhi_voltage_reset = -65.0  # from paper
        self.izhi_recov_reset = 12.0  # from paper

        # Parameters for weird formulas
        self.reverse_potential_exc = 0.0  # from paper
        self.reverse_potential_inh = -80.0  # from paper

        # concatenate excitatory and inhibitory reverse potential to use in the vector form
        self.reverse_potential = np.full(self.n_neurons, self.reverse_potential_exc)

        self.duration_transmitter_after_spike_lat = 0.02  # from code
        self.duration_transmitter_after_spike_ext = 0.02  # from code
        self.rising_time_open_receptors_lat = 8.0  # from paper
        self.rising_time_open_receptors_ext = 8.0  # from paper
        self.decay_time_open_receptors_lat = 8.0  # from paper
        self.decay_time_open_receptors_ext = 8.0  # from paper
        self.maximum_transmitter_concentration_lat = 1.0  # from code
        self.maximum_transmitter_concentration_ext = 1.0  # from code

    def _time_step(self, voltage, recov,
                   open_receptors_lat, open_receptors_ext,
                   transmitter_concentrations_lat, transmitter_concentrations_ext,
                   substeps=2):
        '''
        Worker function for simulation. given parameters and current state variables, compute next time step
        '''

        # Add random noise
        voltage += (np.random.random(len(voltage)) - 0.5) * self.voltage_noise_amount

        ### Step forward ###
        for i in range(substeps):  # for numerical stability, execute at least two substeps per ms
            # Update lateral input
            input_lat = -np.dot(self.connectivity_matrix, open_receptors_lat) * (voltage - self.reverse_potential)

            # Update external input
            input_ext = -self.input_strength * open_receptors_ext * (voltage - self.reverse_potential)

            # Update Izhikevich voltage and recovery
            voltage += (1.0 / substeps) * self.delta_t * (
                        0.04 * (voltage ** 2) + (5 * voltage) + 140 - recov + input_lat + input_ext)
            recov += (1.0 / substeps) * self.delta_t * self.izhi_a * (self.izhi_b * voltage - recov)

            # Update fraction of open receptors in lateral synapses
            open_receptors_lat += (1.0 / substeps) * self.delta_t * (
                        self.rising_time_open_receptors_lat * transmitter_concentrations_lat * (
                            1 - open_receptors_lat) - self.decay_time_open_receptors_lat * open_receptors_lat)  # 364

            # Update fraction of open receptors in external synapses
            open_receptors_ext += (1.0 / substeps) * self.delta_t * (
                        self.rising_time_open_receptors_ext * transmitter_concentrations_ext * (
                            1 - open_receptors_ext) - self.decay_time_open_receptors_ext * open_receptors_ext)  # 367

        ### Update fired neurons ###
        fired = voltage > self.threshold  # array of indices of spikes
        voltage[fired] = self.izhi_voltage_reset  # reset the voltage of every neuron that fired
        recov[fired] += self.izhi_recov_reset  # update the recovery variable of every fired neuron

        return voltage, recov, open_receptors_lat, open_receptors_ext, fired

    def simulate(self, thalamic_input, length, verbose=True):

        # ======================|     Initialization      |======================#
        '''
        #Initialize thalamic noise
        thalamic_input_noisy = thalamic_input.copy()
        if self.input_noise_level > 0:
            thalamic_input_noisy = add_noise(thalamic_input, self.input_noise_type, self.input_noise_level)
        '''

        # Initialize voltage variable
        voltage_out = np.zeros((self.n_neurons, length), dtype=np.float32)
        voltage_out[:, 0] = self.voltage * (np.random.random(self.n_neurons) - 0.5)  # starting voltage

        # Initialize recovery variable
        recov = self.recov * (np.random.random(self.n_neurons) - 0.5)  # starting recovery

        # Initialize firings
        firings_out = np.zeros((self.n_neurons, length), dtype=np.float32)

        # Initialize open receptors variables
        open_receptors_lat = np.zeros((self.n_neurons,), dtype=np.float32)
        open_receptors_ext = np.zeros((self.n_neurons,), dtype=np.float32)

        '''
        Tl - spike_arrivals_lat
        Hs - transmitter_concentrations_lat

        Tnla - spike_arrivals_ext
        Hna - transmitter_concentrations_ext

        Isyn - input_lat

        Vna - reverse_potential

        taus - self.duration_transmitter_after_spike_lat
        tauna - self.duration_transmitter_after_spike_ext
        '''
        t0 = time.perf_counter()  # py3.8

        # Initialize spike time arrival and transmitter concentrations for lateral synapses
        # save the last time each neuron fired. initial values are <0 to ensure that there is no neurotransmitter in the synapes at t=0
        spike_arrivals_lat = np.full((self.n_neurons,), -2 * self.duration_transmitter_after_spike_lat,
                                     dtype=np.float32)
        transmitter_concentrations_lat = np.zeros((self.n_neurons,), dtype=np.float32)

        # Initialize spike time arrival and transmitter concentrations for external synapses
        spike_arrivals_ext = np.full((self.n_neurons,), -2 * self.duration_transmitter_after_spike_ext,
                                     dtype=np.float32)
        transmitter_concentrations_ext = np.zeros((self.n_neurons,), dtype=np.float32)

        # ======================|     Simulation      |======================#

        for t in range(1, length - 1):

            ### Before time step ###
            current_time = t * self.delta_t

            # set lateral transmitter concentrations to max if neurotransmitter is still present
            mask = (spike_arrivals_lat + self.duration_transmitter_after_spike_lat) > current_time
            transmitter_concentrations_lat = np.zeros((self.n_neurons,), dtype=np.float32)
            transmitter_concentrations_lat[mask] = self.maximum_transmitter_concentration_lat

            # set external transmitter concentrations to max if neurotransmitter is still present
            mask = (spike_arrivals_ext + self.duration_transmitter_after_spike_ext) > current_time
            transmitter_concentrations_ext = np.zeros((self.n_neurons,), dtype=np.float32)
            transmitter_concentrations_ext[mask] = self.maximum_transmitter_concentration_ext

            ### Make a time step update ###
            voltage_out[:, t], recov, open_receptors_lat, open_receptors_ext, firings_out[:, t] = self._time_step(
                voltage_out[:, t - 1], recov,
                open_receptors_lat, open_receptors_ext,
                transmitter_concentrations_lat, transmitter_concentrations_ext,
            )

            ### After time step ###
            # update lateral spiking input
            spike_arrivals_lat[firings_out[:, t] == 1] = current_time

            # Initialize thalamic noise
            thalamic_input_noisy = thalamic_input.copy()
            if self.input_noise_level > 0:
                thalamic_input_noisy = add_noise(thalamic_input, self.input_noise_type, self.input_noise_level)

            # sample probabilistic external spiking input
            random_numbers = np.random.random(self.n_neurons)
            thalamic_input_thresholds = thalamic_input_noisy * self.input_firing_rate * self.delta_t
            spike_arrivals_ext[random_numbers < thalamic_input_thresholds] = current_time

            if verbose and t % 1000 == 0:
                print(
                    f"Simulated {str(t)} μs of braintime in {str(time.perf_counter() - t0)} s of computer time.")  # py3.8

        t1 = time.perf_counter()
        print(f"Simulation took {str((t1 - t0))} s")
        return voltage_out, firings_out