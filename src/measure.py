import itertools
import numpy as np
from elephant.spike_train_dissimilarity import van_rossum_distance, victor_purpura_distance
from elephant.spike_train_synchrony import spike_contrast
from neo.core import SpikeTrain

class Measure:
    def __init__(self, firings, metric='rsync'):
        '''
        Function for calculating dissimilarity between multiple spike trains.
        Args:
            metric = metric name. Available metrics: van_rossum, victor_purpura.
            firings = list of sequences of the neuron firings.
        '''
        metrics_available = ('van_rossum', 'victor_purpura', 'spike_contrast', 'rsync')
        if metric not in metrics_available:
            raise Exception('Please select from the available metrics: van_rossum, victor_purpura, spike_contrast, rsync')
        self.metric = metric
        
        if len(firings) < 2:
            raise Exception('Please select 2 or more spike trains to compare')
        if len(set([len(f) for f in firings])) > 1:
            raise Exception('Please select spike trains of the similar length')
            
        self.firings = firings
        self.length = len(firings[0])
        
    def _transform_firing(self, spike_train):
        return SpikeTrain(list(np.nonzero(spike_train))[0], units='ms', t_stop=self.length)
        
    def _pairwise_sim(self, firing1, firing2):
        train1 = self._transform_firing(firing1)
        train2 = self._transform_firing(firing2)

        if self.metric == 'van_rossum':
            return van_rossum_distance((train1, train2))[0,1]
        return victor_purpura_distance((train1, train2))[0,1]
    
    def similarity(self, **kwargs):
        '''
        Measure the distance between arbitrary amount of neurons.
        '''
        if self.metric == 'spike_contrast':
            trains = [self._transform_firing(firing) for firing in self.firings]
            return 1 - spike_contrast(trains)
        
        elif self.metric == 'rsync':
            
            def exp_convolve(spike_train):
                tau = 3.0  # ms
                delta_t = kwargs['delta_t'] * kwargs['downsample']  # 0.5
                exp_kernel_time_steps = np.arange(0, tau*10, delta_t)
                decay = np.exp(-exp_kernel_time_steps/tau)
                exp_kernel = decay
                
                return np.convolve(spike_train, exp_kernel, 'same')  # 'valid'

            self.firings = np.apply_along_axis(exp_convolve, 1, self.firings)
                
            meanfield = np.mean(self.firings, axis=0) # spatial mean across cells, at each time
            variances = np.var(self.firings, axis=1)  # variance over time of each cell
            return np.var(meanfield) / np.mean(variances)
        
        else:
            pairs = list(itertools.combinations(range(len(self.firings)), 2))
            similarities = [self._pairwise_sim(self.firings[pair[0]], self.firings[pair[1]]) for pair in pairs]

        return np.mean(similarities)
        '''
        return {'median': np.median(similarities),
                'mean': np.mean(similarities),
                'max': np.max(similarities),
                'min': np.min(similarities)}
        '''