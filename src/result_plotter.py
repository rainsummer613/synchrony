import os
from typing import overload
from math import ceil

import matplotlib.pyplot as plt 
import matplotlib.image as img
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from PIL import Image

from utils import Observer, get_continuous_cmap

class ResultPlotter(Observer):

    def __init__(self, experiment_path, show_substeps, observables):
        """ResultPlotter that subscribes to Observables and plots the data everytime
        it gets notified about an update.
         
        Args:
            experiment_path = path where plots are saved to
            show_substeps = whether or not to show the plotted substeps
            observables = list of observable object to subscribe to
        """
        
        super().__init__(observables)
        self.experiment_path = experiment_path
        self.show_substeps = show_substeps

    def update(self, observable, *args, **kwargs):
        if "ImagePreprocessor" in type(observable).__name__:
            figure = self._plot_preprocessing(*args, **kwargs)
            file_dir = self.experiment_path

        elif "SimulationExperiment" in type(observable).__name__:
            figure = self._plot_results(*args, **kwargs)
            file_dir = self.experiment_path

        elif "AngleSpatialConnectivity" in type(observable).__name__:
            if str(args[0]) == "file_exists":
                plt.imshow(img.imread(args[1]))
                if self.show_substeps: plt.show()
                return
            else:
                figure = self._plot_connectivity(*args, **kwargs)
                file_dir = args[4]

        elif "RandomClusterConnectivity" in type(observable).__name__:
            figure = self._plot_cluster_connectivity(*args, **kwargs)
            file_dir = args[5]

        else:
            raise Exception(f"{type(observable).__name__} is not supported by the result_plotter")
        file_path = os.path.join(file_dir, f"{args[1]}".replace(" ", "_")+".png")
        figure.savefig(file_path)
        #plt.close(figure) bugged right now (check for when figure is matplotlib figure)
        print(f"plot saved to: {file_path}")

    @overload
    def _plot_preprocessing(self, images:list[Image.Image], plot_name:str, angular_resolution=1) -> plt:
        ...
    def _plot_preprocessing(self, images:list[list], plot_name:str, angular_resolution=1) -> plt:
        """Compile plot from list of images and save all substeps into one figure
        
        args:
            images = list of images to be plotted and saved
        """

        labels = "not_name" #just in case none of the below are true

        if plot_name == "stimulus":
            fig, ax = plt.subplots()
            ax.axis("off")
            cmap = ListedColormap(['whitesmoke', 'darksalmon'])
            ax.imshow(images, cmap=cmap)
            return fig

        elif plot_name == "image preprocessing steps":
            #image labels or Create a dynamical list of the angle labels
            labels = ["original image", "intensity values", "edge detection", "binary image"]

        elif plot_name == "angle detection maps" or plot_name ==  "angle detection filters":
            labels = [f"{i}°" for i in np.arange(0, 180, angular_resolution)]

            labels.append("This weird extra angle") #remove later
            images = np.vstack([images, np.ones((1, images.shape[1], images.shape[2]))]) #remove later
        
        #plot images on multiple axis
        fig, ax = plt.subplots(figsize=(20, 10), nrows=2, ncols=ceil(len(images)/2)) #ncols=int(len(images)/2)) #
        for i, axi in enumerate(ax.flat):
            axi.axis("off")
            axi.set_title(labels[i])
            if i < len(images):
                axi.imshow(images[i])
        plt.suptitle(plot_name)
        if plot_name == "image preprocessing steps": plt.tight_layout()

        return fig

    def _plot_results(self, data, plot_name, scatter=False):
        '''Plot voltage or spikes data'''

        if not scatter:
            h, w = data.shape
            fig, ax = plt.subplots(figsize=(8,8))
            plt.suptitle(plot_name)
            plt.imshow(data)
            plt.colorbar()
            ax.set_aspect(w/h)

        else:
            fig, ax = plt.subplots(figsize=(8, 8))
            y, x = data.nonzero()
            ax.scatter(x, y, facecolor='Black')
        
        return fig

    #helper functions for plotting and saving the resulting connectivity matrix
    def _set_visible_labels(self, labels, step=5):
        '''Decrease the number of labels visible on the plot.'''

        for label in labels:
            if np.int32(label.get_text()) % step == 0:  
                label.set_visible(True)
            else:
                label.set_visible(False)
                
    def _plot_cluster_connectivity(self, connect_matrix, plot_name, all_x, all_y, connections, file_dir): 
        fig, ax = plt.subplots(1, 2, figsize=(20,10), constrained_layout=True)
        #fig.tight_layout()
        
        ax[1] = sns.heatmap(connect_matrix, vmax=connect_matrix.max(), square=True, cbar=True, cmap="PuRd")
        ax[1].set_title('Heatmap of connection weights', fontsize=20)
        ax[1].set_xlabel('neuron indices', fontsize=16)
        ax[1].set_ylabel('neuron indices', fontsize=16)
        ax[1].invert_yaxis()       
                
        ax[1].set_xticks(ax[1].get_yticks())
        ax[1].set_xticklabels(ax[1].get_yticklabels())
        self._set_visible_labels(ax[1].get_xticklabels(), 10)
        self._set_visible_labels(ax[1].get_yticklabels(), 10)
        
        ax[0].plot(all_x, all_y, 'ro')
        ax[0].set_title('Connections between neurons on the grid', fontsize=20)
        ax[0].set_xlabel('grid width', fontsize=16)
        ax[0].set_ylabel('grid height', fontsize=16)
        
        for connection in connections:
            coord1, coord2 = connection
            y1, x1 = coord1
            y2, x2 = coord2
            ax[0].plot([x1, x2], [y1, y2], 'k-', linewidth=connections[connection])    
        
        return plt
            
    def _plot_connectivity(self, connect_matrix, plot_name, ticks_length, filter_count, file_dir):
        '''Plot the connectivity matrix.'''

        hex_codes_matrix = ['#000000', '#6ccdc0', '#eb7341', 'ffefe9']
        cmap_matrix = get_continuous_cmap(hex_codes_matrix)

        plt.figure(figsize=(8,8))
        
        ax = sns.heatmap(connect_matrix, vmin=0, vmax=connect_matrix.max(), square=True, cmap=cmap_matrix)
        ax.set_title(plot_name, fontsize=20)
        ax.invert_yaxis()
        
        ax.set_xticks(ax.get_yticks())
        ax.set_xticklabels(ax.get_yticklabels())
        self._set_visible_labels(ax.get_xticklabels(), 10)
        self._set_visible_labels(ax.get_yticklabels(), 10)   

        return plt

    def plot_angle_tests(self, result_file_path):

        data = pd.read_csv(result_file_path)
        data = np.array(data)
        #data = np.load(result_file_path)
        data_list = [
            data[0:5, :], 
            data[5:10, :], 
            data[10:15, :], 
            data[15:20, :], 
            data[20:25, :], 
            data[25:30, :], 
            data[30:35, :], 
            data[35:40, :],
            data[40:45, :],
            data[45:50, :], 
            data[50:55, :]]
        data_list2 = [data[1:5, :], data[6:10, :], data[11:15, :], data[16:20, :], data[21:25, :], 
            data[26:30, :], data[31:35, :], data[36:40, :]]
        tags = ["s=3.0", "s=3.1", "s=3.2", "s=3.3", "s=3.4", "s=3.5", "s=3.6", "s=3.7", "s=3.8", "s=3.9", "s=4.0"]

        print(data)

        fig, ax = plt.subplots(figsize=(20, 10), nrows=2, ncols=6)
        ax = ax.flatten()
        for i, data in enumerate(data_list):
            ax[i].set_title(f'{tags[i]}', fontsize=10)
            ax[i].set_xlabel('stimuli angle', fontsize=10)
            ax[i].set_ylabel('rsync', fontsize=10)
            ax[i].plot(data[:,1], data[:,2])
        plt.suptitle("comparing different strenght values to different angle stimuli to see if sync increases")
        plt.savefig(result_file_path[:-3]+"png")
        plt.show()

    def plot_angle_tests_avg(self):
        result_data_path_90 = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),'data/result_data_90.npy')
        result_data_90 = np.load(result_data_path_90)
        avg_data_90 = np.average(result_data_90[:71,2])

        result_data_path_112 = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),'data/result_data_112.npy')
        result_data_112 = np.load(result_data_path_112)
        avg_data_112 = np.average(result_data_112[:71,2])

        result_data_path_135 = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),'data/result_data_135.npy')
        result_data_135 = np.load(result_data_path_135)
        avg_data_135 = np.average(result_data_135[:71,2])

        result_data_path_157 = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),'data/result_data_157.npy')
        result_data_157 = np.load(result_data_path_157)
        avg_data_157 = np.average(result_data_157[:65,2])

        result_data_path_180 = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))),'data/result_data_180.npy')
        result_data_180 = np.load(result_data_path_180)
        avg_data_180 = np.average(result_data_180[:71,2])

        x_val = [90, 112, 135, 157, 180]
        y_vals = [avg_data_90, avg_data_112, avg_data_135, avg_data_157, avg_data_180]  


        fig, ax = plt.subplots(figsize=(15.5, 4.5), nrows=1, ncols=1)

        #color = "white"      
        plt.setp(ax.spines.values()) #, color=color)
        plt.setp([ax.get_xticklines(), ax.get_yticklines()]) #, color=color)
        #ax = ax.flatten()
        ax.set_title(f'synchrony measure per angle size', fontsize=15) #,  color=color)
        ax.set_xlabel('stimuli angle', fontsize=15) #, color=color)
        ax.set_ylabel('rsync measure', fontsize=15) #, color=color)
        ax.plot(x_val, y_vals, "ro-")

        result_file_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__)))), 'data/sync_experiment.png')
        plt.savefig(result_file_path)
        plt.show()

if __name__ == '__main__':

    plotter_path = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), ' data/')
    plotter = ResultPlotter(plotter_path, True, [])
    plotter.plot_angle_tests_avg()
