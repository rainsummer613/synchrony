import numpy as np
import warnings

from skimage.filters import threshold_otsu
from skimage.util import view_as_windows
from PIL import Image, ImageFilter

from utils import singledispatch_instance_method, Observable

class ImagePreprocessor(Observable):

    def __init__(self, config) -> None:
        """
        args:
            config = preloaded yaml configuration file in the data directory
             containing all settings for the preprocessing
        """

        super().__init__()
        self.config = config
        self.filter_size = self.config["stimulus"]["filter_size"] if \
            self.config["stimulus"]["filter_size"] != None else \
            self._get_min_filter_size(self.config["stimulus"]["angle_resolution"],)
        self.angle_filters = self._get_angle_filters(
            self.config["stimulus"]["angle_resolution"],)
    
    @singledispatch_instance_method
    def preprocess(self, stimulus, conv_threshold_specificity=1.0,
                   detect_angles=True):
        """Preprocess the image, first from RGB to grayscale. Then apply a Sobel 
        filter on the intensity values to gain edge detection. Threshold result 
        for binary image and convolve angular filters for angle detection maps.
        
        args:
            stimulus = image input as a dictionary with stimulus itself and its coordinates
            conv_threshold_specificity = determines how closely the angle
                    filter needs to match the angle image data, i.e 0,75 would
                    count a patch matching 3/4 filter pixels
            detect_angles = if set to 'False' returns a binary edge detection
                    image without angle detection instead

        returns:
            image_angles = image angle detection maps, one for each angle in the
                    size of the original image input
        """

        print(f'Start preprocessing image object')

        image_input = Image.fromarray(stimulus["img"])
        large_stimulus = len(stimulus["coords"]) > 4
        # convert from RGB to grayscale (intensity values) and apply edge detection filter
        image_gray = image_input.convert("L")
        image_edges = image_gray.filter(ImageFilter.FIND_EDGES)

        # convert intensity values of edge detection map to a binary image with a threshold
        intensity_threshold = threshold_otsu(np.array(image_edges))
        threshold_fn = lambda pixel_value: 1 if pixel_value > intensity_threshold else 0
        image_binary = np.array(image_edges.point(threshold_fn, mode="1"))

        if not detect_angles:
            image_angles = image_binary
        else:
            self.angle_filters_padded = np.zeros((len(self.angle_filters), self.filter_size, self.filter_size))
            image_angles = np.zeros((len(self.angle_filters), image_binary.shape[0], image_binary.shape[1]))

            for i_filter, angle_filter in enumerate(self.angle_filters):
                
                # pad each filter with 0 so that they had filter_size X filter_size shape FOR PLOTTING
                self.angle_filters_padded[i_filter] = np.pad(angle_filter, ((0, self.filter_size-angle_filter.shape[0]),
                                                      (0, self.filter_size-angle_filter.shape[1])), 'constant')

                # represent image array as windows of the filter size
                windows = view_as_windows(image_binary, angle_filter.shape)
                
                # special method for finding regions matching the filter
                # find top-left coordinates of all regions which exactly match angle filter
                y, x = self._custom_filter(windows, angle_filter, large_stimulus)
                
                # mark image pixels which exactly match angle filters
                for pair in list(zip(y, x)):
                    image_angles[i_filter][pair[0]:pair[0]+angle_filter.shape[0], pair[1]:pair[1]+angle_filter.shape[1]] = angle_filter

            #plot and save figures with independent ResultPlotter module
            self.notify(image_gray, "stimulus")
            self.notify([image_gray, image_edges, image_binary], "image preprocessing steps") 
            self.notify(self.angle_filters_padded, "angle detection filters", 180/len(self.angle_filters))
            self.notify(image_angles, "angle detection maps", 180/len(self.angle_filters)) 

        return image_angles
 
    @preprocess.register(str)
    def _(self, image_input:str, conv_threshold_specificity:float=1.0, detect_angles:bool=True) -> np.ndarray:
        """Wrapper for overloading the preprocess method with a 
        file path (str) as image input"""

        print(f'Start preprocessing from file {image_input}')
        try:
            return self.preprocess(Image.open(image_input), conv_threshold_specificity, detect_angles)

        except FileNotFoundError:
            print(f'File "{image_input}" does not exist')

    @preprocess.register(np.ndarray)
    def _(self, image_input:np.ndarray, conv_threshold_specificity:float=1.0, detect_angles:bool=True) -> np.ndarray:
        """Wrapper for overloading the preprocess method with a 
        numpy array (np.ndarray) as image input"""

        print(f'Start preprocessing from numpy arrary')
        return self.preprocess(Image.fromarray(image_input), conv_threshold_specificity, detect_angles)

    def _get_angle_filters(self, angular_resolution: float) -> list[str]:
        """Create filters to detect angles in the image
        
        args:
            angular_resolution = angular resolution of the filters for angle detection
            filter_size = size of the filters for angle detection

        returns:
            set  of filters for angle detection
        """

        #Exception handling for angular resolution
        if not 1 < angular_resolution < 90:
            raise ValueError(f'Angular resolution {angular_resolution}° \
                             is out of range, must be between 1 and 90.')

        elif 180 % angular_resolution != 0:
            warnings.warn("Angular resolution doesn't devide evenly")

        #Exception handling for filter size
        elif self.filter_size < self._get_min_filter_size(angular_resolution):
            raise ValueError('Filter size is too small for specified angular resolution')

        else:
            #termporary exception until dynamic filter generation is available
            if angular_resolution != 22.5:
                raise ValueError(f'Currently only angular resolution of 22.5° is supported')

            angle_filters = [
                             [[0, 0, 0], [1, 1, 1], [0, 0, 0]]  # 180
                            ,[[0, 0, 0, 1, 1], [0, 1, 1, 0, 0]]  # 157
                            ,[[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]  # 135
                            ,[[0, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]  # 112
                            ,[[0, 1, 0], [0, 1, 0], [0, 1, 0]]  # 90
            ]
        return [np.array(el) for el in angle_filters]

    def _get_min_filter_size(self, angular_resolution:float) -> int:
        """Calculate minimal filter size given by the specified angular 
        resolution
        
        returns:
            angular_resolution = angular resolution of the filters for angle detection
            min_filter_size = min. viable filter size of the filters for angle detection
        """

        #TODO: write code to calculate min filter size based on angular_resolution
        min_filter_size = 5
        return min_filter_size

    def _custom_filter(self, arr, angle_filter, large_stimulus=True):
        height = arr.shape[0]
        width = arr.shape[1]

        all_y = []
        all_x = []

        for y in range(height):
            for x in range(width):

                if large_stimulus is True:
                    a = angle_filter[np.nonzero(angle_filter)]
                    b = arr[y][x][np.nonzero(angle_filter)]
                else:
                    a = angle_filter
                    b = arr[y][x]

                if np.all(a == b):
                    all_y.append(y)
                    all_x.append(x)
        return all_y, all_x
