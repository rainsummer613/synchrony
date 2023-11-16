# Cortical spike synchrony as a measure of Gestalt structure
This repository contains the code used for the publication Cortical spike synchrony as a measure of Gestalt structure: link.
It includes all scripts used to create a model, simulate neuronal firings data, calculate relevant statistics and produce figures.

## Running a model
Here we explain how the code base is structured and how to run the scripts to reproduce our results.

### How to run the model
First, create a Python environment with **Python 3.9.7** and install everything from `requirements.txt` with pip (`pip install -r requirements.txt`). 

Second, change parameters in the `data/config.yaml` file or run it with default config parameters, as used for the paper.

Then, run scripts from the folder `paper_scripts` in the following order:
1) `paper_scripts/simulate_data.py` simulates data for one of 3 experiments and saves spike traces to the folder `data/logs/experiment_name/parameter_value/firings` (e.g. `data/logs/proximity/0/firings`, where 0 means between-segment distance = 0). To simulate data for proximity, run `python simulate_data.py -e 0`, for similarity − `python simulate_data.py -e 1`, for continuity − `python simulate_data.py -e 2`.
2) **Alternatively to step 1**, run file `sim_synchrony.sh` once to simulate 3 experiments in parallel. This option requires SLURM system and relatively large computing resources. 
3) `paper_scripts/analyze_data.py` iterates over files with spikу traces, computes experimental and control rsync values for different stimuli and experiments and saves them into corresponding files in the folder `data/logs/experiment_name` (e.g. `data/logs/proximity`).
4) `paper_scripts/parse_logs.py` parses previously generated log files, extracts rsync values and runs statistical tests for various Gestalt-likeness for each experiment. The statistics will be saved in the folder `data/stats`.
5) `paper_scripts/plot_figures.py` plots relevant figures with the rsync between-group statistics.

And you can run the following files if you want to look at the stimuli:
- `paper_scripts/plot_all_stimuli.py` creates a figure which shows the stimuli for all Gestalt-likeness conditions and experiments.
- `paper_scripts/plot_man_mix_stimuli.py` plots only stimuli with minimal and maximal Gestalt-likeness.

### Where to find results
After you ran all your simulations, parsed the logs and created figures as in the steps above:
- `data/logs` contains all the log files.
- `data/plots` contains all the plots.
- `data/stats` contains all calculated statistics.
- `data/connectivity` contains connectivity matrix created for your simulations.

### Source code
The folder `src` contains all source files for building and running the model:
- `src/draw.py` contains functions to create some simple input stimuli.
- `src/image_preprocessor.py` helps to preprocess the input images: detect edges and transform them to black-and-white.
- `src/network.py` contains the logic for building a connectivity matrix between neurons.
- `src/model.py` defines the Izhikevich model, and how neurons update their values over time. `Izhikevich` is the main class to run a model.
- `src/simulation.py` is the main file to run the entire simulation. It calls preprocessing functions, starts the building of the connectivity matrix and transmits the parameters to the Izhikevich model.
- `src/measure.py` helps to measure synchrony among the arbitrary group of neurons.
- `src/utils.py` contains helper functions for various scripts.

## Paper summary: theoretical background for understanding the model
If you just wanted to run the model, you can stop reading here. The following part only summarizes the original paper.

### Main idea
Neurons in the brain cortex can fire synchronously in various situations. This project is modeling firing in *primary visual cortex*, or *V1 area*. Then, firing synchrony of the nruons in the model in response to stimuli of various Gestalt-likeness is measured.

### About V1 area
To understand what's going on in the model, we need to remember how V1 area is organized. V1 neurons are specialized. Different neurons look at different regions of the input images. And different neurons can recognize lines of particulat orientation. So, each neuron is looking at some specific region and can recognize the line of some specific orientation in this regions. How does this all look?
- V1 neurons are grouped into structures called **columns**. Neurons of one columns are looking at one specific part of the image. Columns which are next to each other are also looking at the parts of the image next to each other. This is called **retinotopy**.  
- Several columns form one **hypercolumn**. It is processing information from a specific part of the image, and different columns inside this hypercolumn can recognize different angles at this specific part.  

```
neurons >> columns >> hypercolumns
```

### V1 connectivity and synchrony
Synchrony in V1 is possible due to intracortical horizontal connections between neurons (Stettler, Das et al., 2002). The main rule is:
```
Stronger connections lead to a greater synchrony. 
```
The strength of connections depends on two factors:
- input familiarity,
- geometrical characteristics of the input image.

Input familiarity means that neurons have seen the similar input before. This is in focus of Korndörfer et al. (2017). If neurons are familiar with the input, the horizontal connections are stronger. Stronger connections lead to a greater synchrony.

What about the geometrical characteristics? Well, neurons which are spatially close to each other (and also look at the same part of the image - remember retinotopy) have stronger connections. And neurons which respond to similar angles also form stronger connections (Kohn & Smith, 2005).

So, neurons that are: a) next to each other, b) recognizing specific angles - should be connected more strongly. 
Importantly, such connectivity likely originates from the acquired visual experience. The V1 horizontal connections can change, learn, adapt to the new experience both in children and adults. Thus, the horizontal connectivity in V1 reflects the acquired visual experience. This experience can be manifested as aggregate statistics of all visual stimuli experienced throughout the life (Onat, Jancke, & König, 2013). And the numeruous research shows that aggregate statistics of natural images are consistent with the Gestalt laws of visual perception (Brunswik & Kamiya 1953, Elder & Goldberg, 2002; Geisler et al., 2001).

To summarize: most often experienced and important natural visual stimuli usually have the Gestalt-like structure, and the interneuronal connectivity reflects this. The connectivity, in turn, is assumed to be the basis of spike synchrony.

### Gestalt structure
In our work, we consider three Gestalt laws, and describe them on the example of simple visual stimuli - lines.
- **proximity**: if two line pieces are located close to each other, they are perceived as one line. If they distant from each, they will likely be percieved as separate objects.
- **similarity**: two lines having similar angle orientation tend to be percieved as one object.
- **continuity**: if  two lines form a continuous shape or patterns, we perceive them as one line.

### Experiments
For the simulations, we used multiple stimuli for each Gestalt principle. These stimuli vary from very Gestalt-like (for example, to lines next to each other) to less Gestalt-like (two lines far from each other). 

We ran the model to generate firing response to different stimuli and calculated synchrony of their firings. Then, we had a look at how low or high synchrony was depending on the Gestalt-likeness of the stimulus. 

### Our model
We built a model of V1 area: a neural network, which consists of **Izhikevich** neurons (Izhikevich, 2003). What does it mean?  
The neuron has two intenal variables: `membrane potential` and `recovery variable`. The model operates over time: every time step (e.g. 1 millisecond) several things happen to each neuron:

1. A neuron receives external input + some input from other neurons.
2. It updates its internal variables `membrane potential` and `recovery variable` according to specific formulas.
3. If the value of a `membrane potential` exceeds a certain activation threshold, the neuron produces a spike.
4. After spiking both `membrane potential` and `recovery variable` are reset to initial values.

### Modeling horizontal connections
Wait, but what does this all have to do with horizontal connections?  
Well, remember that each neurons receives the external input and some input from other neurons? The external input comes from the input stimulus. But the input from other neurons is coming to our neuron through the connections.  

Before running the model, we built a connectivity matrix: is specifies the strength of the connection between each pair of neurons. Most of the connections are equal to 0, but neighboring neurons have non-zero connections. The strength of each connection depends on the angle that neurons are recognizing.  

Our model can recognize 4 angles: `0`, `45`, `90` and `135` degrees. The connection between neurons which are recognizing `0` and `45` degrees angles is *stronger* than the connection between `0` and `90` neurons.

### Simulation results
So, we ran the model with a specific connectivity structure with various visual stimuli as input. And we found out that synchrony indeed depends on Gestalt-likeness of the input stimulus. 

### References
* Brunswik, E., & Kamiya, J. (1953). Ecological cue-validity of proximity and of other Gestalt factors. The American journal of psychology, 66(1), 20-32.
* Elder, J. H., & Goldberg, R. M. (2002). Ecological statistics of Gestalt laws for the perceptual organization of contours. Journal of Vision, 2(4), 5-5.
* Geisler, W. S., Perry, J. S., Super, B. J., & Gallogly, D. P. (2001). Edge co-occurrence in natural images predicts contour grouping performance. Vision research, 41(6), 711-724.
* Izhikevich, E. M. (2003). Simple model of spiking neurons. *IEEE Transactions on neural networks*, 14(6), 1569-1572.
* Kohn, A., & Smith, M. A. (2005). Stimulus dependence of neuronal correlation in primary visual cortex of the macaque. *Journal of Neuroscience*, 25(14), 3661–3673.
* Korndörfer, C., Ullner, E., García-Ojalvo, J., & Pipa, G. (2017). Cortical spike synchrony as a measure of input familiarity. *Neural computation*, 29(9), 2491-2510. 
* Onat, S., Jancke, D., & König, P. (2013). Cortical long-range interactions embed statistical knowledge of natural sensory input: a voltage-sensitive dye imaging study. F1000Research, 2.
* Stettler, D. D., Das, A., Bennett, J., & Gilbert, C. D. (2002). Lateral connectivity and contextual interactions in macaque primary visual cortex. *Neuron*, 36(4), 739–750. 
