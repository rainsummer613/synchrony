import argparse
import multiprocessing as mp
import os
import sys
import yaml

paths = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
	os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')]
sys.path.extend(paths)

from src.draw import draw_angle_stimulus, draw_stimulus_proximity
from src.simulation import SimulationExperiment
from src.utils import save_npy, exp_params

def run_exp(exp, exp_param, data_dir, config_file_path, repeat=100):
    log_dir = os.path.join(data_dir, "logs", exp)
    os.makedirs(log_dir, exist_ok=True)

    with open(config_file_path, "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    width, height = config["stimulus"]["width"], config["stimulus"]["height"]

    if exp == "similarity":
        stimulus = draw_angle_stimulus(angle=exp_param, strength=1, width=width, height=height, half=True)
    elif exp == "continuity":
        stimulus = draw_angle_stimulus(angle=exp_param, strength=1, width=width, height=height, half=False)
    elif exp == "proximity":
        stimulus = draw_stimulus_proximity(distance=exp_param, width=width, height=height)

    for trial in range(repeat):
        save_dir = os.path.join(log_dir, str(exp_param), "firings")
        if os.path.isdir(save_dir) and len(os.listdir(save_dir)) >= repeat:
            break
        else:
            try:
                print(f"Experiment {exp} START trial {trial}\n")

                # initialize and run simulation object
                simulation = SimulationExperiment(stimulus=stimulus, config_file_path=config_file_path)
                voltage, firings = simulation.run(config["simulation"]["length"], True)
                save_npy(logs_dir_path=log_dir, data=firings, plot_type="firings", param=exp_param, trial=trial)
            except Exception as e:
                print(f"Experiment {exp} ERROR in trial {trial}: {str(e)}")
    print(f"Finished simulation for parameter {exp_param}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=int, default=0,
                        help="simulate data for the experiment. 0: proximity, 1: similarity, 2: continuity")
    parser.add_argument('-m', '--multiproc', type=int, default=1,
                        help="use multiprocesing for running simulations. 0: no, 1: yes")
    args = parser.parse_args()
    multiproc = bool(args.multiproc)
    exp = list(exp_params.keys())[int(args.experiment)]

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    config_file_path = os.path.join(data_dir, "config.yaml")

    # run the experiment
    if multiproc is False:
        for param in exp_params[exp][1][:1]:
            print(f'Start {exp} simulations for parameter {param}')
            run_exp(exp, param, data_dir, config_file_path,)
    else:
        procs = []
        for param in exp_params[exp][1]:
            print(f'Start {exp} simulations for parameter {param}')
            proc = mp.Process(target=run_exp, args=(exp, param, data_dir, config_file_path,),
                              name=f'{exp}_{param}')
            procs.append(proc)
            proc.start()

        # complete the processes
        for proc in procs:
            proc.join()

        for proc in procs:
            proc.terminate()

