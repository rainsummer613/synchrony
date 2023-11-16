#!/bin/bash

# Simulate data for synchrony experiments

#SBATCH -J synchrony
#SBATCH --array=0-2 # how many tasks in the array
#SBATCH -t 47:00:00
#SBATCH --cpus-per-task=6             
#SBATCH -o out/sync-exp-%a.out

# Load software
# module load Spack
spack load miniconda3@4.10.3 # module load anaconda3
source activate synchrony

# Run python script with a command line argument
srun python paper_scripts/simulate_data.py -e $SLURM_ARRAY_TASK_ID -m 1
