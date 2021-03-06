#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=SPINN
#SBATCH --mail-type=END
#SBATCH --mail-user=sfw268@nyu.edu
#SBATCH --output=SPINN_errors_1.txt

# Run the training script
PYTHONPATH=$PYTHONPATH: python run_SPINN_classifier_1.py