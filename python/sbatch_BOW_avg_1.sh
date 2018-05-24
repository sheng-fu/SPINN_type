#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=BOW
#SBATCH --mail-type=END
#SBATCH --mail-user=sfw268@nyu.edu
#SBATCH --output=BOW_errors_avg_1.txt
# Run the training script
PYTHONPATH=$PYTHONPATH: python run_BOW_classifier_avg_1.py