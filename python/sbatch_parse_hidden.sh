#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=144:00:00
#SBATCH --mem=40GB
#SBATCH --job-name=parse_hidden
#SBATCH --mail-type=END
#SBATCH --mail-user=sfw268@nyu.edu
#SBATCH --output=parsed_hidden.txt

# Run the training script
module purge
module load pytorch/python3.6/0.3.0_4
module load numpy/python3.6/intel/1.14.0

PYTHONPATH=$PYTHONPATH: python parse_output_hidden.py