#!/bin/bash
#SBATCH --job-name example           # how the job will be named
#SBATCH --output logs/omdv2_out.log     # filename for the output messages
#SBATCH --error logs/omdv2_error.log    # filename for the error messages
#SBATCH -n 12                         #number of CPUs
#SBATCH --mem-per-cpu=10G             #memory per CPU
#SBATCH --tmp=1000                   # disk space (per node, i.e.: per job)
#SBATCH --time=10:00:00               # maximum time to complete (after which it is killed)

cd ~/bio_deepl_appl
conda activate machine_learning
mkdir out
python src/main.py --epochs 25 --lr 5e-4 --batchsize 256 --verbosity 1
