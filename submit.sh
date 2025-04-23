#!/bin/bash
#SBATCH -e polyBERT.err
#SBATCH -o polyBERT.out # File to which STDOUT will be written %j is the job #
#SBATCH -J polyBERT # Job name
#SBATCH -n 8 # Number of total cores
#SBATCH -N 1 # Number of nodes
#SBATCH --gpus=1
#SBATCH --time=48:00:00 # Runtime in D-HH:MM
#SBATCH -p batch
##SBATCH --mem=2000 # Memory pool for all cores in MB (see also --mem-per-cpu)

ulimit -n 2048

module load anaconda3
conda activate ML

echo "SLURM_NTASKS: " $SLURM_NTASKS
echo "Job started on `hostname` at `date`"
python polyBERT_Downstream.py
echo " "
echo "Job Ended at `date`"