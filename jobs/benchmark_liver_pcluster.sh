#!/bin/bash

#SBATCH --job-name liver_cemvae
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task 8
#SBATCH --partition braid
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/homefs/home/debroue1/projects/conceptlab/logs/cluster/%x_%j.out.out
#SBATCH --error=/homefs/home/debroue1/projects/conceptlab/logs/cluster/%x_%j.out.err

#### SBATCH --qos=preempt


# it can be useful to know what node the job ran on
hostname

source ~/.bashrc  # or ~/.bash_profile, ~/.zshrc based on your shell
conda deactivate

cd /homefs/home/debroue1/projects/conceptlab/scripts

uv run python benchmark_liver_baselines.py --cemvae # REPLACE WITH YOUR SWEEP PATH !!!!
