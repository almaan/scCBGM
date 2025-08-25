#!/bin/bash

#SBATCH --job-name conceptlab
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --partition braid
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/homefs/home/debroue1/projects/conceptlab/logs/cluster/%x_%j.out.out
#SBATCH --error=/homefs/home/debroue1/projects/conceptlab/logs/cluster/%x_%j.out.err


# it can be useful to know what node the job ran on
hostname

source ~/.bashrc  # or ~/.bash_profile, ~/.zshrc based on your shell
conda deactivate

cd /homefs/home/debroue1/projects/conceptlab/

wandb agent debroue1/conceptlab/w0k8ospv