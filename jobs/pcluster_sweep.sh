#!/bin/bash

#SBATCH --job-name conceptlab
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus 1
#SBATCH --cpus-per-task 8
#SBATCH --partition braid
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=homefs/home/debroue1/projects/conceptlab/logs/cluster/%x_%j.out.out
#SBATCH --error=/homefs/home/debroue1/projects/conceptlab/logs/cluster/%x_%j.out.err

#### SBATCH --qos=preempt

# Usage: sbatch pcluster_sweep.sh debroue1/conceptlab/2bcfk8lh

# it can be useful to know what node the job ran on
hostname

source ~/.bashrc  # or ~/.bash_profile, ~/.zshrc based on your shell

cd /homefs/home/debroue1/projects/conceptlab/scripts

runner=(uv run)          # default
if [[ "${1:-}" == "--conda_mode" ]] || [[ "${1:-}" == "-c" ]]; then
  runner=()
  shift
else
  conda deactivate
fi


# allow SWEEP_ID from env var or first non-flag argument
SWEEP_ID="${SWEEP_ID:-${1:-}}"
if [[ -z "$SWEEP_ID" ]]; then
  echo "Error: no SWEEP_ID provided (set \$SWEEP_ID or pass as arg)" >&2
  exit 1
fi

"${runner[@]}" wandb agent "${SWEEP_ID}"
