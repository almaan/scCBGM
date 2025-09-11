#!/bin/bash
# Usage: ./init_sweep.sh ../fm_config/sweeps/kang/kang_cbm.yaml

set -e  # stop on error

if [ $# -lt 1 ]; then
  echo "Usage: $0 <sweep_config.yaml> [project] [entity]"
  exit 1
fi

CONFIG=$1
PROJECT=${2:-conceptlab}   # default project if not passed
ENTITY=${3:-debroue1}   # default entity if not passed

echo "Launching sweep from config: $CONFIG"
echo "Project: $PROJECT | Entity: $ENTITY"

# Step 1: Create the sweep, capture sweep ID
SWEEP_ID=$(uv run wandb sweep -p "$PROJECT" -e "$ENTITY" "$CONFIG" 2>&1 | \
           grep -oE "$ENTITY/$PROJECT/[a-z0-9]+")

if [ -z "$SWEEP_ID" ]; then
  echo "❌ Failed to extract sweep ID"
  exit 1
fi

echo "✅ Created sweep: $SWEEP_ID"

# Step 2: Launch an agent for the sweep
#uv run wandb agent "$SWEEP_ID"
