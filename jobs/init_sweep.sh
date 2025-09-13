#!/bin/bash
# Usage: ./init_sweep.sh ../fm_config/sweeps/kang_cbm.yaml

set -e  # stop on error

if [ $# -lt 1 ]; then
  echo "Usage: $0 <sweep_config.yaml> [project] [entity]"
  exit 1
fi

runner=(uv run)
for arg in "$@"; do
  if [[ "$arg" =~ ^(--conda-mode|-c)$ ]]; then
    runner=()
    # rebuild args list without the flag
    set -- "${@/$arg}"
    break
  fi
done

CONFIG=$1
PROJECT=${2:-conceptlab}
ENTITY=${3:-debroue1}

echo $CONFIG, $PROJECT, $ENTITY

echo "Launching sweep from config: $CONFIG"
echo "Project: $PROJECT | Entity: $ENTITY"

# Step 1: Create the sweep, capture sweep ID

SWEEP_PATH="$(
  "${runner[@]}" wandb sweep -p "$PROJECT" -e "$ENTITY" "$CONFIG" 2>&1 |
    grep -oE "${ENTITY}/${PROJECT}/sweeps/[a-z0-9]+" | head -n1
)"

SWEEP_ID="${SWEEP_PATH/\/sweeps/}"   # remove the /sweeps segment

if [ -z "$SWEEP_ID" ]; then
  echo "❌ Failed to extract sweep ID"
  exit 1
fi

echo "✅ Created sweep: $SWEEP_ID"

# Step 2: Launch an agent for the sweep
#uv run wandb agent "$SWEEP_ID"
