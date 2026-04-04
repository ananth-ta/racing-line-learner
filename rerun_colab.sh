#!/bin/bash
# 1. Source the Miniforge profile to make the 'conda' command available
echo "Sourcing Miniforge profile..."
source /content/miniforge/etc/profile.d/conda.sh

# 2. Activate the pre-existing environment
echo "Activating f1tenth environment..."
conda activate f1tenth

# 3. Fire up the training loop
echo "Starting training..."
python ga_train.py