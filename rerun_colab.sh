#!/bin/bash
# 1. Source the Miniforge profile to make the 'conda' command available
echo "Sourcing Miniforge profile..."
source /content/miniforge/etc/profile.d/conda.sh

# 2. Activate the pre-existing environment
echo "Activating f1tenth environment..."
conda activate f1tenth

# 3. Fire up the training loop
echo ""
read -p "Which algorithm do you want to train? [ga/ppo]: " algo_choice

if [ "$algo_choice" == "ppo" ]; then
    read -p "Enter a name for your PPO model (e.g., ppo_v1): " model_name
    echo "Starting training (PPO)..."
    python train.py --model_name "$model_name" --timesteps 5000000
else
    echo "Starting training (GA)..."
    python ga_train.py
fi