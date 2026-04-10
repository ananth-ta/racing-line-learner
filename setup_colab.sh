#!/bin/bash
# setup_colab.sh (Hotpatched for Miniforge + Pip Downgrade)

echo "Downloading and installing Miniforge (Bypassing Anaconda ToS)..."
wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /content/miniforge

echo "Initializing Conda Python 3.8 environment..."
source /content/miniforge/etc/profile.d/conda.sh
conda create -n f1tenth python=3.8 -y
conda activate f1tenth

echo "STAGE 1: Pre-installing legacy build tools..."
# Downgrade pip below 24.1 to bypass strict metadata parsing, along with setuptools and wheel
pip install pip==24.0 setuptools==65.5.0 wheel==0.38.4

echo "STAGE 2: Installing project dependencies..."
pip install -r requirements.txt
# --- USER PROMPT ---
echo ""
read -p "Which algorithm do you want to train? [ga/ppo]: " algo_choice

if [ "$algo_choice" == "ppo" ]; then
    read -p "Enter a name for your PPO model (e.g., ppo_v1): " model_name
    echo "Starting Phase 3 Neuroevolution (PPO)..."
    python train.py --model_name "$model_name" --timesteps 5000000
else
    echo "Starting Phase 3 Neuroevolution (GA)..."
    python ga_train.py
fi