import gym
import numpy as np
import argparse
import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from wrapper import F1TenthWrapper

# --- Configuration ---
MAP_NAME = 'maps/vegas' 

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO agent for F1Tenth.")
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=500000, 
        help="Total number of training timesteps (default: 500000)"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="f1tenth_ppo_v7", 
        help="Name of the final saved model zip file (default: f1tenth_ppo_v7)"
    )
    parser.add_argument(
        "--save_dir", 
        type=str, 
        default="models/ppo/", 
        help="Directory where the final model will be saved (default: models/ppo/)"
    )
    parser.add_argument(
        "--checkpoint_dir", 
        type=str, 
        default=None, 
        help="Directory for periodic checkpoints. If None, no checkpoints are saved."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading Map: {MAP_NAME}")
    
    # 1. Create the Environment
    env = gym.make('f110_gym:f110-v0', map=MAP_NAME, map_ext='.png', num_agents=1)
    
    # 2. Apply our Wrapper
    env = F1TenthWrapper(env)
    
    # 3. Initialize the Agent (PPO)
    print(f"Initializing PPO Agent ({args.model_name})...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./f1tenth_logs/", 
        learning_rate=0.0003,
        batch_size=64,
        gamma=0.99
    )

    # 4. Setup Callbacks
    callback = None
    if args.checkpoint_dir:
        print(f"Checkpointing enabled. Saving to: {args.checkpoint_dir}")
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        callback = CheckpointCallback(
            save_freq=100000, 
            save_path=args.checkpoint_dir,
            name_prefix=args.model_name
        )

    # 5. Train
    print(f"Starting training for {args.timesteps} steps...")
    model.learn(total_timesteps=args.timesteps, callback=callback)

    # 6. Save the Final Model
    os.makedirs(args.save_dir, exist_ok=True)
    final_model_path = os.path.join(args.save_dir, args.model_name)
    model.save(final_model_path)
    print(f"Training complete! Final model saved to '{final_model_path}.zip'")

    # 7. Close Environment
    env.close()

if __name__ == "__main__":
    main()
