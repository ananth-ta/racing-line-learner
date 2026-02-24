import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from wrapper import F1TenthWrapper

# --- Configuration ---
MAP_NAME = 'maps/vegas' 
TIMESTEPS = 500000       # v6 Gaussian reward from scratch
MODEL_NAME = "models/ppo/f1tenth_ppo_v6"

def main():
    print(f"Loading Map: {MAP_NAME}")
    
    # 1. Create the Environment
    # We load the 'vegas' map we set up earlier
    env = gym.make('f110_gym:f110-v0', map=MAP_NAME, map_ext='.png', num_agents=1)
    
    # 2. Apply our Wrapper
    # This converts the 1080-point Lidar scan into a clean 22-float vector
    env = F1TenthWrapper(env)
    
    # 3. Sanity Check (Optional)
    # This checks if our wrapper follows the Gym standard. 
    # If this fails, SB3 will likely crash later.
    # check_env(env)  # <--- Uncomment if you want to verify your wrapper logic

    # 4. Initialize the Agent (PPO)
    # MlpPolicy: Standard Dense Neural Network (Good for vector inputs)
    # verbose=1: Print training stats to terminal
    # tensorboard_log: Save graphs to track learning curve
    print("Initializing PPO Agent (v6 from scratch)...")
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./f1tenth_logs/", 
        learning_rate=0.0003,
        batch_size=64,
        gamma=0.99
    )

    # 5. Train
    print(f"Starting training for {TIMESTEPS} steps...")
    # The progress bar will show 'fps' (frames per second). 
    # On an RTX 4070, this should fly (1000+ fps).
    model.learn(total_timesteps=TIMESTEPS)

    # 6. Save the Model
    # This creates a .zip file you can load later
    model.save(MODEL_NAME)
    print(f"Training complete! Model saved to '{MODEL_NAME}.zip'")

    # 7. Close Environment
    env.close()

if __name__ == "__main__":
    main()
