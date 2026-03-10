import gym
import numpy as np
from wrapper import F1TenthWrapper

def get_action(obs):
    # Obs = [20 Lidar Rays, Velocity, Yaw_Rate]
    lidar = obs[:-2] 
    
    # "Follow The Gap": Steer towards the furthest ray
    # 1. Find the index of the max lidar value (deepest gap)
    target_idx = np.argmax(lidar)
    
    # 2. Convert Index (0-19) to Steering Angle (Radians)
    # The Lidar covers -135 to +135 degrees (-2.35 to 2.35 rad)
    angle_min = -2.35
    angle_max = 2.35
    angle_step = (angle_max - angle_min) / 20.0
    
    # Calculate the angle of that target ray
    steering_angle = angle_min + (target_idx * angle_step)
    
    # 3. Simple Speed Controller
    # If steering is small (straight), go fast (5.0). Else slow down (2.0).
    speed = 5.0 if abs(steering_angle) < 0.5 else 2.0
    
    return np.array([[steering_angle, speed]])

# --- Run the Loop ---
if __name__ == "__main__":
    # Create the env
    env = gym.make('f110_gym:f110-v0', map='maps/vegas', map_ext='.png', num_agents=1)
    env = F1TenthWrapper(env, algo='ppo')
    
    # Reset with pose (x=0, y=0, theta=0)
    obs = env.reset(poses=np.array([[0.0, 0.0, 0.0]]))
    
    done = False
    lap_time = 0.0
    
    print("Baseline running... (Press Ctrl+C to stop)")
    while not done:
        action = get_action(obs)
        obs, reward, done, info = env.step(action)
        lap_time += 0.01
        
        # Stop if we crash (f1tenth sets done=True on collision)
        if done:
            print(f"CRASH! Final Time: {lap_time:.2f}s")
            break
