import gym
import numpy as np
from gym import spaces

class F1TenthWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        # We want 20 lidar beams (instead of 1080) + 2 extra states (velocity, steer)
        self.lidar_rays = 20
        
        # Define the new "clean" state space for the AI
        # [Ray 1, Ray 2, ... Ray 20, Velocity, Steer_Angle]
        low = np.zeros(self.lidar_rays + 2)
        high = np.inf * np.ones(self.lidar_rays + 2)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self, **kwargs):
        # 1. Call the weird F1TENTH reset that returns a tuple
        obs, reward, done, info = self.env.reset(**kwargs)
        
        # 2. Process just the observation part
        return self.process_obs(obs)

    def step(self, action):
        # 1. Step the environment
        obs, reward, done, info = self.env.step(action)
        
        # 2. Process the observation part and return the standard tuple
        return self.process_obs(obs), reward, done, info

    def process_obs(self, obs):
       	# 1. Get raw lidar (1080 points)
       	raw_scans = obs['scans'][0]
        
        # 2. Downsample
        step = len(raw_scans) // self.lidar_rays
        downsampled_scans = raw_scans[::step][:self.lidar_rays]
        
        # 3. Add car physics
        # linear_vels_x = How fast we are moving forward
        # ang_vels_z   = How fast we are turning (Yaw Rate)
        velocity = np.array([obs['linear_vels_x'][0]])
        yaw_rate = np.array([obs['ang_vels_z'][0]])  # <--- CHANGED THIS
        
        # 4. Stitch it together
        final_obs = np.concatenate([downsampled_scans, velocity, yaw_rate])
        return final_obs.astype(np.float32)

# --- Test the Wrapper ---
if __name__ == "__main__":
    # Load the base environment
    base_env = gym.make('f110_gym:f110-v0', map='maps/vegas', map_ext='.png', num_agents=1)
    
    # Wrap it!
    env = F1TenthWrapper(base_env)
    
    # FIX: Define the starting position (x=0, y=0, theta=0)
    initial_pose = np.array([[0.0, 0.0, 0.0]])
    
    # Pass the pose to reset()
    obs = env.reset(poses=initial_pose)
    
    print("Old input size: 1080 + physics")
    print(f"New input size: {len(obs)}")
    print(f"Sample Observation (First 5): {obs[:5]}")
