import gym
import numpy as np
from gym import spaces

class F1TenthWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        
        # --- FIX 1: Manually define the Action Space ---
        # PPO needs to know the limits of the controls.
        # We define a Box: [Steering Angle, Speed]
        # Limits: Steer +/- 3.14 rads, Speed 0 to 20 m/s
        self.action_space = spaces.Box(
            low=np.array([-3.14, 0.0]), 
            high=np.array([3.14, 20.0]), 
            dtype=np.float32
        )
        
        # --- Define Observation Space (20 Lidar + 2 Physics) ---
        self.lidar_rays = 20
        # Use -inf to inf to prevent crashes if a value goes slightly negative
        low = -np.inf * np.ones(self.lidar_rays + 2)
        high = np.inf * np.ones(self.lidar_rays + 2)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    def reset(self, **kwargs):
        # FIX: If SB3 calls reset() without a position, inject a default one.
        if 'poses' not in kwargs:
            kwargs['poses'] = np.array([[0.0, 0.0, 0.0]])
        # Handle the F1TENTH tuple return
        obs, reward, done, info = self.env.reset(**kwargs)
        return self.process_obs(obs)

    def step(self, action):
        # --- FIX 2: Reshape Action ---
        # PPO sends a 1D array: [steer, speed]
        # F1TENTH expects a 2D array: [[steer, speed]]
        action_reshaped = np.array([action])
        
        obs, reward, done, info = self.env.step(action_reshaped)
        return self.process_obs(obs), reward, done, info

    def process_obs(self, obs):
        # 1. Get raw lidar
        raw_scans = obs['scans'][0]
        
        # 2. Downsample to 20 rays
        step = len(raw_scans) // self.lidar_rays
        downsampled_scans = raw_scans[::step][:self.lidar_rays]
        
        # 3. Add physics
        velocity = np.array([obs['linear_vels_x'][0]])
        yaw_rate = np.array([obs['ang_vels_z'][0]])
        
        # 4. Concatenate
        final_obs = np.concatenate([downsampled_scans, velocity, yaw_rate])
        return final_obs.astype(np.float32)
