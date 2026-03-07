import gym
import numpy as np
from gym import spaces
from collections import deque

class F1TenthWrapper(gym.Wrapper):
    def __init__(self, env, n_stack=3):
        super().__init__(env)
        
        self.n_stack = n_stack
        
        # --- Checkpoint System (Vegas Map Approximate Sectors) ---
        self.checkpoints = [
            np.array([1.5, 7.5]),     # Checkpoint 1: First Corner Entry
            np.array([-5.0, 7.5]),    # Checkpoint 2: Back Straight Entry
            np.array([-5.0, -8.0]),   # Checkpoint 3: Back Straight Exit
            np.array([1.5, -8.0]),    # Checkpoint 4: Final Corner Entry
        ]
        self.checkpoint_radius = 2.5  # Meters
        self.current_checkpoint_idx = 0
        
        # --- FIX 1: Manually define the Action Space ---
        # PPO needs to know the limits of the controls.
        # We define a Box: [Steering Angle, Speed]
        # Limits: Steer +/- 3.14 rads, Speed 0 to 20 m/s
        self.action_space = spaces.Box(
            low=np.array([-3.14, 0.0]), 
            high=np.array([3.14, 20.0]), 
            dtype=np.float32
        )
        
        # --- Define Observation Space (20 Lidar + 2 Physics) * n_stack ---
        self.lidar_rays = 20
        self.obs_dim = self.lidar_rays + 2
        # Use -inf to inf to prevent crashes if a value goes slightly negative
        low = -np.inf * np.ones(self.obs_dim * self.n_stack)
        high = np.inf * np.ones(self.obs_dim * self.n_stack)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        # Frame stack buffer
        self.frames = deque(maxlen=self.n_stack)
        
        # Steering smoothing tracker
        self.prev_steering = 0.0
        
        # v5: Mechanical steering damper (EMA Filter)
        self.actual_steering = 0.0
        self.alpha = 0.3

    def reset(self, **kwargs):
        # FIX: If SB3 calls reset() without a position, inject a default one.
        if 'poses' not in kwargs:
            kwargs['poses'] = np.array([[0.0, 0.0, 0.0]])
        # Handle the F1TENTH tuple return
        obs, reward, done, info = self.env.reset(**kwargs)
        
        processed_obs = self.process_obs(obs)
        for _ in range(self.n_stack):
            self.frames.append(processed_obs)
            
        self.prev_steering = 0.0
        self.actual_steering = 0.0
        self.current_checkpoint_idx = 0
        self.prev_distance = None
            
        return self._get_stacked_obs()

    def step(self, action):
        # Extract raw target steering from AI
        target_steering = action[0]
        
        # v5: Apply EMA Low-Pass Filter (Mechanical Damper)
        self.actual_steering = (self.alpha * target_steering) + ((1.0 - self.alpha) * self.actual_steering)
        
        # Override AI action with physically smoothed steering
        action[0] = self.actual_steering
        
        # 1. Reshape Action for F1TENTH
        # Apply Speed Floor: Clip speed to [4.5, 20.0]
        steer = action[0]
        speed = np.clip(action[1], 4.5, 20.0)
        action_reshaped = np.array([[steer, speed]])
        
        # 2. Step the simulator
        obs, original_reward, done, info = self.env.step(action_reshaped)
        
        # 3. Process observation and update stack
        current_processed = self.process_obs(obs)
        self.frames.append(current_processed)
        stacked_obs = self._get_stacked_obs()
        
        # 4. CUSTOM REWARD CALCULATION
        # Extract physics from the observation
        # (Recall: We put velocity at index -2 in process_obs)
        velocity = current_processed[-2]
        
        # Calculate steering jitter penalty
        # Note: We penalize the RAW target output, not the damped physical steering
        current_steering = target_steering
        steering_diff = abs(current_steering - self.prev_steering)
        self.prev_steering = current_steering
        jitter_penalty = steering_diff * 0.05
        
        # A. Gaussian Speed Reward
        target_speed = 7.0
        variance = 2.0
        speed_reward = np.exp(-((velocity - target_speed)**2) / (2 * variance**2))
        
        # B. Survival Penalty & Jitter smoothing
        if done:
            reward = -10.0  # Big penalty for death (overrides other rewards)
        else:
            reward = (speed_reward * 2.0) - jitter_penalty
            
        # 5. Checkpoint Tracking & Distance Progress Reward
        agent_x = obs['poses_x'][0]
        agent_y = obs['poses_y'][0]
        agent_pos = np.array([agent_x, agent_y])
        
        checkpoint_cleared = False
        progress_reward = 0.0
        
        if self.current_checkpoint_idx < len(self.checkpoints):
            target_cp = self.checkpoints[self.current_checkpoint_idx]
            dist = np.linalg.norm(agent_pos - target_cp)
            
            if self.prev_distance is None:
                self.prev_distance = dist
                
            distance_closed = self.prev_distance - dist
            progress_reward = distance_closed * 10.0
            
            if dist < self.checkpoint_radius:
                self.current_checkpoint_idx += 1
                checkpoint_cleared = True
                progress_reward = 0.0  # Reset for this step so we don't penalize the jump
                self.prev_distance = None  # Will be recalculated next step against new target
            else:
                self.prev_distance = dist
                
        # Inject tracking data into the info dict so PPO ignores it, but GA can use it
        info['checkpoint_cleared'] = checkpoint_cleared
        info['current_checkpoint_idx'] = self.current_checkpoint_idx
        info['progress_reward'] = progress_reward
        
        return stacked_obs, reward, done, info

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

    def _get_stacked_obs(self):
        assert len(self.frames) == self.n_stack
        return np.concatenate(self.frames)
