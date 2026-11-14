import gym
import numpy as np
import math
from gym import spaces
from collections import deque

class F1TenthWrapper(gym.Wrapper):
    def __init__(self, env, n_stack=3, algo='ga'):
        super().__init__(env)
        
        self.n_stack = n_stack
        self.algo = algo
        
        # --- Waypoint System (Vegas Optimal Line) ---
        import csv
        self.waypoints = []
        try:
            with open('maps/cota_optimal_line.csv', 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if row:
                        self.waypoints.append(np.array([float(row[0]), float(row[1])]))
        except Exception as e:
            print(f"Error loading Vegas optimal line: {e}")
            self.waypoints = [np.array([0.0, 0.0])]
            
        self.checkpoint_radius = 1.5  # Meters (Capture radius for waypoints)
        self.current_waypoint_idx = 0
        
        # --- FIX 1: Manually define the Action Space ---
        # PPO needs to know the limits of the controls.
        # We define a Box: [Steering Angle, Speed]
        # Limits: Steer +/- 3.14 rads, Speed 0 to 20 m/s
        self.action_space = spaces.Box(
            low=np.array([-3.14, 0.0]), 
            high=np.array([3.14, 20.0]), 
            dtype=np.float32
        )
        
        # --- Define Observation Space (20 Lidar + 3 Physics: Vel, Yaw, Heading Error) * n_stack ---
        self.lidar_rays = 20
        self.obs_dim = self.lidar_rays + 3
        
        self.ppo_time_since_last_cp = 0
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
        self.current_waypoint_idx = 0
        self.prev_distance = None
        self.ppo_time_since_last_cp = 0
            
        return self._get_stacked_obs()

    def step(self, action):
        # Extract raw target steering from AI
        target_steering = action[0]
        
        # v5: Apply EMA Low-Pass Filter (Mechanical Damper)
        self.actual_steering = (self.alpha * target_steering) + ((1.0 - self.alpha) * self.actual_steering)
        
        # Override AI action with physically smoothed steering
        action[0] = self.actual_steering
        
        # 1. Reshape Action for F1TENTH
        # Apply Speed Floor: Clip speed to [6.5, 20.0] to prevent tight low-speed spirals
        steer = action[0]
        speed = np.clip(action[1], 6.5, 20.0)
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
        target_speed = 10.0
        variance = 2.0
        speed_reward = np.exp(-((velocity - target_speed)**2) / (2 * variance**2))
        
        # B. Survival Penalty & Jitter smoothing
        if done:
            reward = -10.0  # Big penalty for death (overrides other rewards)
        else:
            reward = (speed_reward * 2.0) - jitter_penalty
            
        # 5. Waypoint Tracking & Distance Progress Reward
        agent_x = obs['poses_x'][0]
        agent_y = obs['poses_y'][0]
        agent_pos = np.array([agent_x, agent_y])
        
        waypoint_cleared = False
        progress_reward = 0.0
        
        if self.waypoints:
            target_cp = self.waypoints[self.current_waypoint_idx]
            dist = np.linalg.norm(agent_pos - target_cp)
            
            if self.prev_distance is None:
                self.prev_distance = dist
                
            distance_closed = self.prev_distance - dist
            progress_reward = distance_closed * 10.0
            
            if dist < self.checkpoint_radius:
                self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
                waypoint_cleared = True
                progress_reward = 0.0  # Reset for this step so we don't penalize the jump
                # Calculate distance to the NEW waypoint immediately
                new_target = self.waypoints[self.current_waypoint_idx]
                self.prev_distance = np.linalg.norm(agent_pos - new_target)
            else:
                self.prev_distance = dist
                
        if self.algo == 'ppo':
            reward += progress_reward
            if waypoint_cleared:
                reward += 50.0
                self.ppo_time_since_last_cp = 0
            else:
                self.ppo_time_since_last_cp += 1
                
            # Apply a static steering penalty to discourage extreme drifting for PPO
            # (Equivalent to late-stage GA curriculum)
            steering_penalty = (abs(target_steering) ** 2) * 2.0
            reward -= steering_penalty

            if self.ppo_time_since_last_cp > 2000:
                done = True
                reward = -10.0

        # Inject tracking data into the info dict so PPO ignores it, but GA can use it
        info['waypoint_cleared'] = waypoint_cleared
        info['current_waypoint_idx'] = self.current_waypoint_idx
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
        
        # Calculate heading error compass to target waypoint
        car_x = obs['poses_x'][0]
        car_y = obs['poses_y'][0]
        car_yaw = obs['poses_theta'][0]
        
        if self.waypoints:
            target_x, target_y = self.waypoints[self.current_waypoint_idx]
            target_angle = math.atan2(target_y - car_y, target_x - car_x)
            heading_error = target_angle - car_yaw
            # Normalize to [-pi, pi]
            heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
        else:
            heading_error = 0.0
            
        heading_arr = np.array([heading_error])
        final_obs = np.concatenate([downsampled_scans, velocity, yaw_rate, heading_arr])
            
        return final_obs.astype(np.float32)

    def _get_stacked_obs(self):
        assert len(self.frames) == self.n_stack
        return np.concatenate(self.frames)
