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
            with open('maps/vegas_optimal_line.csv', 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                all_wps = []
                for row in reader:
                    if row:
                        all_wps.append(np.array([float(row[0]), float(row[1])]))
                # Downsample waypoints: take every 10th waypoint so agents don't skip over them going too fast
                self.waypoints = all_wps[::10]
        except Exception as e:
            print(f"Error loading Vegas optimal line: {e}")
            self.waypoints = [np.array([0.0, 0.0])]
            
        self.checkpoint_radius = 4.0  # Meters (Capture radius for waypoints, increased for sparser gates)
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
        
        # --- Define Observation Space ---
        self.lidar_rays = 20
        # GA: 20 Lidar + 3 Physics (Vel, Yaw, Heading Error)
        # PPO: 20 Lidar + 2 Physics (Vel, Yaw) + 10 Radar + 6 History
        self.obs_dim = self.lidar_rays + (3 if self.algo == 'ga' else 18)
        # Use -inf to inf to prevent crashes if a value goes slightly negative
        low = -np.inf * np.ones(self.obs_dim * self.n_stack)
        high = np.inf * np.ones(self.obs_dim * self.n_stack)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        
        # Frame stack buffer
        self.frames = deque(maxlen=self.n_stack)
        
        # Action history and Curriculum speed cap for PPO
        self.action_history = deque([(0.0, 0.0)] * 3, maxlen=3)
        self.current_speed_cap = 20.0
        
        # Steering smoothing tracker
        self.prev_steering = 0.0
        
        # v5: Mechanical steering damper (EMA Filter)
        self.actual_steering = 0.0
        self.alpha = 0.3
        self.steps_since_last_gate = 0

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
        self.steps_since_last_gate = 0
        self.action_history = deque([(0.0, 0.0)] * 3, maxlen=3)
        
        # Dynamically target the closest waypoint to the spawn location
        if self.waypoints:
            agent_pos = np.array([obs['poses_x'][0], obs['poses_y'][0]])
            distances = [np.linalg.norm(agent_pos - wp) for wp in self.waypoints]
            self.current_waypoint_idx = int(np.argmin(distances))
            # Advance by 1 to target the *next* waypoint, not the one we are sitting on
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
            self.prev_distance = np.linalg.norm(agent_pos - self.waypoints[self.current_waypoint_idx])
        else:
            self.current_waypoint_idx = 0
            self.prev_distance = None
            
        return self._get_stacked_obs()

    def step(self, action):
        # Extract raw target steering from AI
        target_steering = action[0]
        target_speed = action[1]
        
        self.action_history.append((target_steering, target_speed))
        
        # v5: Apply EMA Low-Pass Filter (Mechanical Damper)
        self.actual_steering = (self.alpha * target_steering) + ((1.0 - self.alpha) * self.actual_steering)
        
        # Override AI action with physically smoothed steering
        action[0] = self.actual_steering
        
        # 1. Reshape Action for F1TENTH
        # Apply Speed Floor and Curriculum Cap
        steer = action[0]
        speed = np.clip(action[1], 4.5, getattr(self, 'current_speed_cap', 20.0))
        action_reshaped = np.array([[steer, speed]])
        
        # 2. Step the simulator
        obs, original_reward, done, info = self.env.step(action_reshaped)
        
        # 3. Process observation and update stack
        current_processed = self.process_obs(obs)
        self.frames.append(current_processed)
        stacked_obs = self._get_stacked_obs()
        
        # 4. CUSTOM REWARD CALCULATION
        # Extract physics from the raw observation dict directly
        velocity = float(obs['linear_vels_x'][0])
        
        # Calculate steering jitter penalty
        # Note: We penalize the RAW target output, not the damped physical steering
        current_steering = target_steering
        steering_diff = abs(current_steering - self.prev_steering)
        self.prev_steering = current_steering
        jitter_penalty = steering_diff * 0.05
        
        # A. Gaussian Speed Reward
        target_speed_gauss = 10.0  # Increased peak speed incentive
        variance = 3.0             # Widened variance to allow for smooth acceleration
        speed_reward = np.exp(-((velocity - target_speed_gauss)**2) / (2 * variance**2))
        
        # 5. Waypoint Tracking & Distance Progress Reward
        agent_x = obs['poses_x'][0]
        agent_y = obs['poses_y'][0]
        agent_pos = np.array([agent_x, agent_y])
        
        # B. Base Reward & Jitter smoothing
        if done:
            reward = -1000.0  
        else:
            # PPO gets the Gaussian speed reward to learn basic throttle control
            if self.algo == 'ppo':
                reward = (speed_reward * 2.0) - jitter_penalty
                if self.waypoints:
                    target_cp = self.waypoints[self.current_waypoint_idx]
                    cte = np.linalg.norm(agent_pos - target_cp)
                    reward -= (cte * 0.1)  
            else:
                # GA gets EXACTLY ZERO base reward. It must earn distance points to survive.
                reward = -jitter_penalty
        
        waypoint_cleared = False
        progress_reward = 0.0
        distance_closed = 0.0
        direction_debuff = 0.0
        self.steps_since_last_gate += 1
        
        if self.waypoints:
            target_cp = self.waypoints[self.current_waypoint_idx]
            dist = np.linalg.norm(agent_pos - target_cp)
            
            if self.prev_distance is None:
                self.prev_distance = dist
                
            distance_closed = self.prev_distance - dist
            
            # Direction Debuff: If distance_closed is negative, the agent is moving away from the gate
            if distance_closed < 0:
                direction_debuff = abs(distance_closed) * 100.0  # Double penalty for wrong direction
                if self.algo == 'ppo':
                    reward -= direction_debuff
            
            # Anti-donut zeroing
            if self.algo == 'ga':
                if float(velocity) > 2.0 and distance_closed > 0:
                    progress_reward = distance_closed * 100.0
                else:
                    progress_reward = -direction_debuff  # Punish wrong way heavily
            else:
                progress_reward = (distance_closed * 10.0) - direction_debuff
            
            if dist < self.checkpoint_radius:
                self.current_waypoint_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
                waypoint_cleared = True
                progress_reward = 0.0  # Reset for this step so we don't penalize the jump
                # Export the sector time
                info['steps_taken'] = self.steps_since_last_gate
                self.steps_since_last_gate = 0
                
                # Calculate distance to the NEW waypoint immediately
                new_target = self.waypoints[self.current_waypoint_idx]
                self.prev_distance = np.linalg.norm(agent_pos - new_target)
            else:
                self.prev_distance = dist
                
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
        
        if self.algo == 'ga':
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
        else:
            car_x = obs['poses_x'][0]
            car_y = obs['poses_y'][0]
            car_yaw = obs['poses_theta'][0]
            
            radar_coords = []
            if self.waypoints:
                for i in range(5):
                    idx = (self.current_waypoint_idx + i) % len(self.waypoints)
                    wp_x, wp_y = self.waypoints[idx]
                    dx = wp_x - car_x
                    dy = wp_y - car_y
                    # Transform to car local frame
                    local_x = dx * math.cos(car_yaw) + dy * math.sin(car_yaw)
                    local_y = -dx * math.sin(car_yaw) + dy * math.cos(car_yaw)
                    radar_coords.extend([local_x, local_y])
            else:
                radar_coords = [0.0] * 10
                
            radar_arr = np.array(radar_coords)
            hist_arr = np.array(self.action_history).flatten()
            
            final_obs = np.concatenate([downsampled_scans, velocity, yaw_rate, radar_arr, hist_arr])
            
        return final_obs.astype(np.float32)

    def set_speed_cap(self, max_speed):
        self.current_speed_cap = max_speed

    def _get_stacked_obs(self):
        assert len(self.frames) == self.n_stack
        return np.concatenate(self.frames)
