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
        import math
        self.waypoints = []
        try:
            with open('maps/vegas_optimal_line.csv', 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # Skip header
                for row in reader:
                    if row:
                        self.waypoints.append(np.array([float(row[0]), float(row[1])]))
                        
            if len(self.waypoints) > 1:
                self.start_x = self.waypoints[0][0]
                self.start_y = self.waypoints[0][1]
                self.start_theta = math.atan2(self.waypoints[1][1] - self.start_y, self.waypoints[1][0] - self.start_x)
            else:
                self.start_x, self.start_y, self.start_theta = 0.0, 0.0, 0.0
                
        except Exception as e:
            print(f"Error loading Vegas optimal line: {e}")
            self.waypoints = [np.array([0.0, 0.0])]
            self.start_x, self.start_y, self.start_theta = 0.0, 0.0, 0.0
            
        self.checkpoint_radius = 3.5  # Meters (Capture radius for waypoints)
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
        self.obs_dim = self.lidar_rays + (4 if self.algo == 'ga' else 2) # Changed 3 to 4        # Use -inf to inf to prevent crashes if a value goes slightly negative
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
            kwargs['poses'] = np.array([[self.start_x, self.start_y, self.start_theta]])
        # Handle the F1TENTH tuple return
        obs, reward, done, info = self.env.reset(**kwargs)
        
        processed_obs = self.process_obs(obs)
        for _ in range(self.n_stack):
            self.frames.append(processed_obs)
            
        self.prev_steering = 0.0
        self.actual_steering = 0.0
        self.current_waypoint_idx = 0
        self.prev_distance = None
            
        return self._get_stacked_obs()

    def step(self, action):
        # Extract raw target steering from AI and scale to [-3.14, 3.14]
        target_steering = action[0] * 3.14
        
        # v5: Apply EMA Low-Pass Filter (Mechanical Damper)
        self.actual_steering = (self.alpha * target_steering) + ((1.0 - self.alpha) * self.actual_steering)
        
        # Override AI action with physically smoothed steering
        action[0] = self.actual_steering
        
        # 1. Reshape Action for F1TENTH
        # Map speed from Tanh [-1, 1] up to physical speeds [4.5, 20.0]
        steer = action[0]
        speed = 4.5 + ((action[1] + 1.0) / 2.0) * (20.0 - 4.5)
        action_reshaped = np.array([[steer, speed]])
        
        # 2. Step the simulator
        obs, original_reward, done, info = self.env.step(action_reshaped)
        
        # 3. Process observation and update stack
        current_processed = self.process_obs(obs)
        self.frames.append(current_processed)
        stacked_obs = self._get_stacked_obs()        
        # 4. CUSTOM REWARD CALCULATION
        # FIX: Velocity is now at index -4 !
        # [-1]=prev_steer, [-2]=heading, [-3]=yaw, [-4]=velocity
        velocity = current_processed[-4]
        
        # Calculate steering jitter penalty dynamically based on speed
        current_steering = target_steering
        steering_diff = abs(current_steering - self.prev_steering)
        self.prev_steering = current_steering
        
        # Jitter penalty dynamically scales: more penalty on high speed straights
        # Clamped between 1.0 (corners) and 20.0 (max speed straights)
        speed_factor = np.clip(velocity, 2.0, 20.0) / 5.0 
        jitter_penalty = steering_diff * 10.0 * speed_factor
        # A. Linear Speed Reward
        # Max velocity is 20.0 m/s, yielding up to +2.0 base survival points per frame.
        speed_reward = np.clip(velocity / 10.0, 0.0, 2.0)

        # C. Wrong Way Penalty
        # If the agent is facing more than 90 degrees away from the target waypoint, heavily penalize it.
        # This absolutely murders ping-pong exploits and U-turns.
        heading_error = current_processed[-2]
        wrong_way_penalty = 10.0 if abs(heading_error) > (math.pi / 2.0) else 0.0

        # B. Survival Penalty & Jitter smoothing
        if done:
            reward = -10.0  # Big penalty for death (overrides other rewards)
        else:
            reward = speed_reward - jitter_penalty - wrong_way_penalty
            
       # 5. Waypoint Tracking & Distance Progress Reward
        agent_x = obs['poses_x'][0]
        agent_y = obs['poses_y'][0]
        agent_pos = np.array([agent_x, agent_y])
        
        waypoint_cleared = False
        progress_reward = 0.0
        distance_closed = 0.0
        
        if self.waypoints:
            target_cp = self.waypoints[self.current_waypoint_idx]
            dist = np.linalg.norm(agent_pos - target_cp)
            
            if self.prev_distance is None:
                self.prev_distance = dist
                
            distance_closed = self.prev_distance - dist
            progress_reward = distance_closed * 100.0
            
            # --- THE NEW SELF-HEALING BLOCK STARTS HERE ---
            # Calculate distance to the NEXT waypoint to see if we bypassed the current one
            next_idx = (self.current_waypoint_idx + 1) % len(self.waypoints)
            next_target = self.waypoints[next_idx]
            dist_to_next = np.linalg.norm(agent_pos - next_target)
            
            # Widen radius to 4.5m OR trigger if we accidentally got closer to the next waypoint
            if dist < 4.5 or dist_to_next < dist:
                self.current_waypoint_idx = next_idx
                waypoint_cleared = True
                progress_reward = 0.0  # Reset for this step so we don't penalize the jump
                
                # Calculate prev_distance to the NEW waypoint immediately
                new_target = self.waypoints[self.current_waypoint_idx]
                self.prev_distance = np.linalg.norm(agent_pos - new_target)
            else:
                self.prev_distance = dist
                
        # Inject tracking data into the info dict so PPO ignores it, but GA can use it
        info['waypoint_cleared'] = waypoint_cleared
        info['current_waypoint_idx'] = self.current_waypoint_idx
        info['progress_reward'] = progress_reward
        info['steering_delta'] = steering_diff # <-- Added this line   
             
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
            # Calculate heading error compass to target waypoint (WITH +3 LOOK AHEAD)
            car_x = obs['poses_x'][0]
            car_y = obs['poses_y'][0]
            car_yaw = obs['poses_theta'][0]
            
            if self.waypoints:
                # Dynamic lookahead based on speed: look further ahead on fast straights
                speed = velocity if velocity > 0 else 1.0
                lookahead = int(np.clip(speed / 3.0, 1, 5))
                
                target_idx = (self.current_waypoint_idx + lookahead) % len(self.waypoints)
                target_x, target_y = self.waypoints[target_idx]
                target_angle = math.atan2(target_y - car_y, target_x - car_x)
                heading_error = target_angle - car_yaw
                # Normalize to [-pi, pi]
                heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
            else:
                heading_error = 0.0
                
            heading_arr = np.array([heading_error])
            prev_steer_arr = np.array([self.prev_steering])
            
            # Array structure: [Scans(20), Vel(1), Yaw(1), Heading(1), PrevSteer(1)] = 24 Total
            final_obs = np.concatenate([downsampled_scans, velocity, yaw_rate, heading_arr, prev_steer_arr])
        else:
            final_obs = np.concatenate([downsampled_scans, velocity, yaw_rate])
            
        return final_obs.astype(np.float32)

    def _get_stacked_obs(self):
        assert len(self.frames) == self.n_stack
        return np.concatenate(self.frames)
