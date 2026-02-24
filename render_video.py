"""
render_video.py - Render videos of trained F1TENTH agents (Headless Compatible)

This version generates visualizations from scratch using matplotlib,
no display/pyglet required. Works on WSL2 without X11.

Usage:
    python render_video.py                     # Render trained PPO agent
    python render_video.py --baseline          # Render FTG baseline
    python render_video.py --compare           # Side-by-side comparison
    python render_video.py --model path/to/model.zip  # Custom model path
"""

import gym
import numpy as np
import cv2
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (no display needed)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow, Wedge
from PIL import Image

from stable_baselines3 import PPO
from wrapper import F1TenthWrapper


# ============================================================================
# CONFIGURATION
# ============================================================================

MAP_NAME = 'maps/vegas'
MODEL_PATH = 'models/ppo/f1tenth_ppo_v4A.zip'
OUTPUT_DIR = 'videos'
MAX_STEPS = 2000          # Max steps per episode
FPS = 30                  # Output video framerate


# ============================================================================
# BASELINE CONTROLLER (from baseline.py)
# ============================================================================

def ftg_get_action(obs):
    """Follow-The-Gap controller."""
    lidar = obs[:-2]
    target_idx = np.argmax(lidar)
    
    angle_min = -2.35
    angle_max = 2.35
    angle_step = (angle_max - angle_min) / 20.0
    steering_angle = angle_min + (target_idx * angle_step)
    speed = 5.0 if abs(steering_angle) < 0.5 else 2.0
    
    return np.array([steering_angle, speed])


# ============================================================================
# CUSTOM RENDERER (No Display Required)
# ============================================================================

class HeadlessRenderer:
    """
    Renders F1TENTH state to images without needing a display.
    Creates a bird's-eye view with car position, lidar rays, and track.
    """
    
    def __init__(self, map_path='maps/vegas', map_ext='.png', figsize=(10, 8)):
        self.figsize = figsize
        
        # Load map image
        map_img_path = f"{map_path}{map_ext}"
        try:
            self.map_img = np.array(Image.open(map_img_path).convert('L'))
            self.map_loaded = True
            print(f"Loaded map: {map_img_path}")
        except Exception as e:
            print(f"Could not load map image: {e}")
            self.map_img = np.ones((500, 500)) * 255
            self.map_loaded = False
        
        # Load map metadata (for coordinate transform)
        yaml_path = f"{map_path}.yaml"
        try:
            import yaml
            with open(yaml_path, 'r') as f:
                map_meta = yaml.safe_load(f)
            self.resolution = map_meta.get('resolution', 0.05)
            self.origin = map_meta.get('origin', [0, 0, 0])
        except Exception as e:
            print(f"Could not load map yaml: {e}, using defaults")
            self.resolution = 0.05
            self.origin = [0, 0, 0]
        
        # Track history for trajectory visualization
        self.trajectory = []
        self.max_trajectory_points = 500
    
    def world_to_pixel(self, x, y):
        """Convert world coordinates to pixel coordinates."""
        px = int((x - self.origin[0]) / self.resolution)
        py = int(self.map_img.shape[0] - (y - self.origin[1]) / self.resolution)
        return px, py
    
    def reset(self):
        """Reset trajectory history."""
        self.trajectory = []
    
    def render(self, obs_dict, action, step, total_reward, agent_name="Agent"):
        """
        Render current state to RGB image.
        
        Args:
            obs_dict: Raw observation dictionary from F1TENTH (before wrapper processing)
            action: Current action [steering, speed]
            step: Current timestep
            total_reward: Cumulative reward
            agent_name: Name to display
        
        Returns:
            RGB numpy array (H, W, 3)
        """
        fig, axes = plt.subplots(1, 2, figsize=self.figsize)
        
        # === Left panel: Bird's eye view ===
        ax1 = axes[0]
        
        # Draw map
        ax1.imshow(self.map_img, cmap='gray', origin='upper')
        
        # Get car state
        car_x = obs_dict['poses_x'][0]
        car_y = obs_dict['poses_y'][0]
        car_theta = obs_dict['poses_theta'][0]
        velocity = obs_dict['linear_vels_x'][0]
        
        # Convert to pixel coordinates
        px, py = self.world_to_pixel(car_x, car_y)
        
        # Store trajectory
        self.trajectory.append((px, py))
        if len(self.trajectory) > self.max_trajectory_points:
            self.trajectory.pop(0)
        
        # Draw trajectory
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            ax1.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1, alpha=0.5)
        
        # Draw lidar rays
        lidar_scans = obs_dict['scans'][0]
        num_rays = len(lidar_scans)
        angle_min, angle_max = -2.35, 2.35  # -135° to +135°
        angles = np.linspace(angle_min, angle_max, num_rays)
        
        # Subsample lidar for visualization (every 20th ray)
        for i in range(0, num_rays, 54):
            ray_angle = car_theta + angles[i]
            ray_dist = min(lidar_scans[i], 10.0)  # Clamp for visualization
            
            end_x = car_x + ray_dist * np.cos(ray_angle)
            end_y = car_y + ray_dist * np.sin(ray_angle)
            
            end_px, end_py = self.world_to_pixel(end_x, end_y)
            
            # Color by distance (red=close, green=far)
            color = plt.cm.RdYlGn(min(ray_dist / 10.0, 1.0))
            ax1.plot([px, end_px], [py, end_py], color=color, linewidth=0.5, alpha=0.7)
        
        # Draw car (triangle pointing in heading direction)
        car_size = 15
        dx = car_size * np.cos(-car_theta + np.pi/2)
        dy = car_size * np.sin(-car_theta + np.pi/2)
        
        ax1.scatter(px, py, c='red', s=100, zorder=10, marker='o')
        ax1.arrow(px, py, dx, dy, head_width=8, head_length=5, fc='red', ec='red', zorder=11)
        
        # Set view centered on car
        view_radius = 150  # pixels
        ax1.set_xlim(px - view_radius, px + view_radius)
        ax1.set_ylim(py + view_radius, py - view_radius)  # Inverted y for image coords
        ax1.set_aspect('equal')
        ax1.set_title(f'{agent_name} - Bird\'s Eye View')
        ax1.axis('off')
        
        # === Right panel: Telemetry dashboard ===
        ax2 = axes[1]
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.axis('off')
        
        # Background
        ax2.add_patch(Rectangle((0, 0), 10, 10, facecolor='#1a1a2e', edgecolor='none'))
        
        # Title
        ax2.text(5, 9.2, agent_name, fontsize=16, ha='center', color='cyan', fontweight='bold')
        
        # Telemetry data
        steering_deg = np.degrees(action[0])
        speed_cmd = action[1]
        
        telemetry = [
            ('Step', f'{step}', 'white'),
            ('Time', f'{step * 0.01:.2f}s', 'white'),
            ('Reward', f'{total_reward:.1f}', 'lime' if total_reward > 0 else 'red'),
            ('', '', 'white'),  # Spacer
            ('Velocity', f'{velocity:.2f} m/s', 'cyan'),
            ('Speed Cmd', f'{speed_cmd:.2f} m/s', 'yellow'),
            ('Steering', f'{steering_deg:.1f}°', 'orange'),
            ('', '', 'white'),  # Spacer
            ('Position', f'({car_x:.1f}, {car_y:.1f})', 'white'),
            ('Heading', f'{np.degrees(car_theta):.1f}°', 'white'),
        ]
        
        y_pos = 8.0
        for label, value, color in telemetry:
            if label:
                ax2.text(1, y_pos, f'{label}:', fontsize=11, color='gray', ha='left')
                ax2.text(9, y_pos, value, fontsize=11, color=color, ha='right', fontweight='bold')
            y_pos -= 0.7
        
        # Steering indicator
        ax2.text(5, 2.5, 'Steering', fontsize=10, ha='center', color='gray')
        
        # Draw steering arc
        steer_normalized = np.clip(steering_deg / 45.0, -1, 1)  # Normalize to [-1, 1]
        arc_color = 'orange'
        
        # Center line
        ax2.plot([5, 5], [1.0, 2.0], 'gray', linewidth=2)
        
        # Steering indicator
        steer_x = 5 + steer_normalized * 2
        ax2.plot([5, steer_x], [1.5, 1.5], color=arc_color, linewidth=4)
        ax2.scatter([steer_x], [1.5], c=arc_color, s=100, zorder=10)
        
        # Speed bar
        ax2.text(5, 0.8, 'Speed', fontsize=10, ha='center', color='gray')
        speed_normalized = np.clip(velocity / 10.0, 0, 1)
        ax2.add_patch(Rectangle((2, 0.2), 6, 0.4, facecolor='#333', edgecolor='gray'))
        ax2.add_patch(Rectangle((2, 0.2), 6 * speed_normalized, 0.4, facecolor='cyan', edgecolor='none'))
        
        plt.tight_layout()
        
        # Convert to RGB array
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return img


# ============================================================================
# VIDEO RECORDING
# ============================================================================

def create_env():
    """Create F1TENTH environment (unwrapped for raw obs access)."""
    env = gym.make('f110_gym:f110-v0', map=MAP_NAME, map_ext='.png', num_agents=1)
    return env


def create_wrapped_env():
    """Create wrapped F1TENTH environment."""
    env = create_env()
    return F1TenthWrapper(env)


def record_episode(policy_fn, agent_name="Agent", max_steps=MAX_STEPS):
    """
    Record a single episode with custom rendering.
    
    Args:
        policy_fn: Function that takes wrapped obs and returns action
        agent_name: Name to display on overlay
        max_steps: Maximum steps to record
    
    Returns:
        frames: List of numpy arrays (RGB images)
        metrics: Dict with episode statistics
    """
    # Create both wrapped (for policy) and unwrapped (for rendering) access
    raw_env = create_env()
    wrapped_env = F1TenthWrapper(raw_env)
    
    renderer = HeadlessRenderer(map_path=MAP_NAME)
    renderer.reset()
    
    frames = []
    wrapped_obs = wrapped_env.reset()
    
    total_reward = 0.0
    
    print(f"Recording {agent_name}...")
    
    for step in range(max_steps):
        # Get action from policy (using wrapped observation)
        action = policy_fn(wrapped_obs)
        
        # Ensure action is flat array
        if hasattr(action, 'flatten'):
            action = action.flatten()
        
        # Get raw observation for rendering
        # F1TENTH stores state in sim.agents[i].state = [x, y, theta, velocity, steer_angle, ...]
        try:
            agent = raw_env.sim.agents[0]
            raw_obs = {
                'poses_x': [agent.state[0]],
                'poses_y': [agent.state[1]],
                'poses_theta': [agent.state[2]],
                'linear_vels_x': [agent.state[3]],
                'linear_vels_y': [0.0],
                'ang_vels_z': [agent.state[4] if len(agent.state) > 4 else 0.0],
                'scans': [agent.scan if hasattr(agent, 'scan') else np.ones(1080) * 10],
            }
        except Exception as e:
            # Fallback: construct from wrapped observation
            # with n_stack=3, current obs is the last 22 elements
            current_obs = wrapped_obs[-22:]  
            raw_obs = {
                'poses_x': [0.0],
                'poses_y': [0.0],
                'poses_theta': [0.0],
                'linear_vels_x': [current_obs[-2]],  # velocity from wrapped obs
                'linear_vels_y': [0.0],
                'ang_vels_z': [current_obs[-1]],     # yaw rate from wrapped obs
                'scans': [np.array(list(current_obs[:-2]) * 54)],  # Expand 20 rays back to ~1080
            }
            if step == 0:
                print(f"  Warning: Using fallback observation extraction: {e}")
        
        # Render frame
        try:
            frame = renderer.render(raw_obs, action, step, total_reward, agent_name)
            frames.append(frame)
        except Exception as e:
            print(f"  Render error at step {step}: {e}")
        
        # Step environment
        wrapped_obs, reward, done, info = wrapped_env.step(action)
        total_reward += reward
        
        if done:
            print(f"  Episode ended at step {step} (collision)")
            break
        
        # Progress indicator
        if step % 200 == 0 and step > 0:
            print(f"  Step {step}/{max_steps}, Reward: {total_reward:.1f}")
    
    metrics = {
        'steps': step + 1,
        'total_reward': total_reward,
        'survival_time': (step + 1) * 0.01,
    }
    
    print(f"  Finished: {metrics['steps']} steps, {metrics['survival_time']:.2f}s, Reward: {total_reward:.1f}")
    
    wrapped_env.close()
    
    return frames, metrics


def save_video(frames, output_path, fps=FPS):
    """Save frames to MP4 video."""
    if not frames:
        print("No frames to save!")
        return False
    
    Path(output_path).parent.mkdir(exist_ok=True)
    
    h, w = frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)
    
    writer.release()
    print(f"Saved video: {output_path} ({len(frames)} frames, {len(frames)/fps:.1f}s)")
    return True


def create_comparison_video(frames_a, frames_b):
    """Create side-by-side comparison video."""
    max_len = max(len(frames_a), len(frames_b))
    
    # Pad shorter video
    while len(frames_a) < max_len:
        frames_a.append(frames_a[-1].copy() if frames_a else np.zeros((480, 640, 3), dtype=np.uint8))
    while len(frames_b) < max_len:
        frames_b.append(frames_b[-1].copy() if frames_b else np.zeros((480, 640, 3), dtype=np.uint8))
    
    combined = []
    for fa, fb in zip(frames_a, frames_b):
        # Ensure same height
        ha, wa = fa.shape[:2]
        hb, wb = fb.shape[:2]
        
        if ha != hb:
            target_h = max(ha, hb)
            if ha < target_h:
                fa = cv2.resize(fa, (int(wa * target_h / ha), target_h))
            if hb < target_h:
                fb = cv2.resize(fb, (int(wb * target_h / hb), target_h))
        
        separator = np.ones((fa.shape[0], 5, 3), dtype=np.uint8) * 128
        frame = np.hstack([fa, separator, fb])
        combined.append(frame)
    
    return combined


# ============================================================================
# MAIN FUNCTIONS
# ============================================================================

def render_ppo_agent(model_path=MODEL_PATH, output_name=None):
    """Render video of trained PPO agent."""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)
    
    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    
    frames, metrics = record_episode(ppo_policy, agent_name="PPO Agent")
    
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"ppo_agent_{timestamp}.mp4"
    
    output_path = Path(OUTPUT_DIR) / output_name
    save_video(frames, output_path)
    
    return metrics


def render_baseline(output_name=None):
    """Render video of FTG baseline controller."""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    frames, metrics = record_episode(ftg_get_action, agent_name="FTG Baseline")
    
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"ftg_baseline_{timestamp}.mp4"
    
    output_path = Path(OUTPUT_DIR) / output_name
    save_video(frames, output_path)
    
    return metrics


def render_comparison(model_path=MODEL_PATH, output_name=None):
    """Render side-by-side comparison of PPO vs FTG baseline."""
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    print("=" * 60)
    print("RECORDING COMPARISON VIDEO")
    print("=" * 60)
    
    # Load PPO model
    print(f"\nLoading model: {model_path}")
    model = PPO.load(model_path)
    
    def ppo_policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    
    # Record PPO agent
    print("\n--- PPO Agent ---")
    frames_ppo, metrics_ppo = record_episode(ppo_policy, agent_name="PPO Agent")
    
    # Record baseline
    print("\n--- FTG Baseline ---")
    frames_ftg, metrics_ftg = record_episode(ftg_get_action, agent_name="FTG Baseline")
    
    # Create comparison video
    print("\nCreating side-by-side comparison...")
    combined_frames = create_comparison_video(frames_ppo, frames_ftg)
    
    if output_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"comparison_{timestamp}.mp4"
    
    output_path = Path(OUTPUT_DIR) / output_name
    save_video(combined_frames, output_path)
    
    # Print comparison
    print("\n" + "=" * 60)
    print("RESULTS COMPARISON")
    print("=" * 60)
    print(f"{'Metric':<20} {'PPO Agent':<15} {'FTG Baseline':<15}")
    print("-" * 50)
    print(f"{'Survival Time':<20} {metrics_ppo['survival_time']:.2f}s{'':<9} {metrics_ftg['survival_time']:.2f}s")
    print(f"{'Total Steps':<20} {metrics_ppo['steps']:<15} {metrics_ftg['steps']:<15}")
    print(f"{'Total Reward':<20} {metrics_ppo['total_reward']:.1f}{'':<10} {metrics_ftg['total_reward']:.1f}")
    print("=" * 60)
    
    return metrics_ppo, metrics_ftg


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render F1TENTH agent videos (headless compatible)")
    parser.add_argument('--baseline', action='store_true', help='Render FTG baseline only')
    parser.add_argument('--compare', action='store_true', help='Render side-by-side comparison')
    parser.add_argument('--model', type=str, default=MODEL_PATH, help='Path to trained model')
    parser.add_argument('--output', type=str, default=None, help='Output filename')
    
    args = parser.parse_args()
    
    try:
        if args.compare:
            render_comparison(model_path=args.model, output_name=args.output)
        elif args.baseline:
            render_baseline(output_name=args.output)
        else:
            render_ppo_agent(model_path=args.model, output_name=args.output)
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
