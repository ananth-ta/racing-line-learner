"""
render_ga_lap.py - Render a full lap video of a trained F1TENTH GA mutant.
Outputs a 1920x1080 video, stops exactly when completing one lap using geometric detection.
"""

import gym
import numpy as np
import cv2
import argparse
import os
import torch
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from wrapper import F1TenthWrapper
from ga_model import MutantMLP

# ============================================================================
# CONFIGURATION
# ============================================================================

MAP_NAME = 'maps/vegas'
MAX_STEPS = 10000         # Max steps per episode
FPS = 30                  # Output video framerate

# ============================================================================
# CUSTOM RENDERER (No Display Required)
# ============================================================================

class HeadlessRenderer:
    def __init__(self, map_path='maps/vegas', map_ext='.png', figsize=(19.2, 10.8)):
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
        
        # Load map metadata
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
        
        self.trajectory = []
        self.all_trajectory = []
        self.max_trajectory_points = 500
    
    def world_to_pixel(self, x, y):
        px = int((x - self.origin[0]) / self.resolution)
        py = int(self.map_img.shape[0] - (y - self.origin[1]) / self.resolution)
        return px, py
    
    def reset(self):
        self.trajectory = []
        self.all_trajectory = []
    
    def render(self, obs_dict, action, step, total_reward, agent_name="Agent", fitness_breakdown=None):
        fig, axes = plt.subplots(1, 2, figsize=self.figsize, dpi=100)
        
        ax1 = axes[0]
        ax1.imshow(self.map_img, cmap='gray', origin='upper')
        
        car_x = obs_dict['poses_x'][0]
        car_y = obs_dict['poses_y'][0]
        car_theta = obs_dict['poses_theta'][0]
        velocity = obs_dict['linear_vels_x'][0]
        
        px, py = self.world_to_pixel(car_x, car_y)
        
        self.trajectory.append((px, py))
        self.all_trajectory.append((px, py))
        if len(self.trajectory) > self.max_trajectory_points:
            self.trajectory.pop(0)
        
        if len(self.trajectory) > 1:
            traj = np.array(self.trajectory)
            ax1.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=3, alpha=0.5)
        
        lidar_scans = obs_dict['scans'][0]
        num_rays = len(lidar_scans)
        angle_min, angle_max = -2.35, 2.35
        angles = np.linspace(angle_min, angle_max, num_rays)
        
        for i in range(0, num_rays, 54):
            ray_angle = car_theta + angles[i]
            ray_dist = min(lidar_scans[i], 10.0)
            end_x = car_x + ray_dist * np.cos(ray_angle)
            end_y = car_y + ray_dist * np.sin(ray_angle)
            end_px, end_py = self.world_to_pixel(end_x, end_y)
            color = plt.cm.RdYlGn(min(ray_dist / 10.0, 1.0))
            ax1.plot([px, end_px], [py, end_py], color=color, linewidth=2.0, alpha=0.7)
        
        car_size = 25
        dx = car_size * np.cos(-car_theta + np.pi/2)
        dy = car_size * np.sin(-car_theta + np.pi/2)
        ax1.scatter(px, py, c='red', s=250, zorder=10, marker='o')
        ax1.arrow(px, py, dx, dy, head_width=12, head_length=10, fc='red', ec='red', zorder=11)
        
        view_radius = 250
        ax1.set_xlim(px - view_radius, px + view_radius)
        ax1.set_ylim(py + view_radius, py - view_radius)
        ax1.set_aspect('equal')
        ax1.set_title(f'{agent_name} - 1080p High-Fidelity Render', fontsize=24, pad=20)
        ax1.axis('off')
        
        ax2 = axes[1]
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.axis('off')
        
        ax2.add_patch(Rectangle((0, 0), 10, 10, facecolor='#1a1a2e', edgecolor='none'))
        ax2.text(5, 9.2, agent_name, fontsize=32, ha='center', color='cyan', fontweight='bold')
        
        steering_deg = np.degrees(action[0])
        speed_cmd = action[1]
        
        telemetry = [
            ('Step', f'{step}', 'white'),
            ('Time', f'{step * 0.01:.2f}s', 'white'),
            ('Reward', f'{total_reward:.1f}', 'lime' if total_reward > 0 else 'red'),
        ]
        
        if fitness_breakdown:
            telemetry.extend([
                ('Prog Score', f"+{fitness_breakdown.get('progress_score', 0.0):.1f}", 'lime'),
                ('Wypt Bonus', f"+{fitness_breakdown.get('waypoint_bonus', 0.0):.1f}", 'lime'),
                ('Dir Pnlty', f"-{fitness_breakdown.get('steering_penalty', 0.0):.1f}", 'red'),
                ('Jitter Tax', f"-{fitness_breakdown.get('jitter_tax', 0.0):.1f}", 'red'),
            ])

        telemetry.extend([
            ('', '', 'white'),
            ('Velocity', f'{velocity:.2f} m/s', 'cyan'),
            ('Speed Cmd', f'{speed_cmd:.2f} m/s', 'yellow'),
            ('Steering', f'{steering_deg:.1f}°', 'orange'),
            ('', '', 'white'),
            ('Position', f'({car_x:.1f}, {car_y:.1f})', 'white'),
            ('Heading', f'{np.degrees(car_theta):.1f}°', 'white'),
        ])
        
        y_pos = 8.0
        for label, value, color in telemetry:
            if label:
                ax2.text(1, y_pos, f'{label}:', fontsize=20, color='gray', ha='left')
                ax2.text(9, y_pos, value, fontsize=20, color=color, ha='right', fontweight='bold')
            y_pos -= 0.6
        
        ax2.text(5, 2.5, 'Steering', fontsize=20, ha='center', color='gray')
        steer_normalized = np.clip(steering_deg / 45.0, -1, 1)
        arc_color = 'orange'
        ax2.plot([5, 5], [1.0, 2.0], 'gray', linewidth=4)
        steer_x = 5 + steer_normalized * 3
        ax2.plot([5, steer_x], [1.5, 1.5], color=arc_color, linewidth=8)
        ax2.scatter([steer_x], [1.5], c=arc_color, s=300, zorder=10)
        
        ax2.text(5, 0.8, 'Speed', fontsize=20, ha='center', color='gray')
        speed_normalized = np.clip(velocity / 10.0, 0, 1)
        ax2.add_patch(Rectangle((2, 0.2), 6, 0.4, facecolor='#333', edgecolor='gray'))
        ax2.add_patch(Rectangle((2, 0.2), 6 * speed_normalized, 0.4, facecolor='cyan', edgecolor='none'))
        
        plt.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        
        return img
        
    def save_path_image(self, output_path, agent_name="Agent", stats=None):
        if not self.map_loaded:
            print("Cannot save trajectory map because the map image was not loaded.")
            return
            
        fig, axes = plt.subplots(1, 2, figsize=self.figsize, dpi=100)
        
        ax1 = axes[0]
        ax1.imshow(self.map_img, cmap='gray', origin='upper')
        
        if len(self.all_trajectory) > 1:
            traj = np.array(self.all_trajectory)
            ax1.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=2, alpha=0.8)
            ax1.scatter(traj[0, 0], traj[0, 1], c='lime', s=100, label='Start', zorder=10)
            ax1.scatter(traj[-1, 0], traj[-1, 1], c='red', s=100, label='End', zorder=10)
            ax1.legend(loc='lower left', fontsize=16)
            
        ax1.set_aspect('equal')
        ax1.set_title(f'{agent_name} - Final Trajectory', fontsize=24, pad=20)
        ax1.axis('off')
        
        ax2 = axes[1]
        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.axis('off')
        
        ax2.add_patch(Rectangle((0, 0), 10, 10, facecolor='#1a1a2e', edgecolor='none'))
        ax2.text(5, 9.2, agent_name, fontsize=32, ha='center', color='cyan', fontweight='bold')
        ax2.text(5, 8.5, 'Run Summary', fontsize=24, ha='center', color='white', fontweight='bold')
        ax2.plot([1, 9], [8.0, 8.0], 'gray', linewidth=2)
        
        if stats:
            y_pos = 7.0
            for label, value, color in stats:
                if label:
                    ax2.text(1, y_pos, f'{label}:', fontsize=20, color='gray', ha='left')
                    ax2.text(9, y_pos, value, fontsize=20, color=color, ha='right', fontweight='bold')
                y_pos -= 0.6
        
        plt.tight_layout()
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)

# ============================================================================
# LAP RECORDING LOGIC
# ============================================================================

def create_env():
    env = gym.make('f110_gym:f110-v0', map=MAP_NAME, map_ext='.png', num_agents=1)
    return env

def record_lap(model_path, output_path):
    raw_env = create_env()
    wrapped_env = F1TenthWrapper(raw_env)
    
    print(f"Loading GA mutant model: {model_path}")
    model = MutantMLP()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    renderer = HeadlessRenderer(map_path=MAP_NAME)
    renderer.reset()
    
    Path(output_path).parent.mkdir(exist_ok=True)
    writer = None
    
    wrapped_obs = wrapped_env.reset()
    
    total_reward = 0.0
    fitness_breakdown = {}
    start_pos = None
    left_start_zone = False
    termination_reason = "Time Limit"
    
    print(f"Recording Cinematic Lap (Max {MAX_STEPS} steps)...")
    
    # Calculate penalty weight based on generation number to match ga_train.py 
    try:
        chkpt_str = model_path.split('_gen_')[-1].split('.pt')[0] if '_gen_' in model_path else '500'
        current_gen = int(chkpt_str)
    except ValueError:
        current_gen = 500
    cycle_length = 50
    cycle_pos = current_gen % cycle_length
    penalty_weight = 2.0 * (max(0, cycle_pos - 1) / max(1, cycle_length - 1))
    
    for step in range(MAX_STEPS):
        obs_tensor = torch.FloatTensor(wrapped_obs).unsqueeze(0)
        with torch.no_grad():
            action = model(obs_tensor).squeeze(0).numpy()
            
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
            current_obs = wrapped_obs[-64:]  
            raw_obs = {
                'poses_x': [0.0],
                'poses_y': [0.0],
                'poses_theta': [0.0],
                'linear_vels_x': [current_obs[-4]],
                'linear_vels_y': [0.0],
                'ang_vels_z': [current_obs[-3]],
                'scans': [np.array(list(current_obs[:-4]) * 18)],
            }
            if step == 0:
                print(f"  Warning: Using fallback observation extraction: {e}")
        
        car_x = raw_obs['poses_x'][0]
        car_y = raw_obs['poses_y'][0]
        current_pos = np.array([car_x, car_y])
        
        if step == 0:
            start_pos = current_pos
        else:
            dist = np.linalg.norm(current_pos - start_pos)
            if dist > 10.0:
                left_start_zone = True
            elif left_start_zone and dist < 2.0:
                print(f"  Lap completed at step {step}!")
                termination_reason = "Lap Completed"
                break
        
        try:
            # Get checkpoint info from filename if possible
            chkpt_num = model_path.split('_gen_')[-1].split('.pt')[0] if '_gen_' in model_path else 'Final'
            frame = renderer.render(raw_obs, action, step, total_reward, f"GA Mutant Gen {chkpt_num}", fitness_breakdown)
            
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(output_path), fourcc, FPS, (w, h))
            
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr_frame)
        except Exception as e:
            print(f"  Render error at step {step}: {e}")
            break
            
        wrapped_obs, reward, done, info = wrapped_env.step(action)
        
        # Calculate full GA Fitness logic to match ga_train.py
        progress_score = info.get('progress_reward', 0.0)
        waypoint_bonus = 500.0 if info.get('waypoint_cleared', False) else 0.0
        steering_penalty = (abs(action[0]) ** 2) * penalty_weight
        jitter_tax = info.get('steering_delta', 0.0) * 50.0
        
        fitness_breakdown = {
            'progress_score': progress_score,
            'waypoint_bonus': waypoint_bonus,
            'steering_penalty': steering_penalty,
            'jitter_tax': jitter_tax
        }
        
        step_fitness = reward + progress_score + waypoint_bonus - steering_penalty - jitter_tax
        total_reward += step_fitness
        
        if done:
            print(f"  Episode ended at step {step} (collision)")
            termination_reason = "Collision"
            break
            
        if step % 200 == 0 and step > 0:
            print(f"  Step {step}/{MAX_STEPS}, Reward: {total_reward:.1f}")
            
    print(f"  Finished recording: {step} steps, {(step) * 0.01:.2f}s, Reward: {total_reward:.1f}")
    wrapped_env.close()
    
    if writer is not None:
        writer.release()
        print(f"Saved video: {output_path} ({step} frames, {step/FPS:.1f}s)")
        
        # Save the map trace
        image_output_path = str(Path(output_path).with_suffix('.png'))
        chkpt_num = model_path.split('_gen_')[-1].split('.pt')[0] if '_gen_' in model_path else 'Final'
        
        status_color = 'lime' if termination_reason == "Lap Completed" else ('red' if termination_reason == "Collision" else 'orange')
        
        stats = [
            ('Result', termination_reason, status_color),
            ('', '', 'white'),
            ('Total Steps', f'{step}', 'white'),
            ('Final Time', f'{step * 0.01:.2f}s', 'white'),
            ('Final Reward', f'{total_reward:.1f}', 'lime' if total_reward > 0 else 'red'),
        ]
        
        renderer.save_path_image(image_output_path, f"GA Mutant Gen {chkpt_num}", stats=stats)
        print(f"Saved trajectory map: {image_output_path}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a full lap video of a trained GA mutant")
    parser.add_argument('--model', type=str, required=True, help='Path to trained model .pt checkpoint')
    parser.add_argument('--output', type=str, required=True, help='Output filename for the video (.mp4)')
    args = parser.parse_args()
    
    try:
        record_lap(args.model, args.output)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
