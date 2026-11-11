"""Step 1: record laps of a trained PPO model and save per-step telemetry.

The simulator and model.predict(deterministic=True) are both deterministic, so
repeating the same lap gives identical data. To get real variance:
  * lap 0 is the plain baseline (standard start pose, deterministic actions),
  * every later lap starts from a slightly nudged pose,
  * every second later lap also samples actions stochastically.

Usage (from the repo root):
    python -m src.post_loop.record --model models/ppo/f1tenth_ppo_v6.zip --track cota
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import gym
import numpy as np

from src.post_loop.tracks import REPO_ROOT, get_track, load_waypoints

# wrapper.py and render_ppo_lap.py live at the repo root.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from wrapper import F1TenthWrapper  # noqa: E402

SIM_DT = 0.01                # seconds per simulator step
STAGNATION_STEPS = 2000      # wrapper ends the episode after this many steps with no waypoint
SPEED_FLOOR, SPEED_CAP = 6.5, 20.0   # clip applied inside wrapper.step()
DEFAULT_POSE_NOISE = (0.3, 0.3, 0.05)  # max nudge in x (m), y (m), theta (rad)


def make_env(track, waypoints):
    raw_env = gym.make("f110_gym:f110-v0", map=str(REPO_ROOT / track.map_path),
                       map_ext=track.map_ext, num_agents=1)
    env = F1TenthWrapper(raw_env, algo="ppo")
    # wrapper.py hard-codes the COTA CSV; use this track's waypoints instead.
    # It must stay a list because the wrapper checks `if self.waypoints:`.
    env.waypoints = [np.array(w) for w in waypoints]
    return raw_env, env


def lap_plan(n_laps, rng, base_pose, pose_noise):
    """Yield (mode, start_pose, deterministic) for each lap."""
    for i in range(n_laps):
        if i == 0:
            yield "baseline", tuple(base_pose), True
            continue
        nudge = rng.uniform(-1.0, 1.0, size=3) * np.array(pose_noise)
        pose = tuple(float(v) for v in np.array(base_pose) + nudge)
        stochastic = (i % 2 == 0)
        yield ("stochastic" if stochastic else "perturbed"), pose, not stochastic


def run_lap(model, raw_env, env, start_pose, deterministic, max_steps):
    """Drive one episode and return (telemetry dict of arrays, summary dict)."""
    sim = raw_env.unwrapped
    n_wp = len(env.waypoints)
    wrapped_obs = env.reset(poses=np.array([list(start_pose)]))

    rows = {k: [] for k in (
        "x", "y", "theta", "speed", "yaw_rate",
        "steer_raw", "steer_smoothed", "speed_cmd_raw", "speed_cmd_applied",
        "target_wp", "wp_after", "wp_cleared", "reward", "lidar20",
        "scan_front", "scan_min")}

    total_reward, cleared, termination = 0.0, 0, "timeout"
    for step in range(max_steps):
        # State the car is in when it chooses this step's action. Logging the
        # pre-step state keeps the real crash speed: the simulator zeroes the
        # velocity once it detects a collision.
        raw = sim.current_obs
        scan = np.asarray(raw["scans"][0])
        target_wp = env.current_waypoint_idx

        action, _ = model.predict(wrapped_obs, deterministic=deterministic)
        action = np.array(action, dtype=np.float64).flatten()
        raw_action = action.copy()   # wrapper.step() overwrites action[0] in place

        wrapped_obs, reward, done, info = env.step(action)
        total_reward += reward
        cleared += int(info["waypoint_cleared"])

        rows["x"].append(raw["poses_x"][0])
        rows["y"].append(raw["poses_y"][0])
        rows["theta"].append(raw["poses_theta"][0])
        rows["speed"].append(raw["linear_vels_x"][0])
        rows["yaw_rate"].append(raw["ang_vels_z"][0])
        rows["steer_raw"].append(raw_action[0])
        rows["steer_smoothed"].append(env.actual_steering)
        rows["speed_cmd_raw"].append(raw_action[1])
        rows["speed_cmd_applied"].append(np.clip(raw_action[1], SPEED_FLOOR, SPEED_CAP))
        rows["target_wp"].append(target_wp)
        rows["wp_after"].append(info["current_waypoint_idx"])
        rows["wp_cleared"].append(cleared)
        rows["reward"].append(reward)
        rows["lidar20"].append(env.process_obs(raw)[:env.lidar_rays])
        rows["scan_front"].append(scan[len(scan) // 2])
        rows["scan_min"].append(scan.min())

        if done:
            # The wrapper sets done for both crashes and stagnation, and info
            # does not say which, so check the simulator and the wrapper counter.
            if sim.collisions[0] > 0:
                termination = "crash"
            elif env.ppo_time_since_last_cp > STAGNATION_STEPS:
                termination = "stuck"
            else:
                termination = "sim_done"
            break
        # One full lap = the last waypoint is cleared and the target wraps to 0.
        if info["waypoint_cleared"] and info["current_waypoint_idx"] == 0 and cleared > 1:
            termination = "finished"
            break

    tel = {k: np.asarray(v) for k, v in rows.items()}
    n = len(tel["x"])
    summary = {
        "termination": termination,
        "steps": n,
        "sim_time_s": round(n * SIM_DT, 2),
        "waypoints_cleared": cleared,
        "num_waypoints": n_wp,
        "last_target_wp": int(tel["target_wp"][-1]),
        # Last pre-step pose: on a crash the simulator zeroes speed AND heading.
        "end_pose": [float(tel["x"][-1]), float(tel["y"][-1]), float(tel["theta"][-1])],
        "total_reward": round(float(total_reward), 2),
        "mean_speed": round(float(tel["speed"].mean()), 3),
        "max_speed": round(float(tel["speed"].max()), 3),
        "speed_at_end": round(float(tel["speed"][-1]), 3),
    }
    return tel, summary


def save_plot(track, tel, summary, path, title):
    from render_ppo_lap import HeadlessRenderer  # reuse the team's renderer
    renderer = HeadlessRenderer(map_path=str(REPO_ROOT / track.map_path), map_ext=track.map_ext)
    renderer.all_trajectory = [renderer.world_to_pixel(x, y) for x, y in zip(tel["x"], tel["y"])]
    color = {"finished": "lime", "crash": "red"}.get(summary["termination"], "orange")
    stats = [
        ("Result", summary["termination"], color),
        ("Steps", str(summary["steps"]), "white"),
        ("Time", f"{summary['sim_time_s']:.2f}s", "white"),
        ("Waypoints", f"{summary['waypoints_cleared']}/{summary['num_waypoints']}", "white"),
        ("Mean speed", f"{summary['mean_speed']:.2f} m/s", "white"),
    ]
    renderer.save_path_image(str(path), title, stats=stats)


def record_laps(model_path, track_name="cota", n_laps=10, seed=0, max_steps=10000,
                pose_noise=DEFAULT_POSE_NOISE, out_root=None, plot=False):
    from stable_baselines3 import PPO
    from stable_baselines3.common.utils import set_random_seed

    track = get_track(track_name)
    waypoints = load_waypoints(track)
    model_path = Path(model_path)
    model = PPO.load(str(model_path), device="cpu")
    raw_env, env = make_env(track, waypoints)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_root = Path(out_root) if out_root else REPO_ROOT / "post_loop_runs"
    out_dir = out_root / f"{model_path.stem}__{track.name}__{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)

    rng = np.random.default_rng(seed)
    laps = []
    for i, (mode, pose, deterministic) in enumerate(lap_plan(n_laps, rng, track.start_pose, pose_noise)):
        set_random_seed(seed + i)   # makes stochastic laps reproducible
        tel, summary = run_lap(model, raw_env, env, pose, deterministic, max_steps)
        summary = {"lap": i, "mode": mode, "start_pose": [round(v, 4) for v in pose], **summary}
        np.savez_compressed(out_dir / f"lap_{i:03d}.npz", **tel)
        if plot:
            save_plot(track, tel, summary, out_dir / f"lap_{i:03d}.png",
                      f"{model_path.stem} lap {i} ({mode})")
        laps.append(summary)
        print(f"  lap {i:2d} {mode:10s} -> {summary['termination']:8s} "
              f"steps={summary['steps']:5d} wp={summary['waypoints_cleared']:3d}/{len(waypoints)} "
              f"mean_v={summary['mean_speed']:.2f}")
    env.close()

    meta = {
        "model": str(model_path),
        "track": track.name,
        "seed": seed,
        "n_laps": n_laps,
        "max_steps": max_steps,
        "pose_noise": list(pose_noise),
        "created": stamp,
        "laps": laps,
    }
    (out_dir / "laps.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {n_laps} laps to {out_dir}")
    return out_dir


def main():
    p = argparse.ArgumentParser(description="Record laps of a trained PPO model (post-loop step 1)")
    p.add_argument("--model", required=True, help="path to a PPO .zip checkpoint")
    p.add_argument("--track", default="cota")
    p.add_argument("--laps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--out", default=None, help="output folder (default: post_loop_runs/)")
    p.add_argument("--plot", action="store_true", help="also save a trajectory image per lap")
    a = p.parse_args()
    record_laps(a.model, a.track, a.laps, a.seed, a.max_steps, out_root=a.out, plot=a.plot)


if __name__ == "__main__":
    main()
