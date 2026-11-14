"""Lap-time evaluation: flying laps, standing starts and start-point robustness.

Three modes, all with a deterministic policy:

``flying`` (headline)
    Like qualifying: spawn about a tenth of a lap (at least 50 m) before the
    start/finish line (:meth:`Track.flying_start_waypoint`), build speed on
    the out-lap, time one lap from line to line.
``standing``
    From the track's start pose, clock running from rest, one lap to the line.
``robustness``
    One lap from each of ``n_starts`` points spread around the track (timed
    back to the start point). A reliability diagnostic, not a lap-time metric.

The simulator and a deterministic policy are exactly repeatable, so flying and
standing laps are run ``repeats`` times from start poses perturbed by small
Gaussian offsets (``jitter_xy`` metres, ``jitter_theta`` radians) to get a
mean and spread.
"""

from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import EnvConfig

MODES = ("flying", "standing", "robustness")
DEFAULT_MAX_SIM_STEPS = 30000   # 300 s: generous for one lap plus an out-lap


def _r(x, nd=2):
    return None if x is None else round(float(x), nd)


def drive(model, env, pose=None, spawn_idx=None, max_policy_steps: int = 20000) -> Dict[str, Any]:
    """One deterministic episode; returns how it ended and the lap time if any."""
    obs = env.reset(pose=pose) if pose is not None else env.reset(spawn_idx=spawn_idx)
    info: Dict[str, Any] = {}
    speeds: List[float] = []
    for _ in range(max_policy_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        speeds.append(info.get("speed", 0.0))
        if done:
            break
    return {
        "termination": info.get("termination"),
        "lap_time": _r(info.get("lap_time"), 3),
        "waypoint": info.get("current_waypoint_idx"),
        "mean_speed": _r(np.mean(speeds)) if speeds else None,
        "max_speed": _r(np.max(speeds)) if speeds else None,
    }


def _summary(runs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    laps = [r["lap_time"] for r in runs if r["termination"] == "lap_complete" and r["lap_time"] is not None]
    fails: Dict[int, int] = {}
    for r in runs:
        if r["termination"] != "lap_complete":
            fails[r["waypoint"]] = fails.get(r["waypoint"], 0) + 1
    return {
        "runs": len(runs),
        "completed": len(laps),
        "completion_rate": _r(len(laps) / len(runs)) if runs else None,
        "lap_time_mean": _r(np.mean(laps), 3) if laps else None,
        "lap_time_std": _r(np.std(laps), 3) if len(laps) > 1 else None,
        "lap_time_best": _r(min(laps), 3) if laps else None,
        "failures_at": [{"waypoint": w, "count": n} for w, n in sorted(fails.items(), key=lambda kv: -kv[1])],
    }


def eval_config(config: EnvConfig, lap_timing: str) -> EnvConfig:
    """The training config with curricula off, one-lap episodes and the given lap timing."""
    if config.reward_mode != "laptime":
        lap_timing = "spawn"   # line timing needs the racing-line projection (laptime mode)
    return replace(config, spawn_random=0.0, spawn_mix=0.0, spawn_waypoints=(), max_laps=1,
                   lap_timing=lap_timing, max_episode_steps=DEFAULT_MAX_SIM_STEPS)


def evaluate(model, track, config: EnvConfig, modes: Sequence[str] = MODES, repeats: int = 5,
             n_starts: int = 16, jitter_xy: float = 0.05, jitter_theta: float = 0.02,
             seed: int = 0) -> Dict[str, Any]:
    """Run the requested modes on one track; returns ``{mode: summary with per-run results}``."""
    from ..env.tracks import Track, load_track
    from ..env.wrapper import make_env

    track = track if isinstance(track, Track) else load_track(track)
    rng = np.random.RandomState(seed)

    def jittered(pose):
        x, y, th = pose
        return (x + rng.normal(0, jitter_xy), y + rng.normal(0, jitter_xy), th + rng.normal(0, jitter_theta))

    out: Dict[str, Any] = {"track": track.name}
    for mode in modes:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
        if mode == "flying" and config.reward_mode != "laptime":
            out[mode] = {"skipped": "flying laps need lap_timing='line' (laptime-mode models only)"}
            continue
        env = make_env(track, eval_config(config, "spawn" if mode == "robustness" else "line"), seed=seed)
        try:
            if mode == "robustness":
                n = track.n_waypoints
                starts = [None] + [int(i) for i in np.linspace(0, n, n_starts, endpoint=False)[1:]]
                runs = [dict(drive(model, env, spawn_idx=s), start=s) for s in starts]
            else:
                base = track.spawn_pose(track.flying_start_waypoint()) if mode == "flying" else track.start_pose
                runs = [dict(drive(model, env, pose=jittered(base) if k else base), repeat=k) for k in range(repeats)]
        finally:
            env.close()
        summary = _summary(runs)
        if mode == "flying":
            summary["start_waypoint"] = track.flying_start_waypoint()
        summary["per_run"] = runs
        out[mode] = summary
    return out
