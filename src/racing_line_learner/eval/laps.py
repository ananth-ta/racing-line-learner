"""Lap-time evaluation protocol.

All modes use a deterministic policy and start points derived from track
geometry only (never chosen from model results):

``flying`` (headline)
    Like qualifying: start on the grid, drive one untimed warm-up lap, then
    time lap 2 from start/finish line to start/finish line. The car crosses
    the line in its mid-race state, so no out-lap start point is involved.
``outlap`` (cheap stand-in for ``flying``)
    Start a fixed distance before the line (:meth:`Track.outlap_start_waypoint`,
    moved to a corner exit if it falls mid-corner) and time one lap. Valid for
    a track when its line-crossing speed matches the two-lap protocol's
    (compare ``line_speed_mean`` of the two modes).
``standing`` (diagnostic)
    From the grid, clock running from rest, one lap.
``robustness`` (diagnostic)
    One lap from each of ``n_starts`` points spread around the track, timed
    back to the start point. Measures reliability, not pace.

The simulator and a deterministic policy are exactly repeatable, so flying,
out-lap and standing laps are run ``repeats`` times from start poses with small
Gaussian offsets (``jitter_xy`` metres, ``jitter_theta`` radians) to get a mean
and spread.
"""

from dataclasses import replace
from typing import Any, Dict, List, Sequence

import numpy as np

from ..config import EnvConfig

MODES = ("flying", "outlap", "standing", "robustness")
DEFAULT_MAX_SIM_STEPS = 30000   # 300 s: two laps on every registered track


def _r(x, nd=2):
    return None if x is None else round(float(x), nd)


def drive(model, env, pose=None, spawn_idx=None, max_policy_steps: int = 40000) -> Dict[str, Any]:
    """One deterministic episode: how it ended, every lap time, and the speed at each line crossing."""
    obs = env.reset(pose=pose) if pose is not None else env.reset(spawn_idx=spawn_idx)
    timer = getattr(env, "_lap_timer", None)
    line_timing = env.config.lap_timing == "line"
    info: Dict[str, Any] = {}
    speeds: List[float] = []
    laps: List[float] = []
    line_speeds: List[float] = []
    was_running = bool(timer.running) if line_timing else True
    for _ in range(max_policy_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, info = env.step(action)
        speeds.append(info.get("speed", 0.0))
        if "lap_time" in info:
            laps.append(info["lap_time"])
            line_speeds.append(info.get("speed", 0.0))
        elif line_timing and timer.running and not was_running:   # clock started at the line (out-lap)
            line_speeds.append(info.get("speed", 0.0))
        if line_timing:
            was_running = bool(timer.running)
        if done:
            break
    return {
        "termination": info.get("termination"),
        "lap_times": [_r(t, 3) for t in laps],
        "line_speeds": [_r(v) for v in line_speeds],
        "waypoint": info.get("current_waypoint_idx"),
        "mean_speed": _r(np.mean(speeds)) if speeds else None,
        "max_speed": _r(np.max(speeds)) if speeds else None,
    }


def _timed(run: Dict[str, Any], mode: str):
    """The lap that counts for a mode, and the speed when its clock started (None if not completed)."""
    if run["termination"] != "lap_complete":
        return None, None
    if mode == "flying":   # lap 2; its clock started as lap 1 finished
        return (run["lap_times"][1], run["line_speeds"][0]) if len(run["lap_times"]) >= 2 else (None, None)
    if mode == "outlap":
        return run["lap_times"][0], (run["line_speeds"][0] if run["line_speeds"] else None)
    return run["lap_times"][0], None


def _summary(runs: Sequence[Dict[str, Any]], mode: str) -> Dict[str, Any]:
    timed = [_timed(r, mode) for r in runs]
    laps = [t for t, _ in timed if t is not None]
    line_speeds = [v for t, v in timed if t is not None and v is not None]
    fails: Dict[int, int] = {}
    for r, (t, _) in zip(runs, timed):
        if t is None:
            fails[r["waypoint"]] = fails.get(r["waypoint"], 0) + 1
    return {
        "runs": len(runs),
        "completed": len(laps),
        "completion_rate": _r(len(laps) / len(runs)) if runs else None,
        "lap_time_mean": _r(np.mean(laps), 3) if laps else None,
        "lap_time_std": _r(np.std(laps), 3) if len(laps) > 1 else None,
        "lap_time_best": _r(min(laps), 3) if laps else None,
        "line_speed_mean": _r(np.mean(line_speeds)) if line_speeds else None,
        "failures_at": [{"waypoint": w, "count": n} for w, n in sorted(fails.items(), key=lambda kv: -kv[1])],
    }


def eval_config(config: EnvConfig, lap_timing: str, laps: int = 1) -> EnvConfig:
    """The training config with curricula off, ``laps``-lap episodes and the given lap timing."""
    if config.reward_mode != "laptime":
        lap_timing = "spawn"   # line timing needs the racing-line projection (laptime mode)
    return replace(config, spawn_random=0.0, spawn_mix=0.0, spawn_waypoints=(), max_laps=laps,
                   lap_timing=lap_timing, max_episode_steps=DEFAULT_MAX_SIM_STEPS)


def evaluate(model, track, config: EnvConfig, modes: Sequence[str] = MODES, repeats: int = 5,
             n_starts: int = 16, jitter_xy: float = 0.05, jitter_theta: float = 0.02,
             seed: int = 0) -> Dict[str, Any]:
    """Run the requested modes on one track; returns ``{mode: summary with per-run results}``."""
    from ..env.tracks import Track, load_track
    from ..env.wrapper import make_env

    track = track if isinstance(track, Track) else load_track(track)
    out: Dict[str, Any] = {"track": track.name}
    for mode in modes:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
        if mode == "outlap" and config.reward_mode != "laptime":
            out[mode] = {"skipped": "out-laps need lap_timing='line' (laptime-mode models only)"}
            continue
        rng = np.random.RandomState(seed)   # same offsets for every mode and model

        def jittered(pose):
            x, y, th = pose
            return (x + rng.normal(0, jitter_xy), y + rng.normal(0, jitter_xy), th + rng.normal(0, jitter_theta))

        timing = "spawn" if mode == "robustness" else "line"
        env = make_env(track, eval_config(config, timing, laps=2 if mode == "flying" else 1), seed=seed)
        try:
            if mode == "robustness":
                n = track.n_waypoints
                starts = [None] + [int(i) for i in np.linspace(0, n, n_starts, endpoint=False)[1:]]
                runs = [dict(drive(model, env, spawn_idx=s), start=s) for s in starts]
            else:
                base = track.spawn_pose(track.outlap_start_waypoint()) if mode == "outlap" else track.start_pose
                runs = [dict(drive(model, env, pose=jittered(base) if k else base), repeat=k) for k in range(repeats)]
        finally:
            env.close()
        summary = _summary(runs, mode)
        if mode == "outlap":
            summary["start_waypoint"] = track.outlap_start_waypoint()
        summary["per_run"] = runs
        out[mode] = summary
    return out
