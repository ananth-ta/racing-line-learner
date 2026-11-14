"""Training telemetry: per-step ``info`` dicts -> a compact, deterministic window report.

The in-loop supervisor never reads raw logs. It reads the report built
here, which is small enough to fit in an LLM prompt and contains only
facts computed by code:

* how episodes ended, and where (a crash heatmap by waypoint);
* crash hotspots: clusters of collisions, with the car's approach speed and
  the corner's approximate grip limit;
* per track segment: time share (a proxy for sector time), speed against
  the grip limit, steering saturation, collisions;
* how often the speed command is pinned at the floor or at the maximum;
* the reward broken into its terms, per step;
* optionally a deterministic evaluation probe (lap completion and lap time
  from fixed start points).

"Waypoint" always means the target waypoint (``info['current_waypoint_idx']``):
the car is between waypoint ``i - 1`` and ``i``.
"""

from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from ..config import SIM_STEER_LIMIT, EnvConfig
from ..env.tracks import Track

#: EnvConfig fields copied into the report, so the reader knows the current settings.
REPORTED_CONFIG = (
    "reward_mode", "speed_floor", "speed_max", "target_speed", "progress_coef", "waypoint_bonus",
    "steering_penalty_w", "jitter_coef", "time_cost", "lap_bonus", "terminal_penalty", "spawn_waypoints", "spawn_mix",
)


def _r(x: Optional[float], nd: int = 2) -> Optional[float]:
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


class TelemetryCollector:
    """Accumulates per-step training information for one supervision window.

    Call :meth:`update` once per environment step (per env index when
    vectorised) and :meth:`report` at the end of the window, then
    :meth:`reset`.
    """

    def __init__(self, track: Track, env_config: EnvConfig, n_segments: int = 15,
                 max_hotspots: int = 5, saturation_threshold: float = 0.98, pinned_tolerance: float = 0.05):
        self.track = track
        self.env_config = env_config
        self.n = track.n_waypoints
        self.grip = track.grip_speed_limit(v_cap=env_config.speed_max)
        self.segments = track.segment_ranges(n_segments)
        self._seg_of = np.zeros(self.n, dtype=int)
        for s, (a, b) in enumerate(self.segments):
            self._seg_of[a:b] = s
        # Saturation is measured against the simulator's physical limit: legacy
        # commands beyond +/-0.4189 rad do nothing extra.
        self.steer_limit = min(env_config.steer_limit, SIM_STEER_LIMIT)
        self.max_hotspots = max_hotspots
        self.saturation_threshold = saturation_threshold
        self.pinned_tolerance = pinned_tolerance
        self.reset()

    def reset(self) -> None:
        n, m = self.n, len(self.segments)
        self.steps = 0
        self.reward_sum = 0.0
        self.reward_terms: Dict[str, float] = defaultdict(float)
        self.terminations: Counter = Counter()
        self.term_at: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(self.n, dtype=int))
        self.wp_speed_sum = np.zeros(n)
        self.wp_steps = np.zeros(n, dtype=int)
        self.seg_steps = np.zeros(m, dtype=int)
        self.seg_speed_max = np.zeros(m)
        self.seg_saturated = np.zeros(m, dtype=int)
        self.pinned_floor = 0
        self.pinned_max = 0
        self.saturated = 0
        self.lap_times: List[float] = []
        self.episode_steps: List[int] = []
        self.episode_cleared: List[int] = []
        self._ep_steps: Dict[int, int] = defaultdict(int)
        self._ep_cleared: Dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------ #

    def update(self, info: Dict[str, Any], reward: float, env_index: int = 0) -> None:
        """Record one environment step."""
        c = self.env_config
        wp = int(info["current_waypoint_idx"]) % self.n
        loc = (wp - 1) % self.n  # the car is between loc and wp
        seg = self._seg_of[loc]
        speed = float(info.get("speed", 0.0))

        self.steps += 1
        self.reward_sum += float(reward)
        for k, v in info.get("reward_terms", {}).items():
            self.reward_terms[k] += float(v)

        self.wp_speed_sum[loc] += speed
        self.wp_steps[loc] += 1
        self.seg_steps[seg] += 1
        self.seg_speed_max[seg] = max(self.seg_speed_max[seg], speed)

        cmd = float(info.get("speed_command", np.nan))
        if abs(cmd - c.speed_floor) <= self.pinned_tolerance:
            self.pinned_floor += 1
        elif abs(cmd - c.speed_max) <= self.pinned_tolerance:
            self.pinned_max += 1
        if abs(float(info.get("raw_steering", 0.0))) >= self.saturation_threshold * self.steer_limit:
            self.saturated += 1
            self.seg_saturated[seg] += 1

        self._ep_steps[env_index] += 1
        if info.get("waypoint_cleared"):
            self._ep_cleared[env_index] += 1
        if "lap_time" in info:
            self.lap_times.append(float(info["lap_time"]))

        term = info.get("termination")
        if term:
            self.terminations[term] += 1
            self.term_at[term][wp] += 1
            self.episode_steps.append(self._ep_steps.pop(env_index, 0))
            self.episode_cleared.append(self._ep_cleared.pop(env_index, 0))

    # ------------------------------------------------------------------ #

    def _mean_speed_at(self, idx: Iterable[int]) -> Optional[float]:
        idx = [i % self.n for i in idx]
        steps = self.wp_steps[idx].sum()
        return float(self.wp_speed_sum[idx].sum() / steps) if steps else None

    def crash_hotspots(self) -> List[Dict[str, Any]]:
        """Clusters of collisions, largest first."""
        counts = self.term_at["collision"].copy() if "collision" in self.term_at else np.zeros(self.n, dtype=int)
        total = int(counts.sum())
        out = []
        threshold = max(2, int(0.05 * total))
        while len(out) < self.max_hotspots and total and counts.max() >= threshold:
            peak = int(counts.argmax())
            members = [(peak + o) % self.n for o in range(-3, 4) if counts[(peak + o) % self.n] > 0]
            offsets = sorted(((m - peak + self.n // 2) % self.n) - self.n // 2 for m in members)
            first, last = (peak + offsets[0]) % self.n, (peak + offsets[-1]) % self.n
            n_coll = int(sum(counts[m] for m in members))
            approach = [(first + o) % self.n for o in range(-3, offsets[-1] - offsets[0] + 1)]
            # Car location for target waypoint w is w-1; the corner it is entering spans first-1 .. last.
            corner = [(first - 1 + o) % self.n for o in range(offsets[-1] - offsets[0] + 2)]
            out.append({
                "waypoints": [first, last],
                "collisions": n_coll,
                "share": _r(n_coll / total),
                "approach_speed": _r(self._mean_speed_at(approach)),
                "grip_limit": _r(float(self.grip[corner].min())),
            })
            for m in members:
                counts[m] = 0
        # Clusters are found peak-first, but a broad cluster can outweigh a sharp one.
        return sorted(out, key=lambda h: -h["collisions"])

    def report(self, window: Optional[int] = None, timesteps: Optional[int] = None,
               probe: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """The window report: plain JSON-serialisable data, numbers rounded."""
        c = self.env_config
        steps = max(self.steps, 1)
        episodes = sum(self.terminations.values())
        dt = 0.01 * c.action_repeat
        terms = {k: {"count": v, "share": _r(v / episodes)} for k, v in sorted(self.terminations.items())}

        segments = []
        collisions = self.term_at["collision"] if "collision" in self.term_at else np.zeros(self.n, dtype=int)
        for s, (a, b) in enumerate(self.segments):
            locs = range(a, b)
            segments.append({
                "waypoints": [a, b - 1],
                "time_share": _r(self.seg_steps[s] / steps, 3),
                "mean_speed": _r(self._mean_speed_at(locs)),
                "max_speed": _r(self.seg_speed_max[s]) if self.seg_steps[s] else None,
                "grip_limit_min": _r(float(self.grip[a:b].min())),
                "collisions": int(collisions[[(i + 1) % self.n for i in locs]].sum()),
                "steer_saturation": _r(self.seg_saturated[s] / self.seg_steps[s]) if self.seg_steps[s] else None,
            })

        laps = len(self.lap_times)
        report = {
            "window": window,
            "timesteps": timesteps,
            "track": self.track.name,
            "n_waypoints": self.n,
            "steps": self.steps,
            "episodes": episodes,
            "terminations": terms,
            "laps": {
                "completed": laps,
                "completion_rate": _r(laps / episodes) if episodes else None,
                "lap_time_mean": _r(np.mean(self.lap_times)) if laps else None,
                "lap_time_best": _r(min(self.lap_times)) if laps else None,
            },
            "episode": {
                "mean_length_s": _r(np.mean(self.episode_steps) * dt) if self.episode_steps else None,
                "mean_waypoints_cleared": _r(np.mean(self.episode_cleared), 1) if self.episode_cleared else None,
            },
            "speed": {
                "mean": _r(self.wp_speed_sum.sum() / steps),
                "pinned_at_floor_share": _r(self.pinned_floor / steps),
                "pinned_at_max_share": _r(self.pinned_max / steps),
                "floor": c.speed_floor,
                "max": c.speed_max,
                "floor_limited_lap_time_s": _r(self.track.length / c.speed_floor) if c.speed_floor > 0 else None,
            },
            "steering": {"saturation_share": _r(self.saturated / steps)},
            "reward_per_step": {k: _r(v / steps, 4) for k, v in sorted(self.reward_terms.items())},
            "crash_hotspots": self.crash_hotspots(),
            "segments": segments,
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in
                       ((k, getattr(c, k)) for k in REPORTED_CONFIG)},
        }
        if probe is not None:
            report["probe"] = probe
        return report


def run_episodes(model, env, collector: Optional[TelemetryCollector], spawns: Sequence[Optional[int]],
                 deterministic: bool, max_steps: int = 20000) -> List[Dict[str, Any]]:
    """Drive one episode per spawn (``None`` = default start); optionally feed the collector.

    Returns one summary per episode: spawn, termination, lap time, final target waypoint.
    """
    results = []
    for spawn in spawns:
        obs = env.reset(spawn_idx=spawn) if spawn is not None else env.reset()
        info: Dict[str, Any] = {}
        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            if collector is not None:
                collector.update(info, reward)
            if done:
                break
        results.append({"spawn": spawn, "termination": info.get("termination"),
                        "lap_time": _r(info.get("lap_time")), "waypoint": info.get("current_waypoint_idx")})
    return results


def summarize_probe(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Probe summary for the report: completion rate, lap times, where the failed episodes ended."""
    laps = [r["lap_time"] for r in results if r["termination"] == "lap_complete" and r["lap_time"] is not None]
    fails = Counter(r["waypoint"] for r in results if r["termination"] != "lap_complete")
    return {
        "episodes": len(results),
        "completion_rate": _r(len(laps) / len(results)) if results else None,
        "lap_time_mean": _r(np.mean(laps)) if laps else None,
        "lap_time_best": _r(min(laps)) if laps else None,
        "failures_at": [{"waypoint": w, "count": n} for w, n in fails.most_common()],
    }
