"""Step 2: find where and why laps failed.

Reads a folder written by record.py and writes analysis.json next to it.
For every crash it works out:
  * where: the waypoint the car was heading to, and the nearest waypoint
  * how fast it was going, and how it was steering
  * how sharp that part of the track is (curvature from the waypoint line)
  * a visibility check: how far the car could see straight ahead (ray-cast
    on the map image) compared with the distance it needed to stop
Crashes are then grouped into track segments, and "stuck" laps are listed
separately because they need a different fix.

analysis.json is the ground truth the LLM's diagnosis gets checked against
(steps 3-4), so every number here comes straight from the recorded data.

Usage (from the repo root):
    python -m src.post_loop.analyze --run post_loop_runs/<folder>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from src.post_loop.tracks import get_track, load_waypoints

A_MAX = 9.51           # m/s^2, simulator's max deceleration (f110_env default params)
MAX_RANGE = 30.0       # m, how far the visibility ray looks (LiDAR max range)
LOOKBACK_S = 1.0       # s of driving before a crash that the visibility check examines
SIM_DT = 0.01          # s per simulator step
SEGMENT_GAP = 3        # crashes whose waypoints are this close are grouped together
MIN_SEGMENT_M = 1e-6   # waypoint segments shorter than this count as duplicates
CAPTURE_RADIUS = 1.5   # m, wrapper.checkpoint_radius: how close counts as clearing a waypoint
STAGNATION_STEPS = 2000
# A "stuck" lap is relabelled "missed_waypoint" when, during the stagnation
# window, the car came within PASS_NEAR_M of its target but never inside the
# capture radius, and then drove at least LEFT_BEHIND_M away from it.
PASS_NEAR_M = 5.0
LEFT_BEHIND_M = 20.0


# --------------------------------------------------------------------------
# Track geometry
# --------------------------------------------------------------------------

def waypoint_curvature(wps):
    """Curvature (1/m) at each waypoint, using the closest distinct neighbours.

    Uses the circle through (previous, this, next) point. Neighbours that sit
    on top of this point (the CSV has a duplicated row) are skipped, so a
    zero-length segment never causes a divide-by-zero. The line is a loop.
    """
    n = len(wps)
    kappa = np.zeros(n)
    for i in range(n):
        p = wps[i]
        j = (i - 1) % n
        while np.linalg.norm(wps[j] - p) < MIN_SEGMENT_M and j != i:
            j = (j - 1) % n
        k = (i + 1) % n
        while np.linalg.norm(wps[k] - p) < MIN_SEGMENT_M and k != i:
            k = (k + 1) % n
        a, b = wps[j] - p, wps[k] - p
        la, lb, lc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(wps[k] - wps[j])
        if min(la, lb, lc) < MIN_SEGMENT_M:
            continue
        cross = a[0] * b[1] - a[1] * b[0]
        kappa[i] = 2.0 * abs(cross) / (la * lb * lc)   # Menger curvature
    return kappa


def duplicate_waypoints(wps):
    n = len(wps)
    return [[i, (i + 1) % n] for i in range(n)
            if np.linalg.norm(wps[(i + 1) % n] - wps[i]) < MIN_SEGMENT_M]


# --------------------------------------------------------------------------
# Visibility: ray-cast on the map image
# --------------------------------------------------------------------------

class MapRaycaster:
    def __init__(self, track):
        meta = yaml.safe_load(open(track.map_yaml))
        self.res = float(meta["resolution"])
        self.origin = np.array(meta["origin"][:2], dtype=float)
        img = np.array(Image.open(track.map_image).convert("L"))
        # Same rule as the simulator: 0 = wall, anything above 0 = free.
        self.wall = img == 0
        self.h, self.w = self.wall.shape
        self.samples = np.arange(0.0, MAX_RANGE, self.res / 2)

    def visible_distance(self, x, y, theta):
        """Free distance (m) straight ahead of (x, y) along heading theta."""
        px = ((x + self.samples * np.cos(theta)) - self.origin[0]) / self.res
        py = self.h - ((y + self.samples * np.sin(theta)) - self.origin[1]) / self.res
        px, py = px.astype(int), py.astype(int)
        outside = (px < 0) | (px >= self.w) | (py < 0) | (py >= self.h)
        hit = outside.copy()
        inside = ~outside
        hit[inside] = self.wall[py[inside], px[inside]]
        idx = np.argmax(hit)
        return float(self.samples[idx]) if hit[idx] else MAX_RANGE


def braking_distance(v):
    """Distance (m) needed to stop from speed v at the simulator's max deceleration."""
    return v * v / (2.0 * A_MAX)


def visibility_check(tel, ray):
    """Could the car have seen the wall early enough to stop?

    Looks at the last LOOKBACK_S seconds before the crash. At each step it
    compares visible distance ahead (map ray-cast) with braking distance at
    that step's speed. If the car never had enough visible distance in that
    window, the corner is "blind" (the wall appeared inside braking range).

    visible_m / required_brake_m are measured at the START of the window
    (LOOKBACK_S before the crash), so they show how much warning the car had.
    last_stoppable_s says how long before the crash it could still have stopped.
    """
    n = len(tel["x"])
    start = max(0, n - int(LOOKBACK_S / SIM_DT))
    vis = np.array([ray.visible_distance(tel["x"][t], tel["y"][t], tel["theta"][t])
                    for t in range(start, n)])
    brake = braking_distance(tel["speed"][start:n])
    ok = np.where(vis - brake >= 0)[0]
    blind = len(ok) == 0
    return {
        "blind": blind,
        "lookback_s": round((n - start) * SIM_DT, 2),
        "visible_m": round(float(vis[0]), 2),
        "required_brake_m": round(float(brake[0]), 2),
        "speed_at_check": round(float(tel["speed"][start]), 2),
        "max_visible_m": round(float(vis.max()), 2),
        "last_stoppable_s": None if blind else round((n - 1 - (start + ok[-1])) * SIM_DT, 2),
        "visible_m_at_crash": round(float(vis[-1]), 2),
        # Sanity check of the ray-caster: the LiDAR's own forward ray.
        "lidar_front_m_at_crash": round(float(tel["scan_front"][-1]), 2),
    }


# --------------------------------------------------------------------------
# Per-lap events and grouping
# --------------------------------------------------------------------------

def nearest_waypoint(wps, x, y):
    return int(np.argmin(np.linalg.norm(wps - np.array([x, y]), axis=1)))


def crash_event(lap, tel, wps, kappa, ray):
    target = int(tel["wp_after"][-1])   # info['current_waypoint_idx'] at the crash
    near = nearest_waypoint(wps, tel["x"][-1], tel["y"][-1])
    k = float(kappa[target])
    return {
        "lap": lap["lap"],
        "mode": lap["mode"],
        "step": lap["steps"] - 1,
        "waypoint": target,
        "nearest_waypoint": near,
        "position": [round(float(tel["x"][-1]), 3), round(float(tel["y"][-1]), 3)],
        "speed": round(float(tel["speed"][-1]), 2),
        "steer_raw": round(float(tel["steer_raw"][-1]), 3),
        "steer_smoothed": round(float(tel["steer_smoothed"][-1]), 3),
        "curvature": round(k, 4),
        "corner_radius_m": round(1.0 / k, 2) if k > 1e-6 else None,
        # How far the car rotated in the lookback window. A large value means
        # it was spinning/looping, so the straight-ahead visibility check is
        # less meaningful than it is for a car driving into a wall.
        "heading_change_deg": heading_change_deg(tel),
        "visibility": visibility_check(tel, ray),
    }


def heading_change_deg(tel):
    start = max(0, len(tel["theta"]) - int(LOOKBACK_S / SIM_DT))
    th = np.unwrap(tel["theta"][start:])
    return round(float(np.degrees(abs(th[-1] - th[0]))), 1)


def stuck_event(lap, tel, wps):
    """Describe a lap the wrapper ended for no waypoint progress.

    Returns (kind, event) where kind is "missed_waypoint" if the car drove
    past its target without entering the capture radius and kept going,
    or "stuck" if it really stopped making progress (e.g. circling).
    """
    target = int(tel["target_wp"][-1])
    window = slice(max(0, len(tel["x"]) - STAGNATION_STEPS), None)
    pos = np.c_[tel["x"][window], tel["y"][window]]
    dist = np.linalg.norm(pos - wps[target], axis=1)
    closest = int(np.argmin(dist))
    missed = (dist[closest] <= PASS_NEAR_M and dist[closest] > CAPTURE_RADIUS
              and dist[closest:].max() - dist[closest] >= LEFT_BEHIND_M)
    path_m = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
    event = {
        "lap": lap["lap"],
        "mode": lap["mode"],
        "waypoint": target,
        "nearest_waypoint": nearest_waypoint(wps, tel["x"][-1], tel["y"][-1]),
        "position": [round(float(tel["x"][-1]), 3), round(float(tel["y"][-1]), 3)],
        "closest_approach_m": round(float(dist[closest]), 2),
        "capture_radius_m": CAPTURE_RADIUS,
        "distance_driven_after_m": round(path_m, 1),
        "mean_speed_last_2s": round(float(tel["speed"][-200:].mean()), 2),
    }
    return ("missed_waypoint" if missed else "stuck"), event


def laps_reaching(laps_tel, wp):
    """How many laps got as far as targeting this waypoint (the crash-rate denominator)."""
    return sum(1 for tel in laps_tel if np.any(tel["target_wp"] == wp)
               or np.any(tel["wp_after"] == wp))


def group_crashes(crashes, laps_tel, n_wp):
    """Merge crashes at nearby waypoints into segments and summarise each one."""
    if not crashes:
        return []
    wps = sorted({c["waypoint"] for c in crashes})
    groups, cur = [], [wps[0]]
    for w in wps[1:]:
        if w - cur[-1] <= SEGMENT_GAP:
            cur.append(w)
        else:
            groups.append(cur)
            cur = [w]
    groups.append(cur)

    segments = []
    for g in groups:
        members = [c for c in crashes if g[0] <= c["waypoint"] <= g[-1]]
        reached = max(laps_reaching(laps_tel, w) for w in g)
        vis = [c["visibility"] for c in members]
        segments.append({
            "waypoints": [g[0], g[-1]],
            "crashes": len(members),
            "laps_reaching": reached,
            "crash_rate": round(len(members) / reached, 3) if reached else None,
            "mean_speed": round(float(np.mean([c["speed"] for c in members])), 2),
            "max_curvature": round(max(c["curvature"] for c in members), 4),
            "blind_fraction": round(sum(v["blind"] for v in vis) / len(vis), 3),
            "mean_visible_m": round(float(np.mean([v["visible_m"] for v in vis])), 2),
            "mean_required_brake_m": round(float(np.mean([v["required_brake_m"] for v in vis])), 2),
            "laps": [c["lap"] for c in members],
        })
    segments.sort(key=lambda s: (-s["crashes"], s["waypoints"][0]))
    return segments


def analyze_run(run_dir):
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "laps.json").read_text())
    track = get_track(meta["track"])
    wps = load_waypoints(track)
    kappa = waypoint_curvature(wps)
    ray = MapRaycaster(track)

    laps_tel, crashes, stuck, missed = [], [], [], []
    counts = {}
    for lap in meta["laps"]:
        tel = dict(np.load(run_dir / f"lap_{lap['lap']:03d}.npz"))
        laps_tel.append(tel)
        kind = lap["termination"]
        if kind == "crash":
            crashes.append(crash_event(lap, tel, wps, kappa, ray))
        elif kind == "stuck":
            kind, event = stuck_event(lap, tel, wps)
            (missed if kind == "missed_waypoint" else stuck).append(event)
        counts[kind] = counts.get(kind, 0) + 1
    finished = [lap for lap in meta["laps"] if lap["termination"] == "finished"]

    analysis = {
        "schema_version": 1,
        "source_model": Path(meta["model"]).stem,
        "track": meta["track"],
        "run_dir": run_dir.name,
        "settings": {"a_max": A_MAX, "lookback_s": LOOKBACK_S, "max_range_m": MAX_RANGE,
                     "segment_gap": SEGMENT_GAP},
        "summary": {
            "laps": len(meta["laps"]),
            "terminations": counts,
            "completion_rate": round(len(finished) / len(meta["laps"]), 3),
            "mean_lap_time_s": round(float(np.mean([l["sim_time_s"] for l in finished])), 2)
            if finished else None,
            "mean_waypoints_cleared": round(float(np.mean([l["waypoints_cleared"] for l in meta["laps"]])), 1),
        },
        "track_info": {
            "num_waypoints": len(wps),
            "duplicate_waypoints": duplicate_waypoints(wps),
        },
        "crash_segments": group_crashes(crashes, laps_tel, len(wps)),
        "crashes": crashes,
        "stuck": stuck,
        "missed_waypoints": missed,
    }
    out = run_dir / "analysis.json"
    out.write_text(json.dumps(analysis, indent=2))
    return analysis, out


def print_report(a):
    s = a["summary"]
    print(f"{a['source_model']} on {a['track']}: {s['laps']} laps, {s['terminations']}, "
          f"completion {s['completion_rate']:.0%}")
    for seg in a["crash_segments"]:
        print(f"  waypoints {seg['waypoints'][0]}-{seg['waypoints'][1]}: "
              f"{seg['crashes']}/{seg['laps_reaching']} crashed, "
              f"mean speed {seg['mean_speed']} m/s, blind {seg['blind_fraction']:.0%}, "
              f"visible {seg['mean_visible_m']} m vs brake {seg['mean_required_brake_m']} m")
    for c in a["crashes"]:
        v = c["visibility"]
        stop = f"could still stop {v['last_stoppable_s']}s before" if not v["blind"] else "blind"
        print(f"  crash: lap {c['lap']} at waypoint {c['waypoint']}, {c['speed']} m/s, "
              f"turned {c['heading_change_deg']} deg in last {v['lookback_s']}s; "
              f"{v['lookback_s']}s before: visible {v['visible_m']} m vs brake "
              f"{v['required_brake_m']} m ({stop})")
    for st in a["stuck"]:
        print(f"  stuck: lap {st['lap']} at waypoint {st['waypoint']}")
    for m in a["missed_waypoints"]:
        print(f"  missed waypoint: lap {m['lap']} passed waypoint {m['waypoint']} at "
              f"{m['closest_approach_m']} m (needs <= {m['capture_radius_m']} m), "
              f"then drove {m['distance_driven_after_m']} m")


def main():
    p = argparse.ArgumentParser(description="Find crashes in recorded laps (post-loop step 2)")
    p.add_argument("--run", required=True, help="folder written by record.py")
    a = p.parse_args()
    analysis, out = analyze_run(a.run)
    print_report(analysis)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
