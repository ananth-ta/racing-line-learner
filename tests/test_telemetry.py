import json
import unittest
from dataclasses import replace

import numpy as np

from racing_line_learner.config import EnvConfig, get_preset
from racing_line_learner.env.tracks import load_track
from racing_line_learner.in_loop import detectors
from racing_line_learner.in_loop.detectors import detect
from racing_line_learner.in_loop.telemetry import TelemetryCollector, summarize_probe

TRACK = load_track("cota")
LAPTIME = get_preset("laptime")


def step_info(wp, speed=6.5, cmd=6.5, steer=0.0, termination=None, cleared=False, lap_time=None, terms=None):
    info = {"current_waypoint_idx": wp, "speed": speed, "speed_command": cmd, "raw_steering": steer,
            "termination": termination, "waypoint_cleared": cleared,
            "reward_terms": terms or {"progress": 0.5, "bonus": 0.0, "time": -0.1}}
    if lap_time is not None:
        info["lap_time"] = lap_time
    return info


def drive(collector, start, end, crash=True, speed=6.5, cmd=6.5, env_index=0):
    """One synthetic episode from waypoint ``start`` to ``end``, ending in a crash or a lap."""
    for wp in range(start, end):
        collector.update(step_info(wp, speed, cmd, cleared=True), 1.0, env_index)
    term = "collision" if crash else "lap_complete"
    collector.update(step_info(end, speed, cmd, termination=term, lap_time=None if crash else 70.0), -10.0, env_index)


class TestCollector(unittest.TestCase):
    def test_counts_and_hotspot(self):
        c = TelemetryCollector(TRACK, LAPTIME)
        for _ in range(8):
            drive(c, 90, 111)              # the hairpin
        drive(c, 30, 52)
        drive(c, 0, 149, crash=False)
        r = c.report(window=3, timesteps=750000)
        json.dumps(r)                       # serialisable
        self.assertEqual(r["episodes"], 10)
        self.assertEqual(r["terminations"]["collision"]["count"], 9)
        self.assertEqual(r["laps"]["completed"], 1)
        self.assertAlmostEqual(r["laps"]["completion_rate"], 0.1)
        top = r["crash_hotspots"][0]
        self.assertEqual(top["waypoints"], [111, 111])
        self.assertEqual(top["collisions"], 8)
        self.assertAlmostEqual(top["approach_speed"], 6.5)
        self.assertLess(top["grip_limit"], 6.0)    # tight hairpin
        self.assertAlmostEqual(r["speed"]["pinned_at_floor_share"], 1.0)
        self.assertAlmostEqual(r["speed"]["floor_limited_lap_time_s"], TRACK.length / 6.5, places=1)
        self.assertEqual(len(r["segments"]), 15)
        self.assertAlmostEqual(sum(s["time_share"] for s in r["segments"]), 1.0, places=2)
        self.assertEqual(sum(s["collisions"] for s in r["segments"]), 9)

    def test_hotspot_clusters_neighbours_and_wraps(self):
        c = TelemetryCollector(TRACK, LAPTIME)
        for wp in (149, 0, 0, 1, 149, 0):
            c.update(step_info(wp, termination="collision"), -10.0)
        h = c.crash_hotspots()
        self.assertEqual(len(h), 1)
        self.assertEqual(h[0]["waypoints"], [149, 1])
        self.assertEqual(h[0]["collisions"], 6)

    def test_parallel_envs_tracked_separately(self):
        c = TelemetryCollector(TRACK, LAPTIME)
        for wp in range(10):
            c.update(step_info(wp, cleared=True), 1.0, env_index=0)
            c.update(step_info(50 + wp, cleared=True), 1.0, env_index=1)
        c.update(step_info(10, termination="collision"), -10.0, env_index=0)
        r = c.report()
        self.assertEqual(r["episodes"], 1)
        self.assertEqual(r["episode"]["mean_waypoints_cleared"], 10.0)

    def test_saturation_uses_simulator_limit(self):
        c = TelemetryCollector(TRACK, EnvConfig())  # legacy: action space +/-3.14
        for _ in range(4):
            c.update(step_info(5, steer=0.5), 0.0)  # beyond the 0.4189 rad physical limit
        c.update(step_info(5, steer=0.1), 0.0)
        self.assertAlmostEqual(c.report()["steering"]["saturation_share"], 0.8)

    def test_reset(self):
        c = TelemetryCollector(TRACK, LAPTIME)
        drive(c, 0, 10)
        c.reset()
        r = c.report()
        self.assertEqual((r["steps"], r["episodes"], r["crash_hotspots"]), (0, 0, []))

    def test_probe_summary(self):
        res = [{"spawn": None, "termination": "lap_complete", "lap_time": 70.4, "waypoint": 110},
               {"spawn": 30, "termination": "collision", "lap_time": None, "waypoint": 111},
               {"spawn": 60, "termination": "collision", "lap_time": None, "waypoint": 111}]
        p = summarize_probe(res)
        self.assertAlmostEqual(p["completion_rate"], 0.33)
        self.assertEqual(p["failures_at"], [{"waypoint": 111, "count": 2}])


class TestDetectors(unittest.TestCase):
    def ids(self, report):
        return [f["id"] for f in detect(report)]

    def test_hairpin_diagnosis(self):
        c = TelemetryCollector(TRACK, LAPTIME)
        for _ in range(8):
            drive(c, 90, 111)
        drive(c, 0, 149, crash=False)
        findings = detect(c.report())
        ids = [f["id"] for f in findings]
        self.assertIn("crash_hotspot", ids)
        self.assertIn("floor_blocks_braking", ids)
        self.assertIn("speed_pinned_at_floor", ids)
        self.assertIn("under_driven_segment", ids)
        hot = next(f for f in findings if f["id"] == "crash_hotspot")
        self.assertTrue(hot["facts"]["overspeed"])
        self.assertEqual(hot["waypoints"], [111, 111])
        self.assertEqual(findings[0]["severity"], "high")

    def test_flat_out_policy(self):
        cfg = get_preset("v2")
        c = TelemetryCollector(TRACK, cfg)
        for _ in range(10):
            drive(c, 0, 19, speed=20.0, cmd=20.0)
        ids = self.ids(c.report())
        self.assertIn("speed_pinned_at_max", ids)
        self.assertIn("crash_hotspot", ids)
        self.assertNotIn("floor_blocks_braking", ids)   # floor is 2 m/s in v2

    def test_corkscrew(self):
        c = TelemetryCollector(TRACK, EnvConfig())
        spin = {"progress": 0.0, "bonus": 0.0, "speed": 1.6, "steering": -0.35}
        for ep in range(5):
            for _ in range(200):
                c.update(step_info(3, speed=6.5, steer=3.0, terms=spin), 1.25)
            c.update(step_info(3, termination="stagnation", terms=spin), -10.0)
        ids = self.ids(c.report())
        self.assertIn("corkscrew", ids)
        self.assertIn("stagnation_dominant", ids)

    def test_healthy_run_has_no_high_findings(self):
        cfg = replace(get_preset("v2"))
        c = TelemetryCollector(TRACK, cfg)
        grip = TRACK.grip_speed_limit()
        for _ in range(3):
            for wp in range(150):
                v = float(min(grip[(wp - 1) % 150], 20.0)) * 0.9
                c.update(step_info(wp, speed=v, cmd=v, cleared=True), 1.0)
            c.update(step_info(0, termination="lap_complete", lap_time=40.0), 100.0)
        findings = detect(c.report())
        self.assertFalse([f for f in findings if f["severity"] == "high"], findings)

    def test_minor_hotspot_ignored(self):
        c = TelemetryCollector(TRACK, LAPTIME)
        for start in range(0, 140, 14):     # crashes spread around the track
            drive(c, start, start + 5)
        self.assertNotIn("crash_hotspot", self.ids(c.report()))
        self.assertLess(max(h["share"] for h in c.report()["crash_hotspots"] or [{"share": 0}]),
                        detectors.HOTSPOT_MIN_SHARE)


if __name__ == "__main__":
    unittest.main()
