import math
import unittest
from dataclasses import replace

import numpy as np

from racing_line_learner.config import EnvConfig, get_preset
from racing_line_learner.env.laps import LineLapTimer
from racing_line_learner.env.tracks import load_track
from tests.sim import requires_sim

L = 100.0


class TestLineLapTimer(unittest.TestCase):
    def drive(self, timer, s0, speeds_m_per_step, dt=0.1):
        s, t, laps = s0, 0.0, []
        for ds in speeds_m_per_step:
            t += dt
            lap = timer.update(s, ds, t)
            if lap is not None:
                laps.append(round(lap, 6))
            s = (s + ds) % L
        return laps

    def test_flying_lap(self):
        timer = LineLapTimer(L)
        timer.reset(s0=90.0)                       # 10 m before the line: out-lap
        self.assertFalse(timer.running)
        laps = self.drive(timer, 90.0, [1.0] * 230)
        # Crosses the line after 10 steps (t=1.0), then every 100 steps.
        self.assertEqual(laps, [10.0, 10.0])
        self.assertEqual(timer.laps, 2)

    def test_standing_start_on_line(self):
        for s0 in (0.0, 1.5, 98.8):                # on, just past, just before the line
            timer = LineLapTimer(L)
            timer.reset(s0=s0)
            self.assertTrue(timer.running, s0)
            laps = self.drive(timer, s0, [1.0] * 105)
            self.assertEqual(len(laps), 1, s0)
            signed = s0 if s0 < 50 else s0 - L       # metres past the line at the start
            steps = math.ceil(L - signed - 1e-9)     # whole 1 m steps until a full lap is covered
            self.assertAlmostEqual(laps[0], steps * 0.1, places=6)

    def test_wiggling_over_the_line_is_not_a_lap(self):
        timer = LineLapTimer(L)
        timer.reset(s0=99.0)                       # within the standing tolerance before the line
        laps = self.drive(timer, 99.0, [2.0, -2.0] * 50)
        self.assertEqual(laps, [])

    def test_reversing_does_not_count(self):
        timer = LineLapTimer(L)
        timer.reset(s0=50.0)
        laps = self.drive(timer, 50.0, [-1.0] * 300)  # a full lap backwards
        self.assertEqual(laps, [])

    def test_invalid_length(self):
        with self.assertRaises(ValueError):
            LineLapTimer(0.0)


class TestFlyingStart(unittest.TestCase):
    def test_out_lap_distance(self):
        for name in ("cota", "sepang", "sepang_wide"):
            t = load_track(name)
            w = t.flying_start_waypoint()
            out_lap = t.length - t.cum_s[w]
            self.assertGreaterEqual(out_lap, max(50.0, 0.1 * t.length) - 1e-6, name)
            self.assertLess(out_lap, max(50.0, 0.1 * t.length) + 6.0, name)   # within ~2 waypoints of the target

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            replace(EnvConfig(), lap_timing="line")                       # legacy mode has no projection
        with self.assertRaises(ValueError):
            replace(get_preset("v2"), lap_timing="sector")
        replace(get_preset("v2"), lap_timing="line")


class HeadingDriver:
    """Stand-in policy for v2 envs: steer at the heading error, slow speed."""

    def __init__(self, obs_dim, speed_action=-0.6):
        self.obs_dim, self.speed_action = obs_dim, speed_action

    def predict(self, obs, deterministic=True):
        heading = obs[-self.obs_dim + 22] * np.pi        # last frame's heading error (normalised by pi)
        return np.array([np.clip(heading / 0.4189, -1, 1), self.speed_action], dtype=np.float32), None


@requires_sim
class TestLineTimingInSimulator(unittest.TestCase):
    def test_flying_out_lap_starts_clock_at_line(self):
        from racing_line_learner.env.wrapper import make_env
        cfg = replace(get_preset("v2"), spawn_random=0.0, lap_timing="line", max_laps=1)
        env = make_env("cota", cfg)
        self.addCleanup(env.close)
        driver = HeadingDriver(env.obs_dim)
        obs = env.reset(spawn_idx=env.track.n_waypoints - 3)   # ~9 m before the line
        self.assertFalse(env._lap_timer.running)
        for _ in range(100):
            obs, _, done, info = env.step(driver.predict(obs)[0])
            if env._lap_timer.running or done:
                break
        self.assertTrue(env._lap_timer.running)
        self.assertNotEqual(info["termination"], "lap_complete")

    def test_standing_start_runs_clock_from_rest(self):
        from racing_line_learner.env.wrapper import make_env
        env = make_env("sepang", replace(get_preset("v2"), spawn_random=0.0, lap_timing="line", max_laps=1))
        self.addCleanup(env.close)
        env.reset()                                        # sepang starts at waypoint 0, on the line
        self.assertTrue(env._lap_timer.running)

    def test_evaluate_smoke(self):
        from racing_line_learner.eval.laps import evaluate
        cfg = replace(get_preset("v2"), max_episode_steps=400)
        from racing_line_learner.env.wrapper import make_env
        probe = make_env("cota", cfg)
        driver = HeadingDriver(probe.obs_dim)
        probe.close()
        import racing_line_learner.eval.laps as laps
        old = laps.DEFAULT_MAX_SIM_STEPS
        laps.DEFAULT_MAX_SIM_STEPS = 400                   # keep the smoke test short
        try:
            res = evaluate(driver, "cota", cfg, repeats=2, n_starts=3)
        finally:
            laps.DEFAULT_MAX_SIM_STEPS = old
        self.assertEqual(res["flying"]["runs"], 2)
        self.assertEqual(res["standing"]["runs"], 2)
        self.assertEqual(res["robustness"]["runs"], 3)
        self.assertEqual(res["flying"]["start_waypoint"], load_track("cota").flying_start_waypoint())
        for mode in ("flying", "standing", "robustness"):
            for run in res[mode]["per_run"]:
                self.assertIn(run["termination"], ("time_limit", "collision", "stagnation", "lap_complete"))

    def test_legacy_model_skips_flying(self):
        from racing_line_learner.eval.laps import eval_config
        self.assertEqual(eval_config(EnvConfig(), "line").lap_timing, "spawn")


if __name__ == "__main__":
    unittest.main()
