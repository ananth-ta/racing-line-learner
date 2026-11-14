import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from racing_line_learner.config import (
    OUTPUT_DIR_ENV,
    PRESETS,
    REPO_ROOT,
    SIM_STEER_LIMIT,
    EnvConfig,
    apply_recommendation,
    get_preset,
    output_root,
    save_json,
)
from racing_line_learner.env.rewards import laptime_reward, legacy_reward
from racing_line_learner.schema import WHITELIST, parse_recommendation

LEGACY = EnvConfig()
LAPTIME = get_preset("laptime")


class TestConfig(unittest.TestCase):
    def test_legacy_defaults_match_original_constants(self):
        c = LEGACY
        self.assertEqual((c.speed_floor, c.target_speed, c.speed_variance, c.progress_coef), (6.5, 10.0, 2.0, 10.0))
        self.assertEqual((c.waypoint_bonus, c.steering_penalty_w, c.jitter_coef, c.ema_alpha), (50.0, 2.0, 0.05, 0.3))
        self.assertEqual((c.stagnation_steps, c.checkpoint_radius, c.steer_limit), (2000, 1.5, 3.14))

    def test_whitelist_defaults_match_legacy(self):
        for key, spec in WHITELIST.items():
            if hasattr(LEGACY, key):
                self.assertEqual(getattr(LEGACY, key), spec.default, key)

    def test_laptime_preset(self):
        self.assertEqual(LAPTIME.reward_mode, "laptime")
        self.assertEqual(LAPTIME.steer_limit, SIM_STEER_LIMIT)
        self.assertFalse(LAPTIME.legacy_speed_from_yaw_rate)
        with self.assertRaises(ValueError):
            get_preset("fast")

    def test_validation(self):
        for bad in ({"reward_mode": "x"}, {"speed_floor": 30.0}, {"spawn_mix": 0.3},
                    {"ema_alpha": 0.0}, {"max_episode_steps": 0}, {"spawn_waypoints": (-1,)}, {"max_laps": 3}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(LEGACY, **bad)

    def test_dict_roundtrip_and_unknown_keys(self):
        c = replace(LAPTIME, spawn_waypoints=(3, 7), spawn_mix=0.2)
        d = json.loads(json.dumps(c.to_dict()))
        self.assertEqual(EnvConfig.from_dict(d), c)
        with self.assertRaises(ValueError):
            EnvConfig.from_dict({"bogus": 1})

    def test_apply_recommendation(self):
        rec = parse_recommendation({
            "schema_version": 1, "source_model": "m", "track": "cota",
            "changes": {"ent_coef": 0.02}, "curriculum": {"spawn_waypoints": [104], "spawn_mix": 0.3},
        }, 150)
        cfg, algo = apply_recommendation(LEGACY, rec)
        self.assertEqual(algo, {"ent_coef": 0.02})
        self.assertEqual(cfg.spawn_waypoints, (104,))
        self.assertEqual(cfg.spawn_mix, 0.3)
        rec2 = parse_recommendation({"schema_version": 1, "source_model": "m", "track": "cota",
                                     "changes": {"target_speed": 9.0}}, 150)
        self.assertEqual(apply_recommendation(LEGACY, rec2)[0].target_speed, 9.0)

    def test_output_root(self):
        old = os.environ.pop(OUTPUT_DIR_ENV, None)
        try:
            self.assertEqual(output_root(), REPO_ROOT)
            os.environ[OUTPUT_DIR_ENV] = "/tmp/rll_env"
            self.assertEqual(output_root(), Path("/tmp/rll_env"))
            self.assertEqual(output_root("/tmp/rll_cli"), Path("/tmp/rll_cli"))
        finally:
            os.environ.pop(OUTPUT_DIR_ENV, None)
            if old is not None:
                os.environ[OUTPUT_DIR_ENV] = old

    def test_save_json_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a" / "state.json"
            save_json({"x": 1}, path)
            self.assertEqual(json.loads(path.read_text()), {"x": 1})
            self.assertEqual(os.listdir(path.parent), ["state.json"])


class TestLegacyRewardRegression(unittest.TestCase):
    """Fixed inputs -> hand-computed outputs of the original wrapper.py PPO reward."""

    def r(self, **kw):
        args = dict(speed_signal=8.0, raw_steer=0.5, prev_raw_steer=0.1, progress_reward=1.2,
                    waypoint_cleared=False, env_done=False, stagnated=False)
        args.update(kw)
        return legacy_reward(LEGACY, **args)

    def test_running_step(self):
        # 2*exp(-(8-10)^2 / (2*2^2)) - 0.05*|0.5-0.1| + 1.2 - 2*0.5^2
        reward, terms = self.r()
        self.assertAlmostEqual(reward, 2 * 0.6065306597126334 - 0.02 + 1.2 - 0.5, places=12)
        self.assertAlmostEqual(sum(terms.values()), reward, places=12)

    def test_waypoint_bonus(self):
        self.assertAlmostEqual(self.r(waypoint_cleared=True, progress_reward=0.0)[0],
                               2 * 0.6065306597126334 - 0.02 + 50.0 - 0.5, places=12)

    def test_done_step_still_adds_terms(self):
        # Legacy quirk: -10, then progress, bonus and steering are still applied.
        self.assertAlmostEqual(self.r(env_done=True, waypoint_cleared=True, progress_reward=0.0)[0], 39.5)
        self.assertAlmostEqual(self.r(env_done=True)[0], -10.0 + 1.2 - 0.5)

    def test_stagnation_overrides(self):
        self.assertEqual(self.r(stagnated=True, waypoint_cleared=False)[0], -10.0)

    def test_yaw_rate_spinning_is_rewarded(self):
        # The legacy speed term peaks at speed_signal = 10, and the signal is the yaw rate.
        spin, _ = self.r(speed_signal=8.8, raw_steer=0.0, prev_raw_steer=0.0, progress_reward=0.0)
        straight, _ = self.r(speed_signal=0.0, raw_steer=0.0, prev_raw_steer=0.0, progress_reward=0.0)
        self.assertGreater(spin, 1.6)
        self.assertLess(straight, 1e-4)


class TestLaptimeReward(unittest.TestCase):
    def r(self, **kw):
        args = dict(velocity=10.0, raw_steer=0.2, prev_raw_steer=0.1, progress_m=0.1, waypoints_cleared=1,
                    collision=False, stagnated=False, lap_complete=False)
        args.update(kw)
        return laptime_reward(LAPTIME, **args)

    def test_running_step(self):
        reward, terms = self.r()
        # 10*0.1 + 50 - 0.1 - 2*0.04 - 0.05*0.1
        self.assertAlmostEqual(reward, 1.0 + 50.0 - 0.1 - 0.08 - 0.005, places=12)
        self.assertEqual(terms["speed"], 0.0)

    def test_terminal_replaces_step_reward(self):
        for kw in ({"collision": True}, {"stagnated": True}):
            reward, terms = self.r(**kw)
            self.assertEqual(reward, -10.0)
            self.assertEqual(terms["bonus"], 0.0)

    def test_lap_bonus(self):
        self.assertAlmostEqual(self.r(lap_complete=True)[0] - self.r()[0], 100.0)

    def test_faster_progress_pays_more(self):
        self.assertGreater(self.r(progress_m=0.2)[0], self.r(progress_m=0.1)[0])
        self.assertLess(self.r(progress_m=-0.1, waypoints_cleared=0)[0], 0.0)

    def test_speed_term_reads_velocity(self):
        cfg = replace(LAPTIME, speed_reward_w=2.0)
        r_on, _ = laptime_reward(cfg, velocity=10.0, raw_steer=0.0, prev_raw_steer=0.0, progress_m=0.0,
                                 waypoints_cleared=0, collision=False, stagnated=False, lap_complete=False)
        self.assertAlmostEqual(r_on, 2.0 - 0.1)


if __name__ == "__main__":
    unittest.main()


class TestV2Config(unittest.TestCase):
    def test_v2_preset(self):
        from racing_line_learner.config import get_preset
        v2 = get_preset("v2")
        self.assertEqual((v2.reward_mode, v2.action_normalized, v2.action_repeat), ("laptime", True, 4))
        self.assertEqual((v2.speed_floor, v2.ema_alpha, v2.lookahead_k, v2.obs_normalize), (2.0, 1.0, 8, True))
        self.assertEqual((v2.speed_mid, v2.terminal_penalty, v2.spawn_random), (5.0, 1000.0, 0.75))

    def test_legacy_mode_rejects_v2_features(self):
        for bad in ({"action_repeat": 2}, {"action_normalized": True}, {"lookahead_k": 4}, {"obs_normalize": True}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(LEGACY, **bad)

    def test_parse_overrides(self):
        from racing_line_learner.config import parse_overrides
        o = parse_overrides(["speed_floor=6.5", "lookahead_k=0", "obs_normalize=false", "spawn_waypoints=[3,4]"])
        self.assertEqual(o, {"speed_floor": 6.5, "lookahead_k": 0, "obs_normalize": False, "spawn_waypoints": (3, 4)})
        self.assertEqual(parse_overrides(["reward_mode=laptime"]), {"reward_mode": "laptime"})
        for bad in (["nope=1"], ["speed_floor"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_overrides(bad)
