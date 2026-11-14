"""Wrapper tests against the real simulator.

The parity test drives the original ``wrapper.py`` and the refactored
wrapper (legacy preset) with the same action sequence and requires
identical observations, rewards and dones.
"""

import importlib.util
import os
import unittest
from dataclasses import replace

import numpy as np

from racing_line_learner.config import REPO_ROOT, EnvConfig, get_preset
from tests.sim import requires_sim


def _legacy_module():
    spec = importlib.util.spec_from_file_location("legacy_wrapper", str(REPO_ROOT / "wrapper.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _controller(rng):
    """Noisy heading-following driver: produces progress, waypoint clears and crashes."""
    def act(obs):
        heading_err = float(obs[-1])
        steer = np.clip(1.2 * heading_err + rng.normal(0, 0.15), -3.14, 3.14)
        speed = np.clip(8.0 + rng.normal(0, 2.0), 0.0, 20.0)
        return np.array([steer, speed], dtype=np.float32)
    return act


def _rollout(env, n_steps, seed):
    rng = np.random.RandomState(seed)
    act = _controller(rng)
    obs = env.reset()
    out = [("reset", obs.copy())]
    for _ in range(n_steps):
        a = act(obs)
        obs, reward, done, info = env.step(a.copy())
        out.append((a, obs.copy(), reward, done, info.get("current_waypoint_idx")))
        if done:
            obs = env.reset()
            out.append(("reset", obs.copy()))
    return out


@requires_sim
class TestLegacyParity(unittest.TestCase):
    N_STEPS = 3000

    @classmethod
    def setUpClass(cls):
        import gym
        from racing_line_learner.env.wrapper import make_env

        cls._cwd = os.getcwd()
        os.chdir(str(REPO_ROOT))  # the original wrapper loads its CSV relative to cwd
        try:
            legacy = _legacy_module().F1TenthWrapper(
                gym.make("f110_gym:f110-v0", map="maps/vegas", map_ext=".png", num_agents=1), algo="ppo")
            cls.old = _rollout(legacy, cls.N_STEPS, seed=7)
            legacy.close()
            new = make_env("cota", EnvConfig())
            cls.new = _rollout(new, cls.N_STEPS, seed=7)
            new.close()
        finally:
            os.chdir(cls._cwd)

    def test_spaces_match(self):
        import gym
        from racing_line_learner.env.wrapper import make_env
        env = make_env("cota", EnvConfig())
        os.chdir(str(REPO_ROOT))
        try:
            legacy = _legacy_module().F1TenthWrapper(
                gym.make("f110_gym:f110-v0", map="maps/vegas", map_ext=".png", num_agents=1), algo="ppo")
        finally:
            os.chdir(self._cwd)
        self.assertEqual(env.observation_space.shape, (69,))
        self.assertEqual(env.observation_space, legacy.observation_space)
        self.assertEqual(env.action_space, legacy.action_space)
        env.close()
        legacy.close()

    def test_identical_trajectories(self):
        self.assertEqual(len(self.old), len(self.new))
        resets = 0
        for i, (o, n) in enumerate(zip(self.old, self.new)):
            if isinstance(o[0], str):
                resets += 1
                np.testing.assert_array_equal(o[1], n[1], err_msg=f"reset obs differs at {i}")
                continue
            np.testing.assert_array_equal(o[0], n[0], err_msg=f"action differs at {i}")
            np.testing.assert_array_equal(o[1], n[1], err_msg=f"obs differs at step {i}")
            self.assertEqual(o[2], n[2], f"reward differs at step {i}")
            self.assertEqual(o[3], n[3], f"done differs at step {i}")
            self.assertEqual(o[4], n[4], f"waypoint differs at step {i}")
        self.assertGreaterEqual(resets, 2, "rollout should cover several episodes")

    def test_rollout_covers_waypoint_clears(self):
        wps = [s[4] for s in self.new if not isinstance(s[0], str)]
        self.assertGreater(max(wps), 3)


@requires_sim
class TestWrapperFeatures(unittest.TestCase):
    def make(self, config=EnvConfig(), seed=0):
        from racing_line_learner.env.wrapper import make_env
        env = make_env("cota", config, seed=seed)
        self.addCleanup(env.close)
        return env

    def test_action_not_mutated_and_steering_in_info(self):
        env = self.make()
        env.reset()
        a = np.array([1.0, 8.0], dtype=np.float32)
        _, _, _, info = env.step(a)
        self.assertEqual(a[0], 1.0)
        self.assertEqual(info["raw_steering"], 1.0)
        self.assertAlmostEqual(info["steering"], 0.3, places=6)  # EMA: 0.3*1.0 + 0.7*0
        self.assertEqual(info["speed_command"], 8.0)
        self.assertIn("reward_terms", info)

    def test_collision_termination(self):
        env = self.make()
        env.reset()
        info, done = {}, False
        for _ in range(3000):
            _, _, done, info = env.step(np.array([3.14, 10.0], dtype=np.float32))  # full lock into a wall
            if done:
                break
        self.assertTrue(done)
        self.assertEqual(info["termination"], "collision")

    def test_stagnation_and_time_limit(self):
        env = self.make(replace(EnvConfig(), stagnation_steps=5))
        env.reset()
        info = {}
        for _ in range(50):
            # Drive away from the line: no waypoint is cleared.
            _, reward, done, info = env.step(np.array([0.0, 6.5], dtype=np.float32))
            if done:
                break
        self.assertEqual(info["termination"], "stagnation")
        self.assertEqual(reward, -10.0)

        env = self.make(replace(EnvConfig(), max_episode_steps=10))
        env.reset()
        for _ in range(10):
            _, _, done, info = env.step(np.array([0.0, 6.5], dtype=np.float32))
        self.assertTrue(done)
        self.assertEqual(info["termination"], "time_limit")
        self.assertTrue(info["TimeLimit.truncated"])

    def test_spawn_at_waypoint(self):
        env = self.make()
        obs = env.reset(spawn_idx=100)
        x, y, _ = env.track.spawn_pose(100)
        self.assertAlmostEqual(env.last_raw_obs["poses_x"][0], x)
        self.assertAlmostEqual(env.last_raw_obs["poses_y"][0], y)
        self.assertEqual(env.current_waypoint_idx, 101)
        self.assertLess(abs(obs[-1]), 1e-3)  # facing the target: heading error ~0

    def test_spawn_mix_curriculum(self):
        env = self.make(replace(EnvConfig(), spawn_waypoints=(40, 90), spawn_mix=0.5), seed=3)
        spawns = []
        for _ in range(40):
            env.reset()
            spawns.append(env.spawn_idx)
        self.assertTrue({40, 90, None} <= set(spawns))

    def test_set_config(self):
        env = self.make()
        env.set_config(replace(env.config, target_speed=8.0))
        self.assertEqual(env.config.target_speed, 8.0)
        with self.assertRaises(ValueError):
            env.set_config(replace(env.config, steer_limit=0.4))
        with self.assertRaises(ValueError):
            env.set_config(replace(env.config, spawn_waypoints=(500,), spawn_mix=0.1))

    def test_laptime_mode_progress(self):
        env = self.make(get_preset("laptime"))
        self.assertAlmostEqual(float(env.action_space.high[0]), 0.4189, places=4)
        obs = env.reset(spawn_idx=0)
        total_progress, cleared = 0.0, 0
        for _ in range(300):
            steer = np.clip(1.0 * obs[-1], -0.4189, 0.4189)
            obs, reward, done, info = env.step(np.array([steer, 8.0], dtype=np.float32))
            total_progress += info["reward_terms"]["progress"]
            cleared += int(info["waypoint_cleared"])
            if done:
                break
        self.assertGreater(total_progress, 0.0)
        self.assertGreater(cleared, 0)


if __name__ == "__main__":
    unittest.main()


@requires_sim
class TestV2Features(unittest.TestCase):
    def make(self, **overrides):
        from racing_line_learner.env.wrapper import make_env
        env = make_env("cota", replace(get_preset("v2"), **overrides))
        self.addCleanup(env.close)
        return env

    def test_spaces(self):
        env = self.make()
        self.assertEqual(env.action_space.shape, (2,))
        np.testing.assert_array_equal(env.action_space.low, [-1.0, -1.0])
        self.assertEqual(env.observation_space.shape, ((20 + 3 + 8) * 3,))

    def test_normalized_action_mapping(self):
        env = self.make()
        env.reset(spawn_idx=0)
        _, _, _, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
        self.assertAlmostEqual(info["raw_steering"], 0.4189)
        self.assertAlmostEqual(info["steering"], 0.4189)        # EMA off in v2
        self.assertAlmostEqual(info["speed_command"], 20.0)
        _, _, _, info = env.step(np.array([-1.0, -1.0], dtype=np.float32))
        self.assertAlmostEqual(info["raw_steering"], -0.4189)
        self.assertAlmostEqual(info["speed_command"], 2.0)       # speed floor
        _, _, _, info = env.step(np.array([5.0, 0.0], dtype=np.float32))   # out of range is clipped
        self.assertAlmostEqual(info["raw_steering"], 0.4189)
        self.assertAlmostEqual(info["speed_command"], 5.0)    # action 0 -> speed_mid

    def test_action_repeat_sums_rewards(self):
        env = self.make()
        env.reset(spawn_idx=0)
        for i in range(1, 6):
            _, reward, done, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
            self.assertEqual(info["sim_steps"], 4 * i)
            self.assertEqual(info["episode_steps"], i)
            self.assertAlmostEqual(reward, sum(info["reward_terms"].values()), places=9)
            self.assertAlmostEqual(info["reward_terms"]["time"], -0.4)   # 4 x time_cost
        self.assertFalse(done)

    def test_action_repeat_stops_at_terminal_step(self):
        env = self.make(stagnation_steps=6)
        env.reset(spawn_idx=0)
        for _ in range(10):
            # Steer hard off the line at the floor speed until the episode ends.
            _, reward, done, info = env.step(np.array([1.0, -1.0], dtype=np.float32))
            if done:
                break
        self.assertTrue(done)
        self.assertIn(info["termination"], ("stagnation", "collision"))
        if info["termination"] == "stagnation":
            # Stagnation fires on the 7th simulator step without progress, mid-repeat: the
            # loop must stop there rather than finish the 4-step block.
            self.assertEqual(info["sim_steps"], 7)

    def test_normalized_observation_ranges(self):
        env = self.make()
        obs = env.reset(spawn_idx=100)
        frame = obs[-env.obs_dim:]
        self.assertTrue(np.all(frame[:20] <= 1.0 + 1e-6) and np.all(frame[:20] >= 0.0))
        self.assertTrue(np.all(np.abs(frame[20:23]) <= 1.0 + 1e-6))
        self.assertTrue(np.all(np.abs(frame[23:]) < 3.0))

    def test_lookahead_sees_the_hairpin(self):
        # Spawn a few waypoints before the hairpin at 146: the lookahead should show high curvature.
        env = self.make(obs_normalize=False)
        obs = env.reset(spawn_idx=141)
        look = np.abs(obs[-8:])
        self.assertGreater(look.max(), 0.2)    # radius < 5 m within 16 m
        far = np.abs(env.reset(spawn_idx=125)[-8:])
        self.assertEqual(far.shape, (8,))

    def test_v2_drives(self):
        env = self.make()
        obs = env.reset(spawn_idx=0)
        cleared = 0
        for _ in range(150):
            heading = obs[env.obs_dim * 2 + 22] * np.pi   # last frame's heading error (de-normalised)
            obs, reward, done, info = env.step(np.array([np.clip(heading / 0.4189, -1, 1), -0.2], dtype=np.float32))
            cleared += int(info["waypoint_cleared"])
            if done:
                break
        self.assertGreater(cleared, 3)


@requires_sim
class TestRandomSpawns(unittest.TestCase):
    def test_spawn_random_covers_track(self):
        from racing_line_learner.env.wrapper import make_env
        env = make_env("cota", replace(get_preset("laptime"), spawn_random=0.75), seed=1)
        self.addCleanup(env.close)
        spawns = [(env.reset(), env.spawn_idx)[1] for _ in range(80)]
        random = [s for s in spawns if s is not None]
        self.assertTrue(40 <= len(random) <= 75, len(random))
        self.assertGreater(len(set(random)), 30)          # spread over the lap
        self.assertIn(None, spawns)                        # some standard starts remain
