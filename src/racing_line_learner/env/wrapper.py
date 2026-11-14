"""F1TENTH Gym wrapper: observation processing, action mapping, rewards and termination.

Each observation frame is ``lidar_rays`` downsampled LiDAR ranges plus
``[velocity, yaw_rate, heading_error]`` (the heading error points at the
current target waypoint), optionally followed by ``lookahead_k`` samples of
racing-line curvature ahead of the car. ``n_stack`` frames are concatenated:
23 x 3 = 69 values with the defaults. With ``obs_normalize`` the features
are divided by fixed scales (LiDAR range, ``speed_max``, 10 rad/s, pi,
1/2 m) so they lie roughly in [-1, 1].

Actions are ``[steering, speed]``, either in physical units (radians, m/s)
or, with ``action_normalized``, in [-1, 1] mapped to ``+/-steer_limit`` and
``[speed_floor, speed_max]``. Each action is held for ``action_repeat``
simulator steps (0.01 s each); rewards are summed over them, and the episode
stops at the first terminal simulator step.

Every step's ``info`` carries:

* ``termination``: ``None`` or one of :data:`TERMINATIONS`;
* ``raw_steering`` (the policy's command, radians) and ``steering`` (after
  the EMA filter, what the simulator receives), plus ``speed_command``;
* ``reward_terms``: the reward broken into named parts (summed over the
  repeated simulator steps);
* ``current_waypoint_idx``, ``waypoint_cleared``, ``progress_reward``;
* ``laps`` completed so far, and ``lap_time`` (seconds) on the step a lap
  completes. Laps are counted by the simulator when the car re-enters the
  start zone around its spawn pose; the simulator itself ends the episode
  after 2 laps, ``EnvConfig.max_laps=1`` ends it after one;
* ``episode_steps`` (policy steps) and ``sim_steps`` (simulator steps);
* ``TimeLimit.truncated`` when ``max_episode_steps`` ends the episode, so
  SB3 bootstraps instead of treating it as a true terminal.

With ``EnvConfig()`` (the ``legacy`` preset) observations, rewards and
dones match the original ``wrapper.py`` exactly; ``tests/test_wrapper.py``
checks this against the simulator.
"""

import math
from collections import deque
from typing import Optional, Sequence

import gym
import numpy as np
from gym import spaces

from ..config import CURVATURE_SCALE, FROZEN_FIELDS, LIDAR_MAX_RANGE, YAW_RATE_SCALE, EnvConfig
from .rewards import laptime_reward, legacy_reward
from .tracks import Track, load_track

TERMINATIONS = ("collision", "stagnation", "lap_complete", "time_limit")

#: Pose the original wrapper always started from.
LEGACY_START_POSE = (0.0, 0.0, 0.0)


class F1TenthWrapper(gym.Wrapper):
    """Wraps ``f110_gym:f110-v0`` for single-agent racing on a :class:`Track`."""

    def __init__(self, env: gym.Env, track: Track, config: EnvConfig = EnvConfig(), seed: Optional[int] = None):
        super().__init__(env)
        self.track = track
        self.config = config
        self._check_spawn(config)
        self._rng = np.random.RandomState(seed)

        if config.action_normalized:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(
                low=np.array([-config.steer_limit, 0.0]),
                high=np.array([config.steer_limit, config.speed_max]),
                dtype=np.float32,
            )
        self.obs_dim = config.lidar_rays + 3 + config.lookahead_k
        self.observation_space = spaces.Box(
            -np.inf * np.ones(self.obs_dim * config.n_stack),
            np.inf * np.ones(self.obs_dim * config.n_stack),
            dtype=np.float32,
        )
        self.frames = deque(maxlen=config.n_stack)
        self.last_raw_obs = None
        self._reset_state(target_idx=0, spawn_idx=None)

    # ------------------------------------------------------------------ #
    # Configuration
    # ------------------------------------------------------------------ #

    def set_config(self, config: EnvConfig) -> None:
        """Swap in a new config. Call only at a rollout boundary.

        Reward weights take effect on the next step; spawn settings on the
        next reset. Fields that define the observation/action space or the
        reward mode (:data:`FROZEN_FIELDS`) cannot change.
        """
        changed = [f for f in FROZEN_FIELDS if getattr(config, f) != getattr(self.config, f)]
        if changed:
            raise ValueError(f"cannot change {changed} on a live environment")
        self._check_spawn(config)
        self.config = config

    def _check_spawn(self, config: EnvConfig) -> None:
        bad = [i for i in config.spawn_waypoints if i >= self.track.n_waypoints]
        if bad:
            raise ValueError(f"spawn_waypoints {bad} out of range for track {self.track.name!r}")

    def seed(self, seed: Optional[int] = None):
        """Reseed the spawn curriculum RNG.

        Not forwarded: ``F110Env`` stores its seed as an int attribute that
        shadows ``gym.Env.seed``, and the simulator itself is deterministic.
        """
        self._rng = np.random.RandomState(seed)
        return [seed]

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def _reset_state(self, target_idx: int, spawn_idx: Optional[int]) -> None:
        self.current_waypoint_idx = target_idx
        self.spawn_idx = spawn_idx
        self.prev_steering = 0.0       # previous raw steering (jitter term)
        self.actual_steering = 0.0     # EMA-filtered steering state
        self.prev_distance = None      # legacy: distance to target at previous step
        self.ppo_time_since_last_cp = 0
        self.episode_steps = 0         # policy steps
        self.sim_steps = 0             # simulator steps
        self.laps = 0
        self._lap_start_time = 0.0
        self._s = None                 # laptime: arc length at previous step
        self._seg_hint = None

    def choose_spawn(self) -> Optional[int]:
        """Spawn waypoint for the next episode under the curriculum, or ``None`` for the default start."""
        c = self.config
        # Each check draws from the RNG only when enabled, so configs without
        # random starts keep their exact spawn sequence.
        if c.spawn_random > 0.0 and self._rng.rand() < c.spawn_random:
            return int(self._rng.randint(self.track.n_waypoints))
        if c.spawn_mix > 0.0 and c.spawn_waypoints and self._rng.rand() < c.spawn_mix:
            return int(self._rng.choice(c.spawn_waypoints))
        return None

    def reset(self, spawn_idx: Optional[int] = None, pose: Optional[Sequence[float]] = None, **kwargs):
        """Start an episode.

        Args:
            spawn_idx: Spawn at this waypoint, facing along the line. If
                neither this nor ``pose`` is given, the curriculum decides
                (default start pose ``(0, 0, 0)`` otherwise).
            pose: Explicit ``(x, y, theta)``; overrides ``spawn_idx``. Also
                accepted as ``poses=[[x, y, theta]]`` for compatibility.
        """
        if pose is None and "poses" in kwargs:
            pose = np.asarray(kwargs.pop("poses"), dtype=np.float64).reshape(-1)[:3]
        if pose is None and spawn_idx is None:
            spawn_idx = self.choose_spawn()

        if pose is not None:
            pose = tuple(float(v) for v in pose)
            target = self.track.waypoint_ahead(self.track.project(pose[:2]).s)
        elif spawn_idx is not None:
            pose = self.track.spawn_pose(spawn_idx)
            target = self.track.next_distinct(spawn_idx)
        else:
            pose = LEGACY_START_POSE
            target = 0

        obs, _, _, _ = self.env.reset(poses=np.array([pose]), **kwargs)
        self.last_raw_obs = obs

        if self.config.reward_mode == "legacy":
            # Legacy quirk: the original reset() built the first frame *before*
            # resetting the target waypoint, so its heading error points at the
            # previous episode's target. Reproduced for the default start only.
            legacy_quirk = target == 0 and spawn_idx is None
            if legacy_quirk:
                frame = self.process_obs(obs)
            self._reset_state(target_idx=target, spawn_idx=spawn_idx)
            if not legacy_quirk:
                frame = self.process_obs(obs)
        else:
            self._reset_state(target_idx=target, spawn_idx=spawn_idx)
            proj = self.track.project((obs["poses_x"][0], obs["poses_y"][0]))
            self._s, self._seg_hint = proj.s, proj.segment
            frame = self.process_obs(obs, s=proj.s)

        for _ in range(self.config.n_stack):
            self.frames.append(frame)
        return self._get_stacked_obs()

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def _map_action(self, action):
        """Policy action -> ``(raw_steer, filtered_steer, speed)`` in physical units."""
        c = self.config
        if c.action_normalized:
            a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
            raw_steer = float(a[0]) * c.steer_limit
            self.actual_steering = (c.ema_alpha * raw_steer) + ((1.0 - c.ema_alpha) * self.actual_steering)
            u = float(a[1])
            if c.speed_mid is None:
                speed = c.speed_floor + (u + 1.0) * 0.5 * (c.speed_max - c.speed_floor)
            elif u < 0.0:   # piecewise linear: -1 -> floor, 0 -> mid, +1 -> max
                speed = c.speed_mid + u * (c.speed_mid - c.speed_floor)
            else:
                speed = c.speed_mid + u * (c.speed_max - c.speed_mid)
            return raw_steer, self.actual_steering, speed

        action = np.array(action, copy=True)  # never mutate the caller's array
        raw_steer = action[0]
        # EMA low-pass filter (mechanical damper). Written back into the action
        # array so the value has the action's dtype, exactly as before.
        self.actual_steering = (c.ema_alpha * raw_steer) + ((1.0 - c.ema_alpha) * self.actual_steering)
        action[0] = self.actual_steering
        steer = action[0]
        speed = np.clip(action[1], c.speed_floor, c.speed_max)
        return raw_steer, steer, speed

    def step(self, action):
        c = self.config
        raw_steer, steer, speed = self._map_action(action)

        reward = 0.0
        terms = {}
        cleared = False
        progress_reward = 0.0
        lap_time = None
        termination = None
        done = False
        info = {}
        frame = None

        for _ in range(c.action_repeat):
            obs, _, env_done, info = self.env.step(np.array([[steer, speed]]))
            self.last_raw_obs = obs
            self.sim_steps += 1

            collision = bool(obs["collisions"][0])
            agent_pos = np.array([obs["poses_x"][0], obs["poses_y"][0]])

            # Laps, as counted by the simulator (crossing the spawn pose's start zone).
            lap_event = False
            laps = int(obs["lap_counts"][0])
            if laps > self.laps:
                now = float(getattr(self.env.unwrapped, "current_time", self.sim_steps * 0.01))
                lap_time, self._lap_start_time, self.laps = now - self._lap_start_time, now, laps
                lap_event = True
            max_laps_reached = c.max_laps is not None and laps >= c.max_laps
            lap_complete = (bool(env_done) or max_laps_reached) and not collision

            if c.reward_mode == "legacy":
                frame = self.process_obs(obs)
                r, t, clr, pr, stagnated = self._legacy_step(frame, raw_steer, agent_pos, env_done)
            else:
                proj = self.track.project(agent_pos, hint=self._seg_hint)
                frame = self.process_obs(obs, s=proj.s)
                # float32 round-trip: the same value the observation frame holds.
                velocity = float(np.float32(obs["linear_vels_x"][0]))
                r, t, clr, pr, stagnated = self._laptime_step(
                    proj, velocity, raw_steer, collision, lap_complete, lap_event
                )
            self.prev_steering = raw_steer

            reward += r
            for k, v in t.items():
                terms[k] = terms.get(k, 0.0) + v
            cleared = cleared or clr
            progress_reward += pr

            done = bool(env_done) or stagnated or max_laps_reached
            if collision:
                termination = "collision"
            elif lap_complete:
                termination = "lap_complete"
            elif stagnated:
                termination = "stagnation"
            elif c.max_episode_steps is not None and self.sim_steps >= c.max_episode_steps:
                termination = "time_limit"
                done = True
                info["TimeLimit.truncated"] = True
            if done:
                break

        self.episode_steps += 1
        self.frames.append(frame)
        stacked = self._get_stacked_obs()

        info["termination"] = termination
        info["waypoint_cleared"] = cleared
        info["current_waypoint_idx"] = self.current_waypoint_idx
        info["progress_reward"] = progress_reward
        info["raw_steering"] = float(raw_steer)
        info["steering"] = float(steer)
        info["speed_command"] = float(speed)
        info["reward_terms"] = terms
        info["episode_steps"] = self.episode_steps
        info["sim_steps"] = self.sim_steps
        info["spawn_idx"] = self.spawn_idx
        info["laps"] = self.laps
        if lap_time is not None:
            info["lap_time"] = lap_time
        return stacked, reward, done, info

    def _legacy_step(self, frame, raw_steer, agent_pos, env_done):
        c = self.config
        cleared = False
        progress_reward = 0.0
        target = self.track.waypoints[self.current_waypoint_idx]
        dist = np.linalg.norm(agent_pos - target)
        if self.prev_distance is None:
            self.prev_distance = dist
        progress_reward = (self.prev_distance - dist) * c.progress_coef
        if dist < c.checkpoint_radius:
            self.current_waypoint_idx = (self.current_waypoint_idx + 1) % self.track.n_waypoints
            cleared = True
            progress_reward = 0.0
            self.prev_distance = np.linalg.norm(agent_pos - self.track.waypoints[self.current_waypoint_idx])
        else:
            self.prev_distance = dist

        if cleared:
            self.ppo_time_since_last_cp = 0
        else:
            self.ppo_time_since_last_cp += 1
        stagnated = self.ppo_time_since_last_cp > c.stagnation_steps

        speed_signal = frame[-2] if c.legacy_speed_from_yaw_rate else frame[-3]
        reward, terms = legacy_reward(
            c,
            speed_signal=speed_signal,
            raw_steer=raw_steer,
            prev_raw_steer=self.prev_steering,
            progress_reward=progress_reward,
            waypoint_cleared=cleared,
            env_done=bool(env_done),
            stagnated=stagnated,
        )
        return reward, terms, cleared, progress_reward, stagnated

    def _laptime_step(self, proj, velocity, raw_steer, collision, lap_complete, lap_event):
        c = self.config
        track = self.track
        progress_m = track.progress_delta(self._s, proj.s)
        self._s, self._seg_hint = proj.s, proj.segment

        # Target = first waypoint ahead of the car's projection: a waypoint
        # counts as passed when the car passes it along the line, however far
        # off the line it is (no capture radius to miss).
        new_target = track.waypoint_ahead(proj.s)
        advanced = (new_target - self.current_waypoint_idx) % track.n_waypoints
        cleared_count = advanced if 0 < advanced < track.n_waypoints // 2 else 0
        self.current_waypoint_idx = new_target

        if cleared_count:
            self.ppo_time_since_last_cp = 0
        else:
            self.ppo_time_since_last_cp += 1
        stagnated = self.ppo_time_since_last_cp > c.stagnation_steps and not (collision or lap_complete)

        reward, terms = laptime_reward(
            c,
            velocity=velocity,
            raw_steer=float(raw_steer),
            prev_raw_steer=float(self.prev_steering),
            progress_m=progress_m,
            waypoints_cleared=cleared_count,
            collision=collision,
            stagnated=stagnated,
            lap_complete=lap_event,
        )
        return reward, terms, cleared_count > 0, terms["progress"], stagnated

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def process_obs(self, obs, s: Optional[float] = None) -> np.ndarray:
        """One frame: LiDAR + ``[velocity, yaw_rate, heading_error]`` (+ lookahead curvature).

        ``s`` is the car's arc length on the racing line, needed only for
        the lookahead features (computed if not given).
        """
        c = self.config
        raw_scans = obs["scans"][0]
        step = len(raw_scans) // c.lidar_rays
        scans = raw_scans[::step][: c.lidar_rays]

        velocity = np.array([obs["linear_vels_x"][0]])
        yaw_rate = np.array([obs["ang_vels_z"][0]])

        car_x, car_y, car_yaw = obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]
        target_x, target_y = self.track.waypoints[self.current_waypoint_idx]
        heading_error = math.atan2(target_y - car_y, target_x - car_x) - car_yaw
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
        heading = np.array([heading_error])

        if c.obs_normalize:
            scans = scans / LIDAR_MAX_RANGE
            velocity = velocity / c.speed_max
            yaw_rate = yaw_rate / YAW_RATE_SCALE
            heading = heading / math.pi
        parts = [scans, velocity, yaw_rate, heading]

        if c.lookahead_k:
            if s is None:
                s = self.track.project((car_x, car_y), hint=self._seg_hint).s
            ahead = s + c.lookahead_spacing * np.arange(1, c.lookahead_k + 1)
            curvature = self.track.curvature_at(ahead)
            parts.append(curvature * CURVATURE_SCALE if c.obs_normalize else curvature)

        return np.concatenate(parts).astype(np.float32)

    def _get_stacked_obs(self) -> np.ndarray:
        assert len(self.frames) == self.config.n_stack
        return np.concatenate(self.frames)


def make_env(track="cota", config: EnvConfig = EnvConfig(), seed: Optional[int] = None) -> F1TenthWrapper:
    """Create the simulator for ``track`` (name, spec or :class:`Track`) and wrap it."""
    if not isinstance(track, Track):
        track = load_track(track)
    env = gym.make("f110_gym:f110-v0", map=track.map_stem, map_ext=track.map_ext, num_agents=1)
    return F1TenthWrapper(env, track, config, seed=seed)
