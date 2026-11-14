"""Configuration: environment/reward settings, presets and output locations.

Every constant that used to be hard-coded in ``wrapper.py`` lives in
:class:`EnvConfig`. Three presets are defined:

``legacy``
    Bit-for-bit the reward of the original ``wrapper.py`` (``algo='ppo'``),
    including its known quirks, so old checkpoints and results stay
    reproducible:

    * the "speed" term reads observation index -2, which is the **yaw rate**,
      not the velocity (index -3), so it rewards spinning at ~10 rad/s;
    * any ``done`` (collision *or* finishing the simulator's 2-lap episode)
      sets reward to -10, after which progress, bonus and steering terms are
      still added;
    * the first observation after a reset points its heading error at the
      previous episode's target waypoint;
    * the steering action space is +/-3.14 rad although the simulator
      saturates at +/-0.4189 rad.

``laptime``
    Reward aimed at the real objective, the fastest completed lap:
    arc-length progress along the racing line, a per-step time cost, a
    lap-completion bonus, and terminal penalties that replace (not add to)
    the step reward. Episodes end after one lap, so each gives one lap time.
    The steering range matches the simulator, and the speed reward reads the
    actual velocity (it is off by default).

``v2``
    ``laptime`` plus the efficiency and policy-side fixes for the in-loop arm:

    * **normalised actions** in [-1, 1], mapped to the steering limit and to
      ``[speed_floor, speed_max]``, so PPO's unit-std exploration matches the
      usable range;
    * **action repeat 4**: the policy acts at 25 Hz instead of 100 Hz, so
      gamma=0.99 looks ~4 s ahead instead of ~1 s (enough to plan braking);
    * **speed floor 2 m/s** instead of 6.5: the racing line has corners whose
      grip limit is below 6.5 m/s. The floor existed only to stop the
      corkscrew, which the yaw-rate bug caused;
    * **no EMA steering filter**: the simulator already rate-limits steering,
      and with action repeat the filter mostly adds lag;
    * **lookahead**: curvature of the racing line 2, 4, ..., 16 m ahead;
    * **normalised observations** (fixed scales, no running statistics);
    * **slow neutral speed** (``speed_mid=5``: action 0 means 5 m/s) and a
      **crash penalty of 1000**. With progress paid per metre, a short
      horizon makes "flat out, then crash" score better than surviving;
      a large penalty plus gamma=0.998 (see ``PRESET_HPARAMS`` in
      ``train/ppo.py``) makes braking worth learning;
    * **random starts** for 75% of episodes (``spawn_random``): early
      episodes are short, so starting every one from the grid means the
      policy only ever practises the first corners.

    Ablations are expressed as overrides, e.g. ``--env speed_floor=6.5`` or
    ``--env lookahead_k=0``.
"""

import json
import os
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

#: Repository root (``src/racing_line_learner/config.py`` -> three levels up).
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Environment variable selecting where runs, checkpoints and logs are written.
OUTPUT_DIR_ENV = "RLL_OUTPUT_DIR"

#: Simulator steering limit (f110_gym ``s_max``), in radians.
SIM_STEER_LIMIT = 0.4189

REWARD_MODES = ("legacy", "laptime")


@dataclass(frozen=True)
class EnvConfig:
    """Environment, action-space and reward settings for :class:`F1TenthWrapper`."""

    reward_mode: str = "legacy"

    # --- Observation (fixed for the life of a model) ---
    lidar_rays: int = 20
    n_stack: int = 3
    lookahead_k: int = 0               # racing-line curvature samples ahead of the car (laptime mode)
    lookahead_spacing: float = 2.0     # metres between lookahead samples
    obs_normalize: bool = False        # scale observation features to roughly [-1, 1]

    # --- Action mapping ---
    action_normalized: bool = False    # actions in [-1, 1] mapped to steering/speed ranges
    action_repeat: int = 1             # simulator steps per policy action
    steer_limit: float = 3.14          # action-space bound, radians
    speed_max: float = 20.0            # action-space bound and clip ceiling, m/s
    speed_floor: float = 6.5           # minimum commanded speed (anti-corkscrew in legacy)
    speed_mid: Optional[float] = None  # normalised actions: speed for action 0 (None = midpoint of range)
    ema_alpha: float = 0.3             # steering low-pass filter: s = a*raw + (1-a)*s (1.0 = off)

    # --- Reward terms (both modes) ---
    target_speed: float = 10.0         # centre of the Gaussian speed reward
    speed_variance: float = 2.0        # Gaussian width (std), m/s
    speed_reward_w: float = 2.0        # weight of the Gaussian speed reward
    progress_coef: float = 10.0        # reward per metre of progress
    waypoint_bonus: float = 50.0       # reward per waypoint cleared
    steering_penalty_w: float = 2.0    # weight on raw_steer**2
    jitter_coef: float = 0.05          # weight on |raw_steer - prev_raw_steer|
    terminal_penalty: float = 10.0     # reward on collision / stagnation

    # --- laptime mode only ---
    time_cost: float = 0.0             # subtracted every step
    lap_bonus: float = 0.0             # added on lap completion

    # --- Legacy quirk switches (only read in legacy mode) ---
    legacy_speed_from_yaw_rate: bool = True

    # --- Episode control ---
    checkpoint_radius: float = 1.5     # legacy waypoint capture radius, metres
    stagnation_steps: int = 2000       # simulator steps (0.01 s each) without clearing a waypoint
    max_episode_steps: Optional[int] = None  # time limit in simulator steps (None = no limit)
    max_laps: Optional[int] = None     # end after this many laps (None = simulator default: 2 laps)

    # --- Spawn curriculum ---
    spawn_waypoints: Tuple[int, ...] = ()
    spawn_mix: float = 0.0             # fraction of resets that spawn at a random spawn_waypoint
    spawn_random: float = 0.0          # fraction of resets that spawn at a uniformly random waypoint

    def __post_init__(self) -> None:
        errors = []
        if self.reward_mode not in REWARD_MODES:
            errors.append(f"reward_mode must be one of {REWARD_MODES}, got {self.reward_mode!r}")
        if self.lidar_rays < 1 or self.n_stack < 1:
            errors.append("lidar_rays and n_stack must be >= 1")
        if not 0.0 < self.steer_limit <= 3.14:
            errors.append(f"steer_limit must be in (0, 3.14], got {self.steer_limit}")
        if not 0.0 <= self.speed_floor <= self.speed_max:
            errors.append(f"need 0 <= speed_floor <= speed_max, got {self.speed_floor}, {self.speed_max}")
        if not 0.0 < self.ema_alpha <= 1.0:
            errors.append(f"ema_alpha must be in (0, 1], got {self.ema_alpha}")
        if self.speed_variance <= 0.0:
            errors.append("speed_variance must be > 0")
        if self.checkpoint_radius <= 0.0:
            errors.append("checkpoint_radius must be > 0")
        if self.stagnation_steps < 1:
            errors.append("stagnation_steps must be >= 1")
        if self.max_episode_steps is not None and self.max_episode_steps < 1:
            errors.append("max_episode_steps must be >= 1 or None")
        if self.max_laps is not None and not 1 <= self.max_laps <= 2:
            errors.append("max_laps must be 1, 2 or None (the simulator ends episodes after 2 laps)")
        if not 0.0 <= self.spawn_mix <= 1.0:
            errors.append(f"spawn_mix must be in [0, 1], got {self.spawn_mix}")
        if not 0.0 <= self.spawn_random <= 1.0:
            errors.append(f"spawn_random must be in [0, 1], got {self.spawn_random}")
        if self.spawn_mix > 0.0 and not self.spawn_waypoints:
            errors.append("spawn_mix > 0 requires spawn_waypoints")
        if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in self.spawn_waypoints):
            errors.append("spawn_waypoints must be non-negative integers")
        if self.action_repeat < 1:
            errors.append("action_repeat must be >= 1")
        if self.lookahead_k < 0 or self.lookahead_spacing <= 0.0:
            errors.append("lookahead_k must be >= 0 and lookahead_spacing > 0")
        if self.action_normalized and self.speed_floor >= self.speed_max:
            errors.append("normalised actions need speed_floor < speed_max")
        if self.speed_mid is not None and not self.speed_floor < self.speed_mid < self.speed_max:
            errors.append(f"speed_mid ({self.speed_mid}) must lie strictly between speed_floor ({self.speed_floor}) "
                          f"and speed_max ({self.speed_max}); when raising the floor, set speed_mid too")
        if self.reward_mode == "legacy":
            # The legacy reward reads raw observation values and assumes one
            # simulator step per action; keep that mode exactly as it was.
            v2_only = {"action_normalized": False, "action_repeat": 1, "lookahead_k": 0, "obs_normalize": False}
            bad = [k for k, v in v2_only.items() if getattr(self, k) != v]
            if bad:
                errors.append(f"{bad} are only supported with reward_mode='laptime'")
        if errors:
            raise ValueError("invalid EnvConfig: " + "; ".join(errors))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["spawn_waypoints"] = list(self.spawn_waypoints)
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EnvConfig":
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown EnvConfig keys: {unknown}")
        kwargs = dict(data)
        if "spawn_waypoints" in kwargs:
            kwargs["spawn_waypoints"] = tuple(kwargs["spawn_waypoints"])
        return cls(**kwargs)


#: Fields that change the observation or action space, or the reward
#: definition; they cannot be changed on a live environment.
FROZEN_FIELDS = (
    "reward_mode", "lidar_rays", "n_stack", "steer_limit", "speed_max", "lookahead_k", "lookahead_spacing",
    "obs_normalize", "action_normalized", "action_repeat", "speed_mid",
)

_LAPTIME = EnvConfig(
    reward_mode="laptime",
    steer_limit=SIM_STEER_LIMIT,
    speed_reward_w=0.0,
    legacy_speed_from_yaw_rate=False,
    time_cost=0.1,
    lap_bonus=100.0,
    max_laps=1,
)

PRESETS: Dict[str, EnvConfig] = {
    "legacy": EnvConfig(),
    "laptime": _LAPTIME,
    "v2": replace(
        _LAPTIME,
        action_normalized=True,
        action_repeat=4,
        speed_floor=2.0,
        speed_mid=5.0,
        terminal_penalty=1000.0,
        spawn_random=0.75,
        ema_alpha=1.0,
        lookahead_k=8,
        lookahead_spacing=2.0,
        obs_normalize=True,
    ),
}

#: Fixed observation scales used when ``obs_normalize`` is on.
LIDAR_MAX_RANGE = 30.0      # f110_gym ScanSimulator2D default
YAW_RATE_SCALE = 10.0       # rad/s
CURVATURE_SCALE = 2.0       # 1/m -> ~1 at the tightest COTA corners (R ~ 2 m)


def parse_overrides(items) -> Dict[str, Any]:
    """Parse ``["key=value", ...]`` (values as JSON, else plain strings) into a dict of EnvConfig fields."""
    known = {f.name for f in fields(EnvConfig)}
    out: Dict[str, Any] = {}
    for item in items or ():
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like key=value")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in known:
            raise ValueError(f"unknown EnvConfig field {key!r}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        if isinstance(value, list):
            value = tuple(value)
        out[key] = value
    return out


def get_preset(name: str) -> EnvConfig:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}") from None


def apply_recommendation(config: EnvConfig, rec: Any) -> Tuple[EnvConfig, Dict[str, float]]:
    """Apply a validated :class:`~racing_line_learner.schema.Recommendation`.

    Returns the new environment config and the algorithm-level overrides
    (currently only ``ent_coef``), which the trainer applies to the model.
    ``speed_plan`` is accepted by the schema but not consumed yet (Phase 3);
    callers should warn if it is non-empty.
    """
    env_changes = {k: v for k, v in rec.changes.items() if k != "ent_coef"}
    algo = {k: v for k, v in rec.changes.items() if k == "ent_coef"}
    if not rec.curriculum.is_empty():
        env_changes["spawn_waypoints"] = tuple(rec.curriculum.spawn_waypoints)
        env_changes["spawn_mix"] = rec.curriculum.spawn_mix
    return replace(config, **env_changes), algo


def output_root(cli_value: Optional[str] = None) -> Path:
    """Where runs are written: ``--output_dir``, else ``$RLL_OUTPUT_DIR``, else the repo root."""
    value = cli_value or os.environ.get(OUTPUT_DIR_ENV)
    return Path(value).expanduser().resolve() if value else REPO_ROOT


def save_json(data: Mapping[str, Any], path: Path) -> None:
    """Write JSON atomically (temp file + rename), so a disconnect never leaves a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
