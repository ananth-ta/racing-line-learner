"""Configuration: environment/reward settings, presets and output locations.

Every constant that used to be hard-coded in ``wrapper.py`` lives in
:class:`EnvConfig`. Two presets are defined:

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

    # --- Action mapping ---
    steer_limit: float = 3.14          # action-space bound, radians
    speed_max: float = 20.0            # action-space bound and clip ceiling, m/s
    speed_floor: float = 6.5           # commanded speed is clipped to >= this (anti-corkscrew)
    ema_alpha: float = 0.3             # steering low-pass filter: s = a*raw + (1-a)*s

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
    stagnation_steps: int = 2000       # end episode after this many steps without clearing a waypoint
    max_episode_steps: Optional[int] = None  # time limit (None = no limit, as before)
    max_laps: Optional[int] = None     # end after this many laps (None = simulator default: 2 laps)

    # --- Spawn curriculum ---
    spawn_waypoints: Tuple[int, ...] = ()
    spawn_mix: float = 0.0             # fraction of resets that spawn at a random spawn_waypoint

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
        if self.spawn_mix > 0.0 and not self.spawn_waypoints:
            errors.append("spawn_mix > 0 requires spawn_waypoints")
        if any(not isinstance(i, int) or isinstance(i, bool) or i < 0 for i in self.spawn_waypoints):
            errors.append("spawn_waypoints must be non-negative integers")
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
FROZEN_FIELDS = ("reward_mode", "lidar_rays", "n_stack", "steer_limit", "speed_max")

PRESETS: Dict[str, EnvConfig] = {
    "legacy": EnvConfig(),
    "laptime": EnvConfig(
        reward_mode="laptime",
        steer_limit=SIM_STEER_LIMIT,
        speed_reward_w=0.0,
        legacy_speed_from_yaw_rate=False,
        time_cost=0.1,
        lap_bonus=100.0,
        max_laps=1,
    ),
}


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
