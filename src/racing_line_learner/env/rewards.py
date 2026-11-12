"""Reward functions, kept pure (no simulator state) so they can be tested on fixed inputs.

Each returns ``(reward, terms)`` where ``terms`` breaks the reward into named
parts for telemetry. For ``legacy_reward`` the terms describe the
components *before* the legacy override logic; ``reward`` is always the
exact value the agent receives.
"""

import math
from typing import Dict, Tuple

import numpy as np

from ..config import EnvConfig


def gaussian_speed(value: float, config: EnvConfig) -> float:
    return np.exp(-((value - config.target_speed) ** 2) / (2 * config.speed_variance ** 2))


def legacy_reward(
    config: EnvConfig,
    *,
    speed_signal: float,
    raw_steer: float,
    prev_raw_steer: float,
    progress_reward: float,
    waypoint_cleared: bool,
    env_done: bool,
    stagnated: bool,
) -> Tuple[float, Dict[str, float]]:
    """The original ``wrapper.py`` PPO reward, operation for operation.

    Args:
        speed_signal: Value fed to the Gaussian. In the original code this is
            observation index -2, i.e. the yaw rate (see ``config.py``).
        progress_reward: ``progress_coef * distance closed`` toward the
            target waypoint (0 on the step a waypoint is cleared).
        env_done: The simulator's done flag (collision or lap completion).
        stagnated: Stagnation limit exceeded on this step.
    """
    steering_diff = abs(raw_steer - prev_raw_steer)
    jitter_penalty = steering_diff * config.jitter_coef
    speed_reward = gaussian_speed(speed_signal, config)

    if env_done:
        reward = -config.terminal_penalty
    else:
        reward = (speed_reward * config.speed_reward_w) - jitter_penalty
    reward += progress_reward
    if waypoint_cleared:
        reward += config.waypoint_bonus
    steering_penalty = (abs(raw_steer) ** 2) * config.steering_penalty_w
    reward -= steering_penalty
    if stagnated:
        reward = -config.terminal_penalty

    terms = {
        "speed": 0.0 if env_done else float(speed_reward * config.speed_reward_w),
        "jitter": 0.0 if env_done else -float(jitter_penalty),
        "progress": float(progress_reward),
        "bonus": float(config.waypoint_bonus) if waypoint_cleared else 0.0,
        "steering": -float(steering_penalty),
        "time": 0.0,
        "terminal": -float(config.terminal_penalty) if (env_done or stagnated) else 0.0,
        "lap": 0.0,
    }
    return float(reward), terms


def laptime_reward(
    config: EnvConfig,
    *,
    velocity: float,
    raw_steer: float,
    prev_raw_steer: float,
    progress_m: float,
    waypoints_cleared: int,
    collision: bool,
    stagnated: bool,
    lap_complete: bool,
) -> Tuple[float, Dict[str, float]]:
    """Lap-time reward: paid for progress along the line, charged for time.

    A step's reward is ``progress + bonus + speed - time - steering - jitter``.
    Collision or stagnation replaces the whole step reward with
    ``-terminal_penalty``; lap completion adds ``lap_bonus``.

    Args:
        velocity: Actual longitudinal velocity (m/s).
        progress_m: Signed arc-length progress along the racing line this step (m).
        waypoints_cleared: Waypoints passed this step (forward only).
    """
    terms = {
        "progress": config.progress_coef * progress_m,
        "bonus": config.waypoint_bonus * waypoints_cleared,
        "speed": config.speed_reward_w * float(gaussian_speed(velocity, config)) if config.speed_reward_w else 0.0,
        "time": -config.time_cost,
        "steering": -(raw_steer ** 2) * config.steering_penalty_w,
        "jitter": -abs(raw_steer - prev_raw_steer) * config.jitter_coef,
        "terminal": 0.0,
        "lap": 0.0,
    }
    if collision or stagnated:
        terms = {k: 0.0 for k in terms}
        terms["terminal"] = -config.terminal_penalty
    elif lap_complete:
        terms["lap"] = config.lap_bonus
    terms = {k: float(v) for k, v in terms.items()}
    reward = math.fsum(terms.values())
    return reward, terms
