"""Stateful guardrails for in-loop proposals.

:func:`racing_line_learner.schema.parse_recommendation` checks a proposal on
its own: whitelist, types, absolute ranges and the change cap. This module
adds the checks that depend on the *training state*:

* **Step cap.** A reward weight or ``speed_floor`` may move at most
  ``max_step_fraction`` of its allowed range per window (default 25%), so
  one proposal cannot shock the value function. ``ent_coef``, the
  curriculum and the speed plan are not capped: they do not rescale the
  reward.
* **Cooldown.** A key changed in window ``w`` is judged in window ``w + 1``
  and cannot be changed again until then. If that judgement says the change
  *improved* things, the key may be nudged again straight away, so a large
  correction still takes about two windows. A key whose change was rolled
  back stays locked for ``rollback_cooldown_windows`` more windows.

Both checks can be switched off (``max_step_fraction=None``,
``cooldown_windows=0``) to run them as ablations.
"""

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Mapping, Optional

from ..schema import WHITELIST, Recommendation

#: Keys whose per-window movement is capped: the reward weights and the speed floor.
CAPPED_KEYS: FrozenSet[str] = frozenset(
    {"speed_floor", "target_speed", "progress_coef", "waypoint_bonus", "steering_penalty_w", "jitter_coef"}
)

#: Pseudo-keys used for cooldown bookkeeping of the non-scalar sections.
CURRICULUM_KEY = "curriculum"
SPEED_PLAN_KEY = "speed_plan"

#: Judgement outcomes for a previously applied change.
OUTCOME_PENDING = "pending"          # applied, not yet judged
OUTCOME_IMPROVED = "improved"        # judged better than before: may be nudged again at once
OUTCOME_NEUTRAL = "neutral"          # judged no clear change: normal cooldown applies
OUTCOME_ROLLED_BACK = "rolled_back"  # judged worse and reverted: extended cooldown
OUTCOMES = frozenset({OUTCOME_PENDING, OUTCOME_IMPROVED, OUTCOME_NEUTRAL, OUTCOME_ROLLED_BACK})

_EPS = 1e-9


@dataclass(frozen=True)
class ValidatorConfig:
    """Guardrail settings. ``None`` / ``0`` disables the corresponding check."""

    max_step_fraction: Optional[float] = 0.25
    cooldown_windows: int = 1
    rollback_cooldown_windows: int = 2

    def __post_init__(self) -> None:
        if self.max_step_fraction is not None and not 0.0 < self.max_step_fraction <= 1.0:
            raise ValueError(f"max_step_fraction must be in (0, 1] or None, got {self.max_step_fraction}")
        if self.cooldown_windows < 0 or self.rollback_cooldown_windows < 0:
            raise ValueError("cooldown windows must be >= 0")


@dataclass(frozen=True)
class KeyHistory:
    """The most recent change to one key.

    ``window`` is the window index the change was applied in. ``outcome`` is
    one of :data:`OUTCOMES`; ``judged_window`` is when that outcome was decided
    (for a rollback, the window it was rolled back in).
    """

    window: int
    outcome: str = OUTCOME_PENDING
    judged_window: Optional[int] = None

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise ValueError(f"unknown outcome {self.outcome!r}")


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    errors: List[str] = field(default_factory=list)


def max_step(key: str, config: ValidatorConfig) -> Optional[float]:
    """Largest allowed per-window move for ``key``, or ``None`` if uncapped."""
    if config.max_step_fraction is None or key not in CAPPED_KEYS:
        return None
    spec = WHITELIST[key]
    return config.max_step_fraction * (spec.high - spec.low)


def touched_keys(rec: Recommendation) -> List[str]:
    """Keys (including the curriculum / speed_plan pseudo-keys) that ``rec`` changes."""
    keys = sorted(rec.changes)
    if not rec.curriculum.is_empty():
        keys.append(CURRICULUM_KEY)
    if rec.speed_plan:
        keys.append(SPEED_PLAN_KEY)
    return keys


def _cooldown_error(key: str, hist: KeyHistory, window: int, config: ValidatorConfig) -> Optional[str]:
    if hist.outcome == OUTCOME_ROLLED_BACK:
        rolled = hist.judged_window if hist.judged_window is not None else hist.window
        unlock = rolled + config.rollback_cooldown_windows + 1
        if config.rollback_cooldown_windows > 0 and window < unlock:
            return f"{key}: rolled back in window {rolled}, locked until window {unlock}"
        return None
    if hist.outcome == OUTCOME_IMPROVED:
        return None  # a change that helped may be pushed further at once
    unlock = hist.window + config.cooldown_windows + 1
    if config.cooldown_windows > 0 and window < unlock:
        state = "awaiting judgement" if hist.outcome == OUTCOME_PENDING else "on cooldown"
        return f"{key}: changed in window {hist.window} ({state}), locked until window {unlock}"
    if hist.outcome == OUTCOME_PENDING and config.cooldown_windows > 0:
        return f"{key}: change from window {hist.window} was never judged"
    return None


def validate_proposal(
    rec: Recommendation,
    current: Mapping[str, float],
    window: int,
    history: Optional[Mapping[str, KeyHistory]] = None,
    config: ValidatorConfig = ValidatorConfig(),
) -> ValidationResult:
    """Check a schema-valid proposal against the current training state.

    Args:
        rec: Output of :func:`racing_line_learner.schema.parse_recommendation`.
        current: Values in force now, for every key ``rec.changes`` touches.
        window: Index of the window the proposal would be applied in.
        history: Most recent change per key (see :class:`KeyHistory`).
        config: Guardrail settings.

    Returns:
        A :class:`ValidationResult`; ``errors`` lists every violation. A
        proposal is accepted or rejected as a whole, never partially applied.
    """
    history = history or {}
    errors: List[str] = []

    for key, new in sorted(rec.changes.items()):
        if key not in current:
            errors.append(f"{key}: no current value supplied")
            continue
        limit = max_step(key, config)
        delta = abs(new - current[key])
        if limit is not None and delta > limit + _EPS:
            errors.append(
                f"{key}: move {current[key]:g} -> {new:g} ({delta:g}) exceeds per-window cap {limit:g}"
            )

    for key in touched_keys(rec):
        hist = history.get(key)
        if hist is None:
            continue
        if hist.window >= window:
            errors.append(f"{key}: history window {hist.window} is not before proposal window {window}")
            continue
        err = _cooldown_error(key, hist, window, config)
        if err:
            errors.append(err)

    return ValidationResult(accepted=not errors, errors=errors)


def default_values() -> Dict[str, float]:
    """Whitelist defaults, i.e. the values in force before any intervention."""
    return {k: s.default for k, s in WHITELIST.items()}
