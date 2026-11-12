"""Shared ``recommendations.json`` schema for the in-loop and post-loop agents.

Both reasoning agents (the in-loop supervisor and the post-loop diagnoser)
emit the same document. Training consumes it via ``--config``. The loader is
strict: unknown keys, wrong types, non-finite numbers and out-of-range values
are all rejected, and every problem is reported at once, so a caller (e.g. an
LLM retry loop) can see the full list.

Only ``changes``, ``curriculum`` and ``speed_plan`` are ever applied.
``diagnoses`` and ``rationale`` are kept for logging and verification.

Example document::

    {
      "schema_version": 1,
      "source_model": "ppo_cota_run3_5M",
      "track": "cota",
      "diagnoses": [
        {"waypoints": [108, 114], "cause": "blind_corner",
         "evidence": {"crash_rate": 0.8, "mean_speed": 9.4},
         "verified": true}
      ],
      "changes": {"target_speed": 8.5},
      "curriculum": {"spawn_waypoints": [104], "spawn_mix": 0.3},
      "speed_plan": {"104": 8.0, "108": 6.8},
      "rationale": "short text, logged only"
    }

Waypoint index bounds always come from the loaded track (``n_waypoints``),
never from a hard-coded constant.

This module is standard-library only so either agent can import it without
pulling in the simulator or SB3.
"""

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

SCHEMA_VERSION = 1

#: Default cap on applied changes per recommendation, so effects stay
#: attributable. Each ``changes`` key counts as one; a non-empty
#: ``curriculum`` counts as one; a non-empty ``speed_plan`` counts as one.
DEFAULT_MAX_CHANGES = 2

#: Absolute upper bound for any ``speed_plan`` value (simulator v_max).
SPEED_PLAN_MAX = 20.0

#: Upper bound for ``curriculum.spawn_mix``.
SPAWN_MIX_MAX = 0.5

#: Upper bound on ``rationale`` length, to keep logs (and prompts) small.
RATIONALE_MAX_CHARS = 2000


@dataclass(frozen=True)
class ParamSpec:
    """A tunable parameter: its baseline value and the allowed range."""

    default: float
    low: float
    high: float


#: Whitelist of tunable parameters. ``default`` mirrors the values currently
#: hard-coded in ``wrapper.py`` / ``train.py`` on ``stable``.
WHITELIST: Dict[str, ParamSpec] = {
    "speed_floor": ParamSpec(default=6.5, low=6.5, high=8.0),
    "target_speed": ParamSpec(default=10.0, low=6.0, high=12.0),
    "progress_coef": ParamSpec(default=10.0, low=5.0, high=20.0),
    "waypoint_bonus": ParamSpec(default=50.0, low=20.0, high=100.0),
    "steering_penalty_w": ParamSpec(default=2.0, low=0.5, high=3.0),
    "jitter_coef": ParamSpec(default=0.05, low=0.01, high=0.2),
    "ent_coef": ParamSpec(default=0.01, low=0.001, high=0.05),
}

_TOP_LEVEL_REQUIRED = ("schema_version", "source_model", "track")
_TOP_LEVEL_OPTIONAL = ("diagnoses", "changes", "curriculum", "speed_plan", "rationale")
_DIAGNOSIS_REQUIRED = ("waypoints", "cause")
_DIAGNOSIS_OPTIONAL = ("evidence", "verified")
_CURRICULUM_KEYS = ("spawn_waypoints", "spawn_mix")


class SchemaError(ValueError):
    """Raised when a recommendation fails validation.

    ``errors`` holds every problem found, as ``"path: message"`` strings.
    """

    def __init__(self, errors: List[str]):
        self.errors = list(errors)
        super().__init__("invalid recommendation:\n  - " + "\n  - ".join(self.errors))


@dataclass(frozen=True)
class Diagnosis:
    """A claimed failure cause over a waypoint index range (logged, not applied)."""

    waypoints: Tuple[int, int]
    cause: str
    evidence: Dict[str, float] = field(default_factory=dict)
    verified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "waypoints": list(self.waypoints),
            "cause": self.cause,
            "evidence": dict(self.evidence),
            "verified": self.verified,
        }


@dataclass(frozen=True)
class Curriculum:
    """Spawn curriculum: a fraction ``spawn_mix`` of episodes start at ``spawn_waypoints``."""

    spawn_waypoints: Tuple[int, ...] = ()
    spawn_mix: float = 0.0

    def is_empty(self) -> bool:
        return not self.spawn_waypoints and self.spawn_mix == 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"spawn_waypoints": list(self.spawn_waypoints), "spawn_mix": self.spawn_mix}


@dataclass(frozen=True)
class Recommendation:
    """A validated recommendation. Build it with :func:`parse_recommendation`."""

    source_model: str
    track: str
    changes: Dict[str, float] = field(default_factory=dict)
    curriculum: Curriculum = field(default_factory=Curriculum)
    speed_plan: Dict[int, float] = field(default_factory=dict)
    diagnoses: Tuple[Diagnosis, ...] = ()
    rationale: str = ""
    schema_version: int = SCHEMA_VERSION

    @property
    def n_changes(self) -> int:
        """Number of applied changes, as counted against ``max_changes``."""
        return count_changes(self.changes, self.curriculum, self.speed_plan)

    def is_noop(self) -> bool:
        """True if applying this recommendation changes nothing."""
        return self.n_changes == 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize back to the on-disk JSON layout (speed_plan keys as strings)."""
        return {
            "schema_version": self.schema_version,
            "source_model": self.source_model,
            "track": self.track,
            "diagnoses": [d.to_dict() for d in self.diagnoses],
            "changes": dict(self.changes),
            "curriculum": self.curriculum.to_dict(),
            "speed_plan": {str(k): v for k, v in sorted(self.speed_plan.items())},
            "rationale": self.rationale,
        }


def count_changes(
    changes: Mapping[str, float], curriculum: Curriculum, speed_plan: Mapping[int, float]
) -> int:
    """Count applied changes: one per ``changes`` key, one for a curriculum, one for a speed plan."""
    return len(changes) + (0 if curriculum.is_empty() else 1) + (1 if speed_plan else 0)


# --------------------------------------------------------------------------- #
# Type helpers
# --------------------------------------------------------------------------- #

def _is_number(value: Any) -> bool:
    # bool is a subclass of int; reject it explicitly.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_number(value: Any, path: str, errors: List[str]) -> Optional[float]:
    if not _is_number(value):
        errors.append(f"{path}: expected a number, got {type(value).__name__}")
        return None
    if not math.isfinite(value):
        errors.append(f"{path}: must be finite, got {value}")
        return None
    return float(value)


def _check_waypoint(value: Any, path: str, n_waypoints: int, errors: List[str]) -> Optional[int]:
    if not _is_int(value):
        errors.append(f"{path}: expected an integer waypoint index, got {type(value).__name__}")
        return None
    if not 0 <= value < n_waypoints:
        errors.append(f"{path}: waypoint index {value} out of range [0, {n_waypoints - 1}]")
        return None
    return value


def _check_keys(
    obj: Mapping[str, Any], required: Tuple[str, ...], optional: Tuple[str, ...], path: str,
    errors: List[str],
) -> None:
    for key in required:
        if key not in obj:
            errors.append(f"{path}{key}: required key missing")
    for key in obj:
        if key not in required and key not in optional:
            errors.append(f"{path}{key}: unknown key")


# --------------------------------------------------------------------------- #
# Section parsers
# --------------------------------------------------------------------------- #

def _parse_changes(raw: Any, errors: List[str]) -> Dict[str, float]:
    if not isinstance(raw, dict):
        errors.append(f"changes: expected an object, got {type(raw).__name__}")
        return {}
    out: Dict[str, float] = {}
    for key, value in raw.items():
        path = f"changes.{key}"
        spec = WHITELIST.get(key)
        if spec is None:
            errors.append(f"{path}: not a tunable parameter (allowed: {', '.join(sorted(WHITELIST))})")
            continue
        number = _check_number(value, path, errors)
        if number is None:
            continue
        if not spec.low <= number <= spec.high:
            errors.append(f"{path}: {number} out of range [{spec.low}, {spec.high}]")
            continue
        out[key] = number
    return out


def _parse_curriculum(raw: Any, n_waypoints: int, errors: List[str]) -> Curriculum:
    if not isinstance(raw, dict):
        errors.append(f"curriculum: expected an object, got {type(raw).__name__}")
        return Curriculum()
    _check_keys(raw, (), _CURRICULUM_KEYS, "curriculum.", errors)

    spawn: List[int] = []
    raw_spawn = raw.get("spawn_waypoints", [])
    if not isinstance(raw_spawn, list):
        errors.append(f"curriculum.spawn_waypoints: expected a list, got {type(raw_spawn).__name__}")
    else:
        for i, item in enumerate(raw_spawn):
            idx = _check_waypoint(item, f"curriculum.spawn_waypoints[{i}]", n_waypoints, errors)
            if idx is not None:
                spawn.append(idx)
        if len(set(spawn)) != len(spawn):
            errors.append("curriculum.spawn_waypoints: duplicate indices")

    mix = 0.0
    if "spawn_mix" in raw:
        number = _check_number(raw["spawn_mix"], "curriculum.spawn_mix", errors)
        if number is not None:
            if not 0.0 <= number <= SPAWN_MIX_MAX:
                errors.append(f"curriculum.spawn_mix: {number} out of range [0.0, {SPAWN_MIX_MAX}]")
            else:
                mix = number

    if spawn and mix == 0.0:
        errors.append("curriculum: spawn_waypoints given but spawn_mix is 0 (curriculum would never fire)")
    if mix > 0.0 and isinstance(raw_spawn, list) and not raw_spawn:
        errors.append("curriculum: spawn_mix > 0 requires at least one spawn waypoint")

    return Curriculum(spawn_waypoints=tuple(sorted(set(spawn))), spawn_mix=mix)


def _parse_speed_plan(
    raw: Any, n_waypoints: int, speed_floor: float, errors: List[str]
) -> Dict[int, float]:
    if not isinstance(raw, dict):
        errors.append(f"speed_plan: expected an object, got {type(raw).__name__}")
        return {}
    out: Dict[int, float] = {}
    for key, value in raw.items():
        path = f"speed_plan.{key}"
        # JSON object keys are always strings; accept "104" (and int 104 from Python callers).
        if _is_int(key):
            idx_raw: Any = key
        elif isinstance(key, str) and key.strip().lstrip("-").isdigit():
            idx_raw = int(key)
        else:
            errors.append(f"{path}: key must be a waypoint index, got {key!r}")
            continue
        idx = _check_waypoint(idx_raw, path, n_waypoints, errors)
        number = _check_number(value, path, errors)
        if idx is None or number is None:
            continue
        if not speed_floor <= number <= SPEED_PLAN_MAX:
            errors.append(f"{path}: {number} out of range [{speed_floor}, {SPEED_PLAN_MAX}]")
            continue
        if idx in out:
            errors.append(f"{path}: duplicate waypoint index {idx}")
            continue
        out[idx] = number
    return out


def _parse_diagnoses(raw: Any, n_waypoints: int, errors: List[str]) -> Tuple[Diagnosis, ...]:
    if not isinstance(raw, list):
        errors.append(f"diagnoses: expected a list, got {type(raw).__name__}")
        return ()
    out: List[Diagnosis] = []
    for i, item in enumerate(raw):
        path = f"diagnoses[{i}]"
        if not isinstance(item, dict):
            errors.append(f"{path}: expected an object, got {type(item).__name__}")
            continue
        n_before = len(errors)
        _check_keys(item, _DIAGNOSIS_REQUIRED, _DIAGNOSIS_OPTIONAL, f"{path}.", errors)

        wps = item.get("waypoints")
        start = end = None
        if not (isinstance(wps, list) and len(wps) == 2):
            if "waypoints" in item:
                errors.append(f"{path}.waypoints: expected [start, end]")
        else:
            # start > end is allowed: the range wraps past the start/finish line.
            start = _check_waypoint(wps[0], f"{path}.waypoints[0]", n_waypoints, errors)
            end = _check_waypoint(wps[1], f"{path}.waypoints[1]", n_waypoints, errors)

        cause = item.get("cause")
        if "cause" in item and not (isinstance(cause, str) and cause.strip()):
            errors.append(f"{path}.cause: expected a non-empty string")

        evidence: Dict[str, float] = {}
        raw_ev = item.get("evidence", {})
        if not isinstance(raw_ev, dict):
            errors.append(f"{path}.evidence: expected an object, got {type(raw_ev).__name__}")
        else:
            for k, v in raw_ev.items():
                number = _check_number(v, f"{path}.evidence.{k}", errors)
                if number is not None:
                    evidence[str(k)] = number

        verified = item.get("verified", False)
        if not isinstance(verified, bool):
            errors.append(f"{path}.verified: expected true/false, got {type(verified).__name__}")

        if len(errors) == n_before and start is not None and end is not None:
            out.append(Diagnosis((start, end), cause.strip(), evidence, verified))
    return tuple(out)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def parse_recommendation(
    data: Any,
    n_waypoints: int,
    speed_floor: Optional[float] = None,
    max_changes: int = DEFAULT_MAX_CHANGES,
    expected_track: Optional[str] = None,
) -> Recommendation:
    """Validate a decoded ``recommendations.json`` document.

    Args:
        data: The decoded JSON object.
        n_waypoints: Number of waypoints in the loaded track; all indices must
            lie in ``[0, n_waypoints)``.
        speed_floor: Speed floor currently in force, the lower bound for
            ``speed_plan`` values. If the recommendation itself changes
            ``speed_floor``, the new value is used. Defaults to the whitelist
            default (6.5).
        max_changes: Cap on applied changes (see :func:`count_changes`).
        expected_track: If given, ``track`` must equal it.

    Returns:
        A frozen :class:`Recommendation`.

    Raises:
        SchemaError: listing every validation problem found.
    """
    if not _is_int(n_waypoints) or n_waypoints < 1:
        raise ValueError(f"n_waypoints must be a positive integer, got {n_waypoints!r}")
    if not isinstance(data, dict):
        raise SchemaError([f"<root>: expected an object, got {type(data).__name__}"])

    errors: List[str] = []
    _check_keys(data, _TOP_LEVEL_REQUIRED, _TOP_LEVEL_OPTIONAL, "", errors)

    version = data.get("schema_version")
    if "schema_version" in data and (not _is_int(version) or version != SCHEMA_VERSION):
        errors.append(f"schema_version: unsupported version {version!r} (expected {SCHEMA_VERSION})")

    strings = {}
    for key in ("source_model", "track"):
        value = data.get(key)
        if key in data and not (isinstance(value, str) and value.strip()):
            errors.append(f"{key}: expected a non-empty string")
        strings[key] = value.strip() if isinstance(value, str) else ""
    if expected_track is not None and strings["track"] and strings["track"] != expected_track:
        errors.append(f"track: recommendation is for {strings['track']!r}, expected {expected_track!r}")

    changes = _parse_changes(data.get("changes", {}), errors)

    floor = changes.get("speed_floor")
    if floor is None:
        floor = WHITELIST["speed_floor"].default if speed_floor is None else float(speed_floor)

    curriculum = _parse_curriculum(data.get("curriculum", {}), n_waypoints, errors)
    speed_plan = _parse_speed_plan(data.get("speed_plan", {}), n_waypoints, floor, errors)
    diagnoses = _parse_diagnoses(data.get("diagnoses", []), n_waypoints, errors)

    rationale = data.get("rationale", "")
    if not isinstance(rationale, str):
        errors.append(f"rationale: expected a string, got {type(rationale).__name__}")
        rationale = ""
    elif len(rationale) > RATIONALE_MAX_CHARS:
        errors.append(f"rationale: {len(rationale)} chars exceeds limit of {RATIONALE_MAX_CHARS}")

    n = count_changes(changes, curriculum, speed_plan)
    if n > max_changes:
        errors.append(f"<root>: {n} changes proposed, at most {max_changes} allowed per recommendation")

    if errors:
        raise SchemaError(errors)

    return Recommendation(
        source_model=strings["source_model"],
        track=strings["track"],
        changes=changes,
        curriculum=curriculum,
        speed_plan=speed_plan,
        diagnoses=diagnoses,
        rationale=rationale,
    )


def load_recommendation(path: str, n_waypoints: int, **kwargs: Any) -> Recommendation:
    """Read and validate a ``recommendations.json`` file. See :func:`parse_recommendation`."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise SchemaError([f"<file {path}>: not valid JSON ({exc})"]) from exc
    return parse_recommendation(data, n_waypoints, **kwargs)


def dump_recommendation(rec: Recommendation, path: str) -> None:
    """Write a recommendation as pretty-printed JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rec.to_dict(), f, indent=2)
        f.write("\n")


def recommendation_json_schema(n_waypoints: int, max_changes: int = DEFAULT_MAX_CHANGES) -> Dict[str, Any]:
    """JSON Schema for constrained LLM decoding (llama.cpp / vLLM guided JSON).

    It encodes types, key whitelists and static ranges. Cross-field rules
    (the change cap, ``speed_plan`` vs. a changed ``speed_floor``,
    spawn_mix/spawn_waypoints consistency) cannot be expressed here, so
    output must still go through :func:`parse_recommendation`.
    """
    last = n_waypoints - 1
    waypoint = {"type": "integer", "minimum": 0, "maximum": last}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(_TOP_LEVEL_REQUIRED),
        "properties": {
            "schema_version": {"type": "integer", "enum": [SCHEMA_VERSION]},
            "source_model": {"type": "string", "minLength": 1},
            "track": {"type": "string", "minLength": 1},
            "diagnoses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(_DIAGNOSIS_REQUIRED),
                    "properties": {
                        "waypoints": {"type": "array", "items": waypoint, "minItems": 2, "maxItems": 2},
                        "cause": {"type": "string", "minLength": 1},
                        "evidence": {"type": "object", "additionalProperties": {"type": "number"}},
                        "verified": {"type": "boolean"},
                    },
                },
            },
            "changes": {
                "type": "object",
                "additionalProperties": False,
                "maxProperties": max_changes,
                "properties": {
                    k: {"type": "number", "minimum": s.low, "maximum": s.high}
                    for k, s in WHITELIST.items()
                },
            },
            "curriculum": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "spawn_waypoints": {"type": "array", "items": waypoint, "uniqueItems": True},
                    "spawn_mix": {"type": "number", "minimum": 0.0, "maximum": SPAWN_MIX_MAX},
                },
            },
            "speed_plan": {
                "type": "object",
                "propertyNames": {"pattern": "^[0-9]+$"},
                "additionalProperties": {
                    "type": "number",
                    "minimum": WHITELIST["speed_floor"].low,
                    "maximum": SPEED_PLAN_MAX,
                },
            },
            "rationale": {"type": "string", "maxLength": RATIONALE_MAX_CHARS},
        },
    }
