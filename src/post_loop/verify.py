"""Step 4: check an LLM recommendation against the recorded data.

Ananth's rule (ananth_notes.txt, step 4): every number the LLM cites must
match the recorded data, every waypoint index must exist, and every change
must be on the whitelist and within range. Otherwise the recommendation is
rejected.

Checks, in order:
  1. Ananth's strict loader (``parse_recommendation``): format, waypoint
     indices in [0, num_waypoints), whitelist, ranges, at most 2 changes.
     It must also be for this run's track and source model.
  2. Each diagnosis must cover a real recorded failure (a crash, missed
     waypoint or stuck lap, or a crash segment) at the waypoints it names.
  3. Each evidence number must match a recorded value with the same field
     name, taken from the failures that diagnosis covers:
       * ``<field>`` matches any one of the recorded values (or their mean);
       * ``<field>_min`` / ``_max`` / ``_mean`` match the min / max / mean;
       * true/false values count as 1/0;
       * the tolerance allows for rounding: max(0.01, 1% of the value).
     A field name that isn't in the data is rejected: it can't be checked.
  4. A recorded failure that no diagnosis covers is a WARNING, not a
     rejection (the recommendation can still be applied).

A recommendation that passes gets ``verified: true`` on every diagnosis.

Standard library only (runs in the Python 3.8 ``f1tenth`` env).

Usage (from the repo root):
    python -m src.post_loop.verify --run post_loop_runs/<folder>
        # checks every trial in <run>/llm_results.json and writes the first
        # one that passes to <run>/recommendations.json
    python -m src.post_loop.verify --run post_loop_runs/<folder> --input <file.json>
        # <file.json> is either an llm_results file or one recommendation
"""
import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from src.racing_line_learner.schema import SchemaError, parse_recommendation

#: Rounding tolerance: a cited number may differ from the data by this much.
ABS_TOL = 0.01
REL_TOL = 0.01

#: Suffixes the LLM may add to a field name, and how to aggregate for them.
AGGREGATES = {
    "_min": min,
    "_max": max,
    "_mean": lambda xs: sum(xs) / len(xs),
}

#: Event lists in analysis.json that each describe one failure on one lap.
EVENT_KINDS = ("crashes", "missed_waypoints", "stuck")
EVENT_NAMES = {"crashes": "crash", "missed_waypoints": "missed waypoint", "stuck": "stuck lap"}


@dataclass
class VerifyResult:
    """Outcome of :func:`verify`. ``recommendation`` is set only when ``ok``."""

    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendation: Optional[Dict[str, Any]] = None


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def waypoint_range(first: int, last: int, n_waypoints: int) -> Set[int]:
    """Waypoints in [first, last]; wraps past the finish line if first > last."""
    if first <= last:
        return set(range(first, last + 1))
    return set(range(first, n_waypoints)) | set(range(0, last + 1))


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def _add_numbers(pool: Dict[str, List[float]], record: Dict[str, Any]) -> None:
    """Add every number field of ``record`` to ``pool`` (nested dicts flattened)."""
    for key, value in record.items():
        if isinstance(value, dict):
            _add_numbers(pool, value)
            continue
        number = _as_number(value)
        if number is not None:
            pool.setdefault(key, []).append(number)


def close(cited: float, recorded: float) -> bool:
    return abs(cited - recorded) <= max(ABS_TOL, REL_TOL * abs(recorded))


def failures_in(analysis: Dict[str, Any], wps: Set[int]) -> Dict[str, List[Dict[str, Any]]]:
    """The recorded failures (events and crash segments) at any waypoint in ``wps``."""
    found = {kind: [e for e in analysis.get(kind, []) if e.get("waypoint") in wps]
             for kind in EVENT_KINDS}
    n = analysis["track_info"]["num_waypoints"]
    found["crash_segments"] = [
        s for s in analysis.get("crash_segments", [])
        if waypoint_range(s["waypoints"][0], s["waypoints"][1], n) & wps
    ]
    return found


def evidence_pool(found: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[float]]:
    """Field name -> recorded values, over the given failures."""
    pool: Dict[str, List[float]] = {}
    for kind in EVENT_KINDS:
        for event in found[kind]:
            _add_numbers(pool, event)
    for seg in found["crash_segments"]:
        _add_numbers(pool, {k: v for k, v in seg.items() if k not in ("waypoints", "laps")})
        for lap in seg.get("laps", []):
            pool.setdefault("lap", []).append(float(lap))
    return pool


def check_number(key: str, cited: float, pool: Dict[str, List[float]]) -> Optional[str]:
    """None if ``cited`` matches the data under ``key``; otherwise the reason it doesn't."""
    if key in pool:
        values = pool[key]
        if any(close(cited, v) for v in values) or close(cited, AGGREGATES["_mean"](values)):
            return None
        return f"{cited} does not match the recorded {key} ({_fmt(values)})"
    for suffix, agg in AGGREGATES.items():
        base = key[:-len(suffix)]
        if key.endswith(suffix) and base in pool:
            target = agg(pool[base])
            if close(cited, target):
                return None
            return (f"{cited} does not match the {suffix[1:]} of the recorded {base} "
                    f"({target:g}, from {_fmt(pool[base])})")
    return "field not found in the recorded data at these waypoints, so it can't be checked"


def _fmt(values: Iterable[float]) -> str:
    uniq = sorted(set(values))
    return ", ".join(f"{v:g}" for v in uniq[:8]) + (", ..." if len(uniq) > 8 else "")


# --------------------------------------------------------------------------
# Main check
# --------------------------------------------------------------------------

def verify(rec: Any, analysis: Dict[str, Any]) -> VerifyResult:
    """Check one recommendation (a decoded JSON object) against ``analysis``."""
    n = analysis["track_info"]["num_waypoints"]
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Format, waypoint indices, whitelist, ranges, change cap (Ananth's loader).
    try:
        parse_recommendation(rec, n, expected_track=analysis["track"])
    except SchemaError as exc:
        errors.extend(exc.errors)
    if isinstance(rec, dict) and rec.get("source_model") != analysis["source_model"]:
        errors.append(f"source_model: recommendation is for {rec.get('source_model')!r}, "
                      f"expected {analysis['source_model']!r}")
    if errors:
        return VerifyResult(False, errors, warnings)

    # 2 + 3. Each diagnosis covers a real failure, and its numbers match the data.
    covered: Set[int] = set()
    for i, diag in enumerate(rec.get("diagnoses", [])):
        path = f"diagnoses[{i}]"
        first, last = diag["waypoints"]
        wps = waypoint_range(first, last, n)
        found = failures_in(analysis, wps)
        if not any(found.values()):
            errors.append(f"{path}: nothing failed at waypoints [{first}, {last}] in the recorded laps")
            continue
        covered |= wps
        pool = evidence_pool(found)
        for key, cited in diag.get("evidence", {}).items():
            problem = check_number(key, float(cited), pool)
            if problem:
                errors.append(f"{path}.evidence.{key}: {problem}")

    # 4. Failures nobody explained: warn only.
    for kind in EVENT_KINDS:
        for event in analysis.get(kind, []):
            if event.get("waypoint") not in covered:
                warnings.append(f"no diagnosis covers the {EVENT_NAMES[kind]} "
                                f"at waypoint {event.get('waypoint')} (lap {event.get('lap')})")
    warnings = sorted(set(warnings))

    if errors:
        return VerifyResult(False, errors, warnings)
    verified = json.loads(json.dumps(rec))
    for diag in verified.get("diagnoses", []):
        diag["verified"] = True
    return VerifyResult(True, errors, warnings, verified)


def load_candidates(path: Path) -> List[Optional[Dict[str, Any]]]:
    """Recommendations in ``path``: an llm_results file (one per trial) or a single one."""
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "trials" in data:
        return [t.get("recommendation") for t in data["trials"]]
    return [data]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", required=True, help="folder with analysis.json (from analyze.py)")
    p.add_argument("--input", default=None,
                   help="llm_results file or one recommendation (default: <run>/llm_results.json)")
    p.add_argument("--out", default=None,
                   help="where to write the first verified recommendation "
                        "(default: <run>/recommendations.json)")
    p.add_argument("--no-write", action="store_true", help="only report, write nothing")
    args = p.parse_args(argv)

    run = Path(args.run)
    analysis = json.loads((run / "analysis.json").read_text())
    src = Path(args.input) if args.input else run / "llm_results.json"
    candidates = load_candidates(src)

    first_ok = None
    for i, rec in enumerate(candidates):
        label = f"[{i + 1}/{len(candidates)}]"
        if rec is None:
            print(f"{label} REJECTED: the LLM gave no valid recommendation")
            continue
        r = verify(rec, analysis)
        diags = ", ".join(f"{d['waypoints']} {d['cause']}" for d in rec.get("diagnoses", []))
        print(f"{label} {'PASSED' if r.ok else 'REJECTED'}: {diags or 'no diagnoses'}")
        for e in r.errors:
            print(f"      error: {e}")
        for w in r.warnings:
            print(f"      warning: {w}")
        if r.ok and first_ok is None:
            first_ok = r.recommendation

    n_ok = sum(1 for rec in candidates if rec is not None and verify(rec, analysis).ok)
    print(f"\n{n_ok}/{len(candidates)} recommendation(s) passed")
    if first_ok is None:
        return 1
    if not args.no_write:
        out = Path(args.out) if args.out else run / "recommendations.json"
        out.write_text(json.dumps(first_ok, indent=2) + "\n")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
