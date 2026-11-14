"""Deterministic findings from a telemetry report.

Each detector turns report numbers into a stated fact the proposer (rules or
LLM) can act on, so the LLM is never asked to spot patterns in raw numbers.
A finding is a plain dict::

    {"id": "crash_hotspot", "severity": "high", "waypoints": [110, 112],
     "message": "...", "facts": {...}}

Thresholds are module constants so tests and the report can cite them.
"""

from typing import Any, Dict, List

HOTSPOT_MIN_SHARE = 0.2          # a hotspot must hold this share of all collisions
OVERSPEED_MARGIN = 1.05          # approach speed above grip_limit * margin counts as overspeed
PINNED_SHARE = 0.8               # speed command at floor/max this often counts as pinned
HEADROOM_MIN = 5.0               # m/s of unused grip on a segment to call it under-driven
STAGNATION_SHARE = 0.3
SATURATION_SHARE = 0.5
CORKSCREW_PROGRESS_FRACTION = 0.05   # progress reward per step below this fraction of the reward per step


def _finding(fid: str, severity: str, message: str, waypoints=None, **facts) -> Dict[str, Any]:
    return {"id": fid, "severity": severity, "waypoints": waypoints, "message": message, "facts": facts}


def detect(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """All findings for one report, most severe first."""
    findings: List[Dict[str, Any]] = []
    speed = report["speed"]
    floor = speed["floor"]
    pinned_floor = (speed["pinned_at_floor_share"] or 0.0) >= PINNED_SHARE
    pinned_max = (speed["pinned_at_max_share"] or 0.0) >= PINNED_SHARE

    for h in report["crash_hotspots"]:
        if (h["share"] or 0.0) < HOTSPOT_MIN_SHARE:
            continue
        grip, approach = h["grip_limit"], h["approach_speed"]
        overspeed = grip is not None and approach is not None and approach > grip * OVERSPEED_MARGIN
        msg = f"{h['share']:.0%} of collisions occur at waypoints {h['waypoints'][0]}-{h['waypoints'][1]}"
        if overspeed:
            msg += f"; approach speed {approach:.1f} m/s exceeds the corner's grip limit of about {grip:.1f} m/s"
        findings.append(_finding("crash_hotspot", "high", msg, h["waypoints"], collisions=h["collisions"],
                                 share=h["share"], approach_speed=approach, grip_limit=grip, overspeed=overspeed))
        if grip is not None and floor > grip * OVERSPEED_MARGIN:
            findings.append(_finding(
                "floor_blocks_braking", "high",
                f"the speed floor ({floor:.1f} m/s) is above the grip limit at waypoints "
                f"{h['waypoints'][0]}-{h['waypoints'][1]} (about {grip:.1f} m/s): the car cannot slow enough",
                h["waypoints"], speed_floor=floor, grip_limit=grip))

    if pinned_floor:
        findings.append(_finding(
            "speed_pinned_at_floor", "high",
            f"the speed command sits at the floor ({floor:.1f} m/s) {speed['pinned_at_floor_share']:.0%} of the time; "
            f"lap time cannot beat about {speed['floor_limited_lap_time_s']:.1f} s this way",
            share=speed["pinned_at_floor_share"], floor=floor, floor_limited_lap_time_s=speed["floor_limited_lap_time_s"]))
    if pinned_max:
        findings.append(_finding(
            "speed_pinned_at_max", "high",
            f"the speed command sits at the maximum ({speed['max']:.1f} m/s) {speed['pinned_at_max_share']:.0%} "
            f"of the time; the policy is not braking for corners",
            share=speed["pinned_at_max_share"]))

    terms = report["terminations"]
    stag = terms.get("stagnation", {}).get("share") or 0.0
    if stag >= STAGNATION_SHARE:
        findings.append(_finding("stagnation_dominant", "medium",
                                 f"{stag:.0%} of episodes end by stagnation (no waypoint progress)", share=stag))

    sat = report["steering"]["saturation_share"] or 0.0
    rps = report["reward_per_step"]
    total_rps = sum(v for v in rps.values() if v is not None)
    progress_rps = (rps.get("progress") or 0.0) + (rps.get("bonus") or 0.0)
    if total_rps > 0 and progress_rps < CORKSCREW_PROGRESS_FRACTION * total_rps and sat >= SATURATION_SHARE:
        findings.append(_finding(
            "corkscrew", "high",
            f"reward per step is positive ({total_rps:.3f}) while progress earns almost none of it and steering is "
            f"saturated {sat:.0%} of the time: the policy is farming reward without driving the lap",
            reward_per_step=total_rps, progress_per_step=progress_rps, steer_saturation=sat))
    elif sat >= SATURATION_SHARE:
        findings.append(_finding("steering_saturated", "medium",
                                 f"steering is at full lock {sat:.0%} of the time", share=sat))

    # Under-driven segments: lots of unused grip and no crashes. Most time first.
    under = []
    for s in report["segments"]:
        if s["mean_speed"] is None or s["max_speed"] is None or s["collisions"]:
            continue
        headroom = s["grip_limit_min"] - s["max_speed"]
        if headroom >= HEADROOM_MIN:
            under.append((s["time_share"], s, headroom))
    for share, s, headroom in sorted(under, key=lambda t: -t[0])[:3]:
        findings.append(_finding(
            "under_driven_segment", "medium",
            f"waypoints {s['waypoints'][0]}-{s['waypoints'][1]} take {share:.0%} of the time at up to "
            f"{s['max_speed']:.1f} m/s, though grip allows about {s['grip_limit_min']:.1f} m/s",
            s["waypoints"], time_share=share, max_speed=s["max_speed"], grip_limit=s["grip_limit_min"],
            headroom=round(headroom, 2)))

    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(findings, key=lambda f: order[f["severity"]])
