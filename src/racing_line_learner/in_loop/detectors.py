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
# Smoothness thresholds (deterministic driving, straights). A smooth driver corrects its steering at most a few
# times a second, holds the line to within ~0.2 m and changes its speed command gradually. The baseline models
# measure 8-10 reversals/s, 0.38-0.48 m drift and 0.5-1.4 m/s of command change per 0.04 s step.
WEAVING_REVERSALS_PER_S = 4.0
WEAVING_LATERAL_STD_M = 0.25
THROTTLE_CHANGE_PER_STEP = 0.3       # m/s per policy step (7.5 m/s^2)
INCONSISTENT_LINE_STD_M = 0.6        # lateral spread in a corner segment
# Probe failures: the deterministic policy fails at least half its probe laps. Above that, checkpoint noise alone
# (0-8/8 between unchanged 100k checkpoints) makes a "failure" finding unreliable.
PROBE_COMPLETION_MAX = 0.5

#: Which whitelisted levers (schema v2 keys; v1 keys in brackets where they differ) can address each finding.
#: Used by the grounding check: a proposed change must cite a current finding whose lever set contains it.
LEVERS = {
    "crash_hotspot": ("overspeed_coef", "grip_margin", "spawn_corner_mix", "jitter_coef", "steering_penalty_w",
                      "curriculum", "speed_floor", "target_speed"),
    "floor_blocks_braking": ("speed_floor",),
    "speed_pinned_at_floor": ("speed_floor", "time_cost", "progress_coef"),
    "speed_pinned_at_max": ("overspeed_coef", "grip_margin", "time_cost"),
    "stagnation_dominant": ("progress_coef", "time_cost", "ent_coef"),
    "corkscrew": ("steering_penalty_w", "jitter_coef", "progress_coef"),
    "steering_saturated": ("steering_penalty_w", "jitter_coef"),
    "under_driven_segment": ("time_cost", "progress_coef", "lap_bonus", "grip_margin", "target_speed"),
    # Not steering_penalty_w: it penalises steering *magnitude*, so it also limits how hard the car can turn in
    # corners. In the 2026-11-23 arms every LLM seed raised it for weaving and every LLM seed lapped 2-5 s slower than
    # the baseline; the measured fix for weaving is jitter_coef (smaller corrections, lever check 2026-11-15).
    "weaving": ("jitter_coef",),
    "throttle_jitter": ("throttle_change_coef",),
    "inconsistent_line": ("spawn_corner_mix", "jitter_coef", "curriculum"),
    "probe_failures": ("spawn_corner_mix", "overspeed_coef", "grip_margin"),
}


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
            f"{s.get('name', 'segment')} (waypoints {s['waypoints'][0]}-{s['waypoints'][1]}) takes {share:.0%} of the time at up to "
            f"{s['max_speed']:.1f} m/s, though grip allows about {s['grip_limit_min']:.1f} m/s",
            s["waypoints"], time_share=share, max_speed=s["max_speed"], grip_limit=s["grip_limit_min"],
            headroom=round(headroom, 2)))

    findings.extend(_probe_failure_findings(report))
    findings.extend(_smoothness_findings(report))

    order = {"high": 0, "medium": 1, "low": 2}
    return sorted(findings, key=lambda f: order[f["severity"]])


def _probe_failure_findings(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The deterministic policy fails most probe laps: reliability, not smoothness, is the problem."""
    probe = report.get("probe") or {}
    rate, n = probe.get("completion_rate"), probe.get("episodes") or 0
    if rate is None or not n or rate > PROBE_COMPLETION_MAX:
        return []
    fails = probe.get("failures_at") or []
    msg = f"the deterministic probe completes only {round(rate * n)}/{n} laps"
    if fails:
        msg += "; failed laps end at waypoints " + ", ".join(f"{f['waypoint']} (x{f['count']})" for f in fails[:4])
    # Failure points can be scattered round the lap, so no single waypoint range is claimed.
    return [_finding("probe_failures", "high", msg, None, completion_rate=rate, episodes=n, failures_at=fails[:4])]


def _smoothness_findings(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Weaving, throttle jitter and inconsistent lines, from the deterministic probe when it has them.

    Training rollouts are stochastic, so their smoothness mostly measures exploration noise; the probe shows
    how the policy actually drives. Falls back to the training numbers, saying so.
    """
    probe = report.get("probe") or {}
    sm, segs, source = probe.get("smoothness"), probe.get("segments"), "deterministic probe"
    if not sm:
        sm, segs, source = report.get("smoothness"), report.get("segments"), "training rollouts (includes exploration noise)"
    out: List[Dict[str, Any]] = []
    if not sm:
        return out
    rev, lat = sm.get("steer_reversals_per_s"), sm.get("lateral_std_m")
    if rev is not None and lat is not None and rev >= WEAVING_REVERSALS_PER_S and lat >= WEAVING_LATERAL_STD_M:
        out.append(_finding(
            "weaving", "medium",
            f"on straights the steering reverses {rev:.1f} times per second, by {sm.get('steer_change_per_step')} rad "
            f"per step on average, and the car drifts {lat:.2f} m (std) around the racing line ({source})",
            steer_reversals_per_s=rev, steer_change_per_step=sm.get("steer_change_per_step"), lateral_std_m=lat,
            lateral_range_m=sm.get("lateral_range_m"), source=source))
    thr = sm.get("throttle_change_per_step")
    if thr is not None and thr >= THROTTLE_CHANGE_PER_STEP:
        out.append(_finding(
            "throttle_jitter", "medium",
            f"on straights the speed command changes by {thr:.2f} m/s per step on average and reverses "
            f"{sm.get('throttle_reversals_per_s')} times per second ({source})",
            throttle_change_per_step=thr, throttle_reversals_per_s=sm.get("throttle_reversals_per_s"), source=source))
    wobbly = [sg for sg in (segs or []) if sg.get("is_corner") and (sg.get("lateral_std") or 0) >= INCONSISTENT_LINE_STD_M]
    for sg in sorted(wobbly, key=lambda g: -g["lateral_std"])[:3]:
        out.append(_finding(
            "inconsistent_line", "medium",
            f"through corner {sg.get('name', '')} (waypoints {sg['waypoints'][0]}-{sg['waypoints'][1]}) the car's line varies by "
            f"{sg['lateral_std']:.2f} m (std) ({source})",
            sg["waypoints"], lateral_std=sg["lateral_std"], source=source))
    return out


def allowed_levers(findings: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """For each lever, the ids of current findings that justify changing it."""
    out: Dict[str, List[str]] = {}
    for f in findings:
        for key in LEVERS.get(f["id"], ()):
            out.setdefault(key, [])
            if f["id"] not in out[key]:
                out[key].append(f["id"])
    return out
