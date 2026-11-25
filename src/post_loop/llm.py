"""Step 3: ask an LLM to turn analysis.json into a recommendation.

The LLM reads a compact summary of analysis.json (step 2) and replies with
one ``recommendations.json`` document in Ananth's shared format
(``src/racing_line_learner/schema.py``). Three layers keep the reply in shape:

  1. JSON-schema mode: the request carries a JSON schema, and the server
     (llama.cpp, vLLM, ...) constrains decoding to it, so the reply is always
     parseable JSON with the right keys and types.
  2. ``parse_recommendation`` (Ananth's strict loader) then checks every
     range and cross-field rule. Some of these can't be written as a JSON
     schema, and llama.cpp does not enforce number (non-integer) ranges.
  3. If the loader rejects the reply, its error list goes back to the LLM
     and it gets another try (``max_retries``).

Backends are swappable. Each one only has to turn (messages, json_schema)
into a reply string:
  * ``DummyBackend``: no model; simple rules over analysis.json. For tests
    and as a baseline.
  * ``OpenAICompatBackend``: any server with an OpenAI-style
    ``/v1/chat/completions`` endpoint, given by URL (llama.cpp's
    ``llama-server``, vLLM, or a hosted API later).

Standard library only, so it runs in the Python 3.8 ``f1tenth`` env or any
newer Python, with no extra packages.

Usage (from the repo root; start llama-server on port 8081 first, see
colab_setup.sh. Port 8080 is taken by Colab itself, so 8081 is the default):
    python -m src.post_loop.llm --run post_loop_runs/<folder> \\
        --expect 146:misplaced_waypoint --expect 44:spin --trials 3
"""
import argparse
import copy
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.racing_line_learner.schema import (
    DEFAULT_MAX_CHANGES,
    SCHEMA_VERSION,
    SPAWN_MIX_MAX,
    SPEED_PLAN_MAX,
    WHITELIST,
    SchemaError,
    parse_recommendation,
    recommendation_json_schema,
)

#: Causes the post-loop agent may name. Ananth's schema allows any string;
#: a fixed list makes answers comparable and lets us score them.
CAUSES = {
    "misplaced_waypoint": "the car drives the corner, but keeps passing just outside one "
                          "waypoint's capture radius and drives on, so that waypoint is never "
                          "cleared. The waypoint is in the wrong place, not the driving",
    "spin": "the car loses control: its heading turns a large amount (about 90 degrees or "
            "more) in the last second before it crashes",
    "late_braking": "the wall was visible in time, but the car was too fast to stop or turn",
    "blind_corner": "the wall only became visible too late to stop (visibility check: blind)",
    "stagnation": "the car stops making progress without passing its target waypoint",
    "other": "none of the above fits",
}

SPEED_FLOOR_NOTE = ("The wrapper clamps the commanded speed to at least speed_floor "
                    "(6.5 m/s), so the car can never slow below it.")


# --------------------------------------------------------------------------
# Prompt and schema
# --------------------------------------------------------------------------

def summarize_analysis(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Compact, LLM-friendly view of analysis.json.

    Only renames fields and groups repeated missed-waypoint events; every
    number is copied unchanged, so step 4 can check citations against it.
    """
    s = analysis["summary"]
    crashes = []
    for c in analysis.get("crashes", []):
        v = c.get("visibility", {})
        crashes.append({
            "lap": c["lap"], "mode": c["mode"], "waypoint": c["waypoint"],
            "speed": c["speed"], "steer_raw": c["steer_raw"],
            "steer_smoothed": c["steer_smoothed"],
            "corner_radius_m": c["corner_radius_m"],
            "heading_change_deg": c["heading_change_deg"],
            "visible_m": v.get("visible_m"), "required_brake_m": v.get("required_brake_m"),
            "blind": v.get("blind"),
        })
    missed: Dict[int, Dict[str, Any]] = {}
    for m in analysis.get("missed_waypoints", []):
        g = missed.setdefault(m["waypoint"], {"waypoint": m["waypoint"], "laps": [],
                                              "closest_approach_m": [],
                                              "capture_radius_m": m["capture_radius_m"],
                                              "distance_driven_after_m": [],
                                              "mean_speed_last_2s": []})
        g["laps"].append(m["lap"])
        g["closest_approach_m"].append(m["closest_approach_m"])
        g["distance_driven_after_m"].append(m["distance_driven_after_m"])
        g["mean_speed_last_2s"].append(m["mean_speed_last_2s"])
    for g in missed.values():
        for k in ("closest_approach_m", "distance_driven_after_m", "mean_speed_last_2s"):
            g[k + "_min"], g[k + "_max"] = min(g[k]), max(g[k])
            del g[k]
    stuck = [{"lap": e.get("lap"), "waypoint": e.get("waypoint")}
             for e in analysis.get("stuck", [])]
    return {
        "track": analysis["track"],
        "source_model": analysis["source_model"],
        "num_waypoints": analysis["track_info"]["num_waypoints"],
        "laps": s["laps"],
        "terminations": s["terminations"],
        "completion_rate": s["completion_rate"],
        "mean_waypoints_cleared": s["mean_waypoints_cleared"],
        "crashes": crashes,
        "crash_segments": [{k: seg[k] for k in ("waypoints", "crashes", "laps_reaching",
                                                 "crash_rate", "mean_speed", "blind_fraction")}
                           for seg in analysis.get("crash_segments", [])],
        "missed_waypoints": list(missed.values()),
        "stuck": stuck,
    }


def build_json_schema(analysis: Dict[str, Any],
                      max_changes: int = DEFAULT_MAX_CHANGES) -> Dict[str, Any]:
    """Ananth's schema, tightened for the post-loop agent.

    Extra rules (all still valid under ``parse_recommendation``): source_model
    and track are fixed to this run, ``diagnoses`` must be non-empty and use a
    cause from CAUSES, and ``verified`` is always false (step 4 sets it).
    Every top-level key is required so the model can't skip a section.
    """
    schema = copy.deepcopy(recommendation_json_schema(
        analysis["track_info"]["num_waypoints"], max_changes))
    props = schema["properties"]
    props["source_model"] = {"type": "string", "enum": [analysis["source_model"]]}
    props["track"] = {"type": "string", "enum": [analysis["track"]]}
    props["diagnoses"]["minItems"] = 1
    diag = props["diagnoses"]["items"]
    diag["properties"]["cause"] = {"type": "string", "enum": list(CAUSES)}
    diag["properties"]["verified"] = {"type": "boolean", "enum": [False]}
    diag["required"] = ["waypoints", "cause", "evidence", "verified"]
    schema["required"] = list(props)
    return schema


def build_messages(analysis: Dict[str, Any],
                   max_changes: int = DEFAULT_MAX_CHANGES) -> List[Dict[str, str]]:
    whitelist = "\n".join(f"  - {k}: current {s.default}, allowed {s.low} to {s.high}"
                          for k, s in WHITELIST.items())
    causes = "\n".join(f"  - {k}: {v}." for k, v in CAUSES.items())
    system = f"""You are the post-training analyst for an F1TENTH racing car trained with PPO.
You read lap telemetry analysis and write ONE JSON recommendation for the next training run.

How a lap works: the car must pass within the capture radius of each waypoint, in order, to
clear it. A lap ends on a crash, or after 2000 steps (20 s) without clearing a waypoint.
{SPEED_FLOOR_NOTE}

Causes you may use:
{causes}

Rules:
  - Explain EVERY failure in the data: write a diagnosis for every crash in "crashes" and every
    entry in "missed_waypoints" and "stuck", even one that happened on a single lap. A rare
    failure still needs its own diagnosis; don't drop it because another failure is more common.
    Different failures need separate diagnoses (a crash and a missed waypoint never share one).
  - Write one diagnosis per distinct failure location. waypoints is [first, last] index of the
    affected stretch ([w, w] for a single waypoint).
  - evidence: only numbers copied exactly from the data, using the data's field names.
  - verified: always false (a separate checker sets it).
  - Settings you may change (at most {max_changes} changes in total; each key in "changes"
    counts as one, a non-empty curriculum counts as one, a non-empty speed_plan counts as one):
{whitelist}
  - curriculum: spawn_waypoints are indices 0-{analysis['track_info']['num_waypoints'] - 1},
    spawn_mix 0 to {SPAWN_MIX_MAX}; use both or neither.
  - speed_plan: waypoint index (as a string) -> target speed, from speed_floor to {SPEED_PLAN_MAX}.
  - None of these settings can move a waypoint. If a failure is caused by the track data
    itself, diagnose it, but do not propose a change for it; say in the rationale that the
    waypoint file needs fixing.
  - Prefer no change over a change you can't justify from the data. Empty objects are fine.
  - rationale: a few short sentences.
  - schema_version is {SCHEMA_VERSION}."""
    summary = summarize_analysis(analysis)
    todo = ([f"crash at waypoint {c['waypoint']} (lap {c['lap']})" for c in summary["crashes"]]
            + [f"missed waypoint {m['waypoint']} (laps {m['laps']})"
               for m in summary["missed_waypoints"]]
            + [f"stuck at waypoint {s['waypoint']} (lap {s['lap']})" for s in summary["stuck"]])
    checklist = ("Failures to diagnose (each needs its own diagnosis):\n- " + "\n- ".join(todo)
                 if todo else "No failures were recorded.")
    user = ("Analysis of the evaluation laps:\n"
            + json.dumps(summary, indent=1)
            + "\n\n" + checklist
            + "\n\nReply with the recommendation JSON only.")
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

class Backend:
    """Turns chat messages + a JSON schema into a reply string."""

    name = "base"

    def complete(self, messages: List[Dict[str, str]],
                 json_schema: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Return (reply text, metadata such as token counts and timings)."""
        raise NotImplementedError


class DummyBackend(Backend):
    """No model. Rule-based answer built from the analysis in the prompt.

    Rules: every missed waypoint -> misplaced_waypoint; a crash whose heading
    turned >= 90 deg -> spin; a blind crash -> blind_corner; any other crash
    -> late_braking. It proposes no changes.
    """

    name = "dummy"

    def complete(self, messages, json_schema):
        # The JSON (no blank lines inside) sits between the header line and the checklist.
        data = json.loads(messages[1]["content"].split("\n", 1)[1].split("\n\n", 1)[0])
        diagnoses = []
        for m in data["missed_waypoints"]:
            w = m["waypoint"]
            diagnoses.append({"waypoints": [w, w], "cause": "misplaced_waypoint",
                              "evidence": {"closest_approach_m_min": m["closest_approach_m_min"],
                                           "capture_radius_m": m["capture_radius_m"]},
                              "verified": False})
        for c in data["crashes"]:
            w = c["waypoint"]
            if c["heading_change_deg"] >= 90:
                cause = "spin"
            elif c["blind"]:
                cause = "blind_corner"
            else:
                cause = "late_braking"
            diagnoses.append({"waypoints": [w, w], "cause": cause,
                              "evidence": {"heading_change_deg": c["heading_change_deg"],
                                           "speed": c["speed"]},
                              "verified": False})
        if not diagnoses:
            diagnoses.append({"waypoints": [0, 0], "cause": "other", "evidence": {},
                              "verified": False})
        reply = {"schema_version": SCHEMA_VERSION, "source_model": data["source_model"],
                 "track": data["track"], "diagnoses": diagnoses, "changes": {},
                 "curriculum": {}, "speed_plan": {},
                 "rationale": "Rule-based dummy backend; no changes proposed."}
        return json.dumps(reply), {}


class OpenAICompatBackend(Backend):
    """Any OpenAI-compatible chat server, by URL (llama.cpp, vLLM, hosted APIs).

    Sends the schema as ``response_format = {"type": "json_schema", ...}``,
    which llama-server turns into a decoding grammar and vLLM into guided
    decoding.
    """

    name = "openai"

    def __init__(self, base_url: str = "http://127.0.0.1:8081/v1", model: str = "local",
                 api_key: Optional[str] = None, temperature: float = 0.0,
                 max_tokens: int = 1024, timeout_s: float = 300.0, seed: Optional[int] = None):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.seed = seed

    def complete(self, messages, json_schema):
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "recommendation", "strict": True,
                                                "schema": json_schema}},
        }
        if self.seed is not None:
            body["seed"] = self.seed
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base_url + "/chat/completions",
                                     data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                out = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"LLM server returned HTTP {exc.code}: "
                               f"{exc.read().decode(errors='replace')[:500]}") from exc
        choice = out["choices"][0]
        meta = {"finish_reason": choice.get("finish_reason"), "usage": out.get("usage", {})}
        if "timings" in out:   # llama-server extra: prompt / generation speed
            t = out["timings"]
            meta["timings"] = {k: t[k] for k in ("prompt_ms", "predicted_ms",
                                                 "predicted_per_second") if k in t}
        return choice["message"]["content"], meta


def make_backend(name: str, **kwargs) -> Backend:
    if name == "dummy":
        return DummyBackend()
    if name == "openai":
        return OpenAICompatBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r} (choose dummy or openai)")


# --------------------------------------------------------------------------
# Ask, validate, retry
# --------------------------------------------------------------------------

def recommend(analysis: Dict[str, Any], backend: Backend, max_retries: int = 2,
              max_changes: int = DEFAULT_MAX_CHANGES) -> Dict[str, Any]:
    """Ask the backend for a recommendation, validating and retrying on errors.

    Returns a dict with ``recommendation`` (validated document, or None if
    every attempt failed), ``attempts`` (each raw reply, its errors, timing),
    and ``total_s``.
    """
    n_wp = analysis["track_info"]["num_waypoints"]
    schema = build_json_schema(analysis, max_changes)
    messages = build_messages(analysis, max_changes)
    attempts = []
    t_start = time.perf_counter()
    for _ in range(max_retries + 1):
        t0 = time.perf_counter()
        text, meta = backend.complete(messages, schema)
        attempt = {"seconds": round(time.perf_counter() - t0, 2), "reply": text, **meta}
        attempts.append(attempt)
        try:
            data = json.loads(text)
            rec = parse_recommendation(data, n_wp, max_changes=max_changes,
                                       expected_track=analysis["track"])
        except json.JSONDecodeError as exc:
            errors = [f"<reply>: not valid JSON ({exc})"]
        except SchemaError as exc:
            errors = exc.errors
        else:
            attempt["errors"] = []
            return {"recommendation": rec.to_dict(), "attempts": attempts,
                    "total_s": round(time.perf_counter() - t_start, 2)}
        attempt["errors"] = errors
        messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": "That recommendation was rejected:\n- "
             + "\n- ".join(errors) + "\nReply with a corrected recommendation JSON only."}]
    return {"recommendation": None, "attempts": attempts,
            "total_s": round(time.perf_counter() - t_start, 2)}


def score(recommendation: Optional[Dict[str, Any]],
          expected: Dict[int, str]) -> Dict[int, Dict[str, Any]]:
    """For each expected waypoint -> cause, which causes did the LLM give it?

    A diagnosis covers waypoint w if w lies in its [first, last] range
    (ranges with first > last wrap past the start line). Correct means the
    expected cause is given and no other cause is.
    """
    result = {}
    diags = recommendation["diagnoses"] if recommendation else []
    for wp, want in expected.items():
        got = []
        for d in diags:
            a, b = d["waypoints"]
            if (a <= wp <= b) if a <= b else (wp >= a or wp <= b):
                got.append(d["cause"])
        result[wp] = {"expected": want, "got": got, "correct": set(got) == {want}}
    return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_expect(items: List[str]) -> Dict[int, str]:
    out = {}
    for item in items or []:
        wp, cause = item.split(":", 1)
        if cause not in CAUSES:
            raise SystemExit(f"--expect {item}: unknown cause (choose from {', '.join(CAUSES)})")
        out[int(wp)] = cause
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", required=True, help="folder with analysis.json (from analyze.py)")
    p.add_argument("--backend", choices=["dummy", "openai"], default="openai")
    p.add_argument("--url", default="http://127.0.0.1:8081/v1",
                   help="base URL of an OpenAI-compatible server")
    p.add_argument("--model", default="local", help="model name sent to the server")
    p.add_argument("--api-key", default=None)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--trials", type=int, default=1, help="ask this many times (timing / consistency)")
    p.add_argument("--expect", action="append", metavar="WP:CAUSE",
                   help="known answer to score against, e.g. 146:misplaced_waypoint")
    p.add_argument("--out", default=None, help="where to write results (default: <run>/llm_results.json)")
    args = p.parse_args(argv)

    run = Path(args.run)
    analysis = json.loads((run / "analysis.json").read_text())
    expected = parse_expect(args.expect)
    kwargs = {}
    if args.backend == "openai":
        kwargs = dict(base_url=args.url, model=args.model, api_key=args.api_key,
                      temperature=args.temperature)

    trials = []
    for i in range(args.trials):
        backend = make_backend(args.backend, **kwargs)
        if args.backend == "openai" and args.temperature > 0:
            backend.seed = i     # different but reproducible samples per trial
        r = recommend(analysis, backend, max_retries=args.max_retries)
        r["score"] = score(r["recommendation"], expected)
        trials.append(r)
        n_att = len(r["attempts"])
        ok = "valid" if r["recommendation"] else "INVALID"
        gen = r["attempts"][-1].get("usage", {}).get("completion_tokens")
        print(f"trial {i + 1}: {r['total_s']:.1f} s, {n_att} attempt(s), {ok}"
              + (f", {gen} tokens generated" if gen else ""))
        for wp, s in r["score"].items():
            print(f"    wp {wp}: expected {s['expected']}, got {s['got'] or 'nothing'} -> "
                  + ("CORRECT" if s["correct"] else "WRONG"))
        if r["recommendation"]:
            print("    changes:", json.dumps(r["recommendation"]["changes"]),
                  "curriculum:", json.dumps(r["recommendation"]["curriculum"]),
                  "speed_plan:", json.dumps(r["recommendation"]["speed_plan"]))
            print("    rationale:", r["recommendation"]["rationale"])

    out = Path(args.out) if args.out else run / "llm_results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"backend": args.backend, "url": args.url if args.backend == "openai" else None,
                               "model": args.model, "temperature": args.temperature,
                               "expected": {str(k): v for k, v in expected.items()},
                               "trials": trials}, indent=2))
    if expected:
        n_ok = sum(all(s["correct"] for s in t["score"].values()) for t in trials)
        print(f"\nall expected answers correct in {n_ok}/{len(trials)} trial(s)")
    times = [t["total_s"] for t in trials]
    print(f"time per answer: min {min(times):.1f} s, max {max(times):.1f} s, "
          f"mean {sum(times) / len(times):.1f} s")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
