"""LLM proposer for the in-loop supervisor.

The model never sees raw telemetry tables. It gets what the rule-based tuner
gets: the probe summary, the detector findings, and for each lever its
current value, range, the largest move allowed this window, whether it is
locked, and which current findings justify it. It answers with a small JSON
object (``changes``, ``grounds``, ``rationale``) that this module wraps into a
schema-v2 recommendation.

Three layers keep a weak model from doing damage:

1. **Structured output.** The JSON schema sent to the server lists only
   the levers that are available this window, bounds each one to the range it
   can reach this window, and limits ``grounds`` to the ids of findings that
   justify that lever. A server that enforces the schema (Ollama ``format``,
   OpenAI-compatible ``response_format``) cannot return an illegal key.
2. **One retry with the validator's errors.** A reply that fails the schema
   or the stateful validator is sent back once with the error list.
3. **The supervisor's own guard and judge**, unchanged: parse, validate,
   checkpoint, judge next window, roll back.

Backends are plain HTTP (standard library only): Ollama's ``/api/chat`` and
any OpenAI-compatible ``/chat/completions`` (llama.cpp server, vLLM, ...).
Each backend takes a ``transport`` callable so tests run without a server.
"""

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ..schema import WHITELIST_V2, SchemaError, parse_recommendation
from .detectors import allowed_levers
from .validator import (KeyHistory, ValidatorConfig, _cooldown_error, inert_reason, max_step, step_unit,
                        validate_proposal)

SCHEMA = 2

#: What each lever does, for the prompt. Kept factual: effect of raising it, and known failure modes.
LEVER_DOCS: Dict[str, str] = {
    "progress_coef": "weight of the reward for progress along the racing line; raising it pays more for covering ground",
    "waypoint_bonus": "bonus per waypoint cleared",
    "steering_penalty_w": "penalty on steering magnitude; raising it discourages large steering angles, which also "
                          "limits how hard the car can turn in corners and can cost corner speed",
    "jitter_coef": "penalty on the change in steering between steps; raising it makes steering corrections smaller "
                   "(it shrinks their size more than their frequency)",
    "throttle_change_coef": "penalty on the change in speed command between steps; raising it smooths the throttle. "
                            "Too high (about 2) made the policy pin its speed at maximum and crash",
    "overspeed_coef": "penalty on speed above grip_margin x the corner's grip limit; raising it teaches braking for "
                      "corners, and slows the car in every corner: in training runs it cost about 4 s per lap",
    "grip_margin": "multiplier on the grip limit used by the overspeed penalty; lower is more conservative",
    "time_cost": "cost per step; raising it pushes for shorter laps",
    "lap_bonus": "bonus for completing a lap",
    "ent_coef": "PPO entropy bonus; raising it increases exploration",
    "spawn_random": "share of training episodes that start at a random point on the track",
    "spawn_corner_mix": "share of training episodes that start just before a corner (corner practice)",
}

#: Prompt v2 adds facts about how the levers act, because v1 models (4B to 30B) moved jitter_coef by 1-14% of its
#: per-window cap, far below where it acts, and changed grip_margin while it had no effect. Only the controlled lever
#: checks (notes sections 11-12) and the reward's own definition are used. Effective values only: a first draft that
#: also quoted the values with no effect (0.15, 1.3) made small models copy exactly those numbers.
LEVER_MAGNITUDES: Dict[str, str] = {
    "jitter_coef": "effective from about 10: at 15 (roughly 10% of the reward) steering corrections shrank by 26%",
    "throttle_change_coef": "effective around 0.1-0.5 (0.1 cut throttle changes by 17%); 2.0 caused reward hacking "
                            "(speed pinned at maximum, no laps completed)",
    "grip_margin": "acts only through the overspeed penalty: no effect at all while overspeed_coef is 0. Lower makes the "
                   "penalty start at a lower speed (more braking); higher allows faster corners",
    "time_cost": "lowering it weakens the push for shorter laps",
}

PROMPT_VERSIONS = ("v1", "v2", "v3")

#: v3 replies with a direction and a step per lever instead of a value; code turns it into the number.
STEP_SIZES: Dict[str, float] = {"quarter": 0.25, "half": 0.5, "full": 1.0}
DIRECTIONS: Dict[str, int] = {"raise": +1, "lower": -1}

SYSTEM_PROMPT = """You supervise the training of a reinforcement-learning racing driver (PPO, F1TENTH simulator).
The objective is the fastest completed lap; finishing laps reliably is a constraint, not the goal.
Every training window you read a telemetry summary and may change at most two training levers
(reward weights and curriculum settings). Your change is judged after the next window: if lap
completion or lap time gets clearly worse, it is rolled back and those levers are locked for a while.

Rules:
- Only change levers whose status is "available".
- Give the new absolute value; stay inside the lever's reachable range for this window.
- For every changed lever, cite in "grounds" one or more finding ids from the findings list whose
  "justified by" entry for that lever includes them. Never invent findings.
- Prefer one well-justified change over two. If nothing is worth changing, or every relevant lever
  is locked, return empty "changes" and "grounds".
- Reply with one JSON object only: {"changes": {...}, "grounds": {...}, "rationale": "..."}.
  The rationale is at most three sentences."""

#: v2: the same rules plus guidance on magnitude (ranges span orders of magnitude; tiny moves are judged "no change").
SYSTEM_PROMPT_V2 = SYSTEM_PROMPT.replace(
    "- Prefer one well-justified change over two.",
    "- Size matters: lever ranges span orders of magnitude, and a change far below a lever's effective size has no\n"
    "  measurable effect, so the next window is wasted (judged \"no clear change\"). Use the measured magnitudes in\n"
    "  the lever table where given; otherwise choose a step large enough to show an effect within one window.\n"
    "- Prefer one well-justified change over two.")

#: v3: v2's facts, but the model picks lever, direction and step size, and code computes the value. In the v1/v2
#: benchmarks models of every size (4B-30B) moved jitter_coef by 1-14% of its cap whatever the prompt said: choosing a
#: number on a range that spans 0.01-50 is the weak spot, not choosing what to change or which way.
SYSTEM_PROMPT_V3 = (SYSTEM_PROMPT
    .replace("- Give the new absolute value; stay inside the lever's reachable range for this window.",
             "- For each change give a direction (\"raise\" or \"lower\") and a step: \"full\" (the lever's full step,\n"
             "  the largest move allowed this window), \"half\" or \"quarter\". The new value is computed for you and\n"
             "  kept inside the lever's range.")
    .replace("- Prefer one well-justified change over two.",
             "- Size matters: lever ranges span orders of magnitude, and a small step often has no measurable effect\n"
             "  within one window (judged \"no clear change\", so the window is wasted). Use \"full\" unless a smaller\n"
             "  step is clearly safer, for example near a value with a known failure mode.\n"
             "- Prefer one well-justified change over two.")
    .replace('{"changes": {...}, "grounds": {...}, "rationale": "..."}',
             '{"changes": {"<lever>": {"direction": "raise", "step": "full"}},\n'
             '  "grounds": {"<lever>": ["<finding id>"]}, "rationale": "..."}'))

SYSTEM_PROMPTS = {"v1": SYSTEM_PROMPT, "v2": SYSTEM_PROMPT_V2, "v3": SYSTEM_PROMPT_V3}


# --------------------------------------------------------------------------- #
# Lever state and prompt
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LeverState:
    key: str
    current: float
    low: float
    high: float
    reach_low: float           # lowest value allowed this window (range and step cap)
    reach_high: float
    available: bool
    lock_reason: Optional[str]
    justified_by: Tuple[str, ...]
    unit: float = 0.0          # one full step (v3 replies)


def lever_states(current: Mapping[str, float], findings: List[Dict[str, Any]],
                 history: Mapping[str, KeyHistory], window: int,
                 validator: ValidatorConfig = ValidatorConfig()) -> List[LeverState]:
    """Per-lever facts for the prompt and the response schema, in whitelist order."""
    justify = allowed_levers(findings)
    out = []
    for key, spec in WHITELIST_V2.items():
        if key not in current:
            continue
        cur = float(current[key])
        step = max_step(key, validator, SCHEMA)
        lo = spec.low if step is None else max(spec.low, cur - step)
        hi = spec.high if step is None else min(spec.high, cur + step)
        hist = history.get(key)
        lock = _cooldown_error(key, hist, window, validator) if hist is not None else None
        just = tuple(justify.get(key, ()))
        if lock is None and not spec.low - 1e-9 <= cur <= spec.high + 1e-9:
            lock = f"{key}: current value {cur:g} is outside the allowed range"
            lo = hi = cur
        if lock is None and validator.require_effect:
            reason = inert_reason(key, current)
            lock = f"{key}: {reason}" if reason else None
        available = lock is None and bool(just)
        out.append(LeverState(key, cur, spec.low, spec.high, round(lo, 6), round(hi, 6), available, lock, just,
                              round(step_unit(key, validator, SCHEMA), 6)))
    return out


def _fmt(v: Any, nd: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}g}" if abs(v) >= 1e-3 or v == 0 else f"{v:.2e}"
    return str(v)


def build_user_prompt(report: Dict[str, Any], findings: List[Dict[str, Any]], levers: List[LeverState],
                      window: int, recent: Optional[List[Dict[str, Any]]] = None, version: str = "v1") -> str:
    """The per-window message: probe, training summary, recent decisions, findings, levers, task."""
    lines: List[str] = [f"Window {window} (training step {report.get('timesteps', '?')}), track {report.get('track', '?')}."]
    probe = report.get("probe") or {}
    if probe:
        n = probe.get("episodes") or 0
        done = round((probe.get("completion_rate") or 0.0) * n)
        sm = probe.get("smoothness") or {}
        lines += ["", f"Probe ({n} deterministic laps from the out-lap start, current policy):",
                  f"- completed {done}/{n}; lap time mean {_fmt(probe.get('lap_time_mean'))} s, "
                  f"std {_fmt(probe.get('lap_time_std'))}, best {_fmt(probe.get('lap_time_best'))}",
                  f"- straights: steering reversals {_fmt(sm.get('steer_reversals_per_s'))}/s, steering change "
                  f"{_fmt(sm.get('steer_change_per_step'))} rad/step, drift {_fmt(sm.get('lateral_std_m'))} m (std), "
                  f"throttle change {_fmt(sm.get('throttle_change_per_step'))} m/s per step"]
        fails = probe.get("failures_at") or []
        if fails:
            lines.append("- failed laps ended at waypoints: " + ", ".join(f"{f['waypoint']} (x{f['count']})" for f in fails))
    terms = report.get("terminations") or {}
    if terms:
        spd = report.get("speed") or {}
        lines += ["", "Training rollouts this window (stochastic, include exploration):",
                  "- episode endings: " + ", ".join(f"{k} {_fmt(v.get('share'), 2)}" for k, v in terms.items()),
                  f"- speed mean {_fmt(spd.get('mean'))} m/s; command at floor {_fmt(spd.get('pinned_at_floor_share'), 2)} "
                  f"and at maximum {_fmt(spd.get('pinned_at_max_share'), 2)} of the time"]
    if recent:
        lines += ["", "Recent decisions (oldest first):"]
        for d in recent:
            ch = ", ".join(f"{k} {_fmt(d['before'].get(k))} -> {_fmt(v)}" for k, v in d["applied"].items())
            res = d.get("outcome") or "awaiting judgement"
            reason = f" ({d['reason']})" if d.get("reason") else ""
            lines.append(f"- window {d['window']}: {ch}; grounds {d.get('grounds')}; result: {res}{reason}")
    lines += ["", "Findings (from deterministic detectors):"]
    if findings:
        for f in findings:
            lines.append(f"- {f['id']} [{f['severity']}]: {f['message']}")
    else:
        lines.append("- none")
    steps = version == "v3"
    reach = "full step" if steps else "reachable this window"
    lines += ["", f"Levers (current | full range | {reach} | status | justified by | effect):"]
    for lv in levers:
        lock = lv.lock_reason.split(": ", 1)[-1] if lv.lock_reason else None
        status = "available" if lv.available else ("locked (" + lock + ")" if lock else "no current finding")
        span = _fmt(lv.unit, 4) if steps else f"{_fmt(lv.reach_low, 4)}..{_fmt(lv.reach_high, 4)}"
        lines.append(f"- {lv.key}: {_fmt(lv.current, 4)} | {_fmt(lv.low)}..{_fmt(lv.high)} | {span} | {status} | "
                     f"{', '.join(lv.justified_by) or '-'} | {_lever_doc(lv.key, version)}")
    lines += ["", "Propose at most two changes as JSON, or empty changes if none is warranted."]
    return "\n".join(lines)


def _lever_doc(key: str, version: str) -> str:
    doc = LEVER_DOCS.get(key, "")
    if version in ("v2", "v3") and key in LEVER_MAGNITUDES:
        doc += f" ({LEVER_MAGNITUDES[key]})"
    return doc


def _step_schema(directions: List[str]) -> Dict[str, Any]:
    return {"type": "object", "required": ["direction", "step"], "additionalProperties": False,
            "properties": {"direction": {"type": "string", "enum": directions},
                           "step": {"type": "string", "enum": list(STEP_SIZES)}}}


def _directions(lv: LeverState) -> List[str]:
    """Directions that move the lever (a lever at its limit can only go back)."""
    out = []
    if lv.current < lv.high - 1e-9:
        out.append("raise")
    if lv.current > lv.low + 1e-9:
        out.append("lower")
    return out


def response_schema(levers: List[LeverState], constrain: bool = True, steps: bool = False) -> Dict[str, Any]:
    """JSON schema for the reply. ``constrain`` limits keys, values and grounds to what is legal this window;
    ``steps`` asks for a direction and a step per lever instead of a value (prompt v3)."""
    if not constrain:
        value = _step_schema(list(DIRECTIONS)) if steps else {"type": "number"}
        return {"type": "object", "required": ["changes", "grounds", "rationale"],
                "properties": {"changes": {"type": "object", "additionalProperties": value},
                               "grounds": {"type": "object",
                                           "additionalProperties": {"type": "array", "items": {"type": "string"}}},
                               "rationale": {"type": "string"}}}
    avail = [lv for lv in levers if lv.available and (_directions(lv) or not steps)]

    def value(lv: LeverState) -> Dict[str, Any]:
        if steps:
            return _step_schema(_directions(lv))
        return {"type": "number", "minimum": lv.reach_low, "maximum": lv.reach_high}
    return {
        "type": "object", "required": ["changes", "grounds", "rationale"], "additionalProperties": False,
        "properties": {
            "changes": {"type": "object", "additionalProperties": False,
                        "properties": {lv.key: value(lv) for lv in avail}},
            "grounds": {"type": "object", "additionalProperties": False,
                        "properties": {lv.key: {"type": "array", "minItems": 1,
                                                "items": {"type": "string", "enum": list(lv.justified_by)}}
                                       for lv in avail}},
            "rationale": {"type": "string"},
        },
    }


def steps_to_values(changes: Mapping[str, Any], levers: List[LeverState]) -> Tuple[Dict[str, float], List[str]]:
    """Turn v3 ``{lever: {"direction", "step"}}`` into absolute values, clipped to each lever's range."""
    by_key = {lv.key: lv for lv in levers}
    values: Dict[str, float] = {}
    errors: List[str] = []
    for key, ask in changes.items():
        lv = by_key.get(key)
        if lv is None:
            errors.append(f"{key}: not a lever")
            continue
        if not isinstance(ask, Mapping) or ask.get("direction") not in DIRECTIONS or ask.get("step") not in STEP_SIZES:
            errors.append(f"{key}: give {{\"direction\": \"raise\"|\"lower\", \"step\": \"quarter\"|\"half\"|\"full\"}}, "
                          f"got {ask!r}")
            continue
        new = lv.current + DIRECTIONS[ask["direction"]] * STEP_SIZES[ask["step"]] * lv.unit
        new = round(min(max(new, lv.low), lv.high), 6)
        if abs(new - lv.current) < 1e-9:
            errors.append(f"{key}: already at its {'upper' if ask['direction'] == 'raise' else 'lower'} limit")
            continue
        values[key] = new
    return values, errors


# --------------------------------------------------------------------------- #
# Reply parsing
# --------------------------------------------------------------------------- #

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def extract_json(text: str) -> Dict[str, Any]:
    """The first JSON object in a reply (tolerates think tags, code fences and surrounding prose)."""
    text = _THINK_RE.sub("", text or "")
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = dec.raw_decode(text, i)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise ValueError("no JSON object in the reply")


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #

Transport = Callable[[str, Dict[str, Any], float, Dict[str, str]], Dict[str, Any]]


class ServerError(RuntimeError):
    """The LLM server answered with an HTTP error (model failed to load, out of memory, unknown model, ...)."""


def http_post_json(url: str, payload: Dict[str, Any], timeout: float, headers: Dict[str, str]) -> Dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:500]
        raise ServerError(f"HTTP {exc.code} from {url}: {body}") from exc


@dataclass
class LLMReply:
    text: str
    thinking: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    latency_s: float = 0.0


@dataclass
class OllamaBackend:
    """Ollama ``/api/chat`` with ``format`` set to the JSON schema. ``think`` None leaves the model default."""

    model: str
    url: str = "http://localhost:11434"
    think: Optional[Any] = None    # True/False, or "low"/"medium"/"high" for models with reasoning effort (gpt-oss)
    temperature: float = 0.2
    num_ctx: int = 8192
    seed: Optional[int] = None
    timeout: float = 600.0
    keep_alive: str = "2h"     # windows are minutes apart; the default 5 min would reload the model every window
    transport: Transport = http_post_json
    name: str = "ollama"

    def chat(self, messages: List[Dict[str, str]], schema: Optional[Dict[str, Any]]) -> LLMReply:
        options: Dict[str, Any] = {"temperature": self.temperature, "num_ctx": self.num_ctx}
        if self.seed is not None:
            options["seed"] = self.seed
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "stream": False, "options": options,
                                   "keep_alive": self.keep_alive}
        if schema is not None:
            payload["format"] = schema
        if self.think is not None:
            payload["think"] = self.think
        t0 = time.time()
        resp = self.transport(self.url.rstrip("/") + "/api/chat", payload, self.timeout, {})
        msg = resp.get("message") or {}
        return LLMReply(text=msg.get("content", ""), thinking=msg.get("thinking"),
                        prompt_tokens=resp.get("prompt_eval_count"), completion_tokens=resp.get("eval_count"),
                        latency_s=time.time() - t0)


@dataclass
class OpenAIBackend:
    """Any OpenAI-compatible ``/chat/completions`` server (llama.cpp, vLLM) with a ``json_schema`` response format."""

    model: str
    url: str = "http://localhost:8080/v1"
    api_key: Optional[str] = None
    temperature: float = 0.2
    seed: Optional[int] = None
    timeout: float = 600.0
    transport: Transport = http_post_json
    name: str = "openai"

    def chat(self, messages: List[Dict[str, str]], schema: Optional[Dict[str, Any]]) -> LLMReply:
        payload: Dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.temperature}
        if self.seed is not None:
            payload["seed"] = self.seed
        if schema is not None:
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": "proposal", "schema": schema, "strict": True}}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        t0 = time.time()
        resp = self.transport(self.url.rstrip("/") + "/chat/completions", payload, self.timeout, headers)
        choice = (resp.get("choices") or [{}])[0].get("message") or {}
        usage = resp.get("usage") or {}
        return LLMReply(text=choice.get("content") or "", thinking=choice.get("reasoning_content"),
                        prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                        latency_s=time.time() - t0)


def make_backend(kind: str, model: str, url: Optional[str] = None, **kwargs: Any):
    if kind == "ollama":
        return OllamaBackend(model=model, **({"url": url} if url else {}), **kwargs)
    if kind == "openai":
        kwargs.pop("think", None)
        kwargs.pop("num_ctx", None)
        kwargs.pop("keep_alive", None)
        return OpenAIBackend(model=model, **({"url": url} if url else {}), **kwargs)
    raise ValueError(f"unknown LLM backend {kind!r} (ollama, openai)")


# --------------------------------------------------------------------------- #
# Recent decisions (for the prompt), read from proposals.jsonl
# --------------------------------------------------------------------------- #

def recent_decisions(log_path: Path, before_window: Optional[int] = None, n: int = 4,
                     base: Optional[Mapping[str, float]] = None) -> List[Dict[str, Any]]:
    """The last ``n`` accepted changes with their judgement, from a supervisor log.

    Only entries a supervisor deciding window ``before_window`` would have seen are used: proposals from earlier
    windows and judgements up to and including that window. ``base`` (the lever values before any change) fills in
    the "before" value for logs written before proposals recorded ``values_before``.
    """
    path = Path(log_path)
    if not path.exists():
        return []
    decisions: Dict[int, Dict[str, Any]] = {}
    current: Dict[str, float] = dict(base or {})
    for line in path.read_text().splitlines():
        e = json.loads(line)
        w = int(e["window"])
        if e.get("type") == "proposal":
            if before_window is not None and w >= before_window:
                continue
            if e.get("status") == "accepted":
                applied = e.get("applied") or {}
                before = {k: e.get("values_before", {}).get(k, current.get(k)) for k in applied}
                decisions[w] = {"window": w, "applied": applied, "before": before, "grounds": e.get("grounds"),
                                "outcome": None, "reason": None}
                current.update(applied)
        elif e.get("type") == "judgement":
            if before_window is not None and w > before_window:
                continue
            d = decisions.get(int(e["change_window"]))
            if d is not None:
                d["outcome"], d["reason"] = e.get("outcome"), e.get("reason")
                if e.get("outcome") == "rolled_back":
                    current.update({k: v for k, v in d["before"].items() if v is not None})
    return [decisions[w] for w in sorted(decisions)][-n:]


# --------------------------------------------------------------------------- #
# The proposer
# --------------------------------------------------------------------------- #

class LLMError(RuntimeError):
    """The model produced no usable JSON after all attempts."""


@dataclass
class LLMProposer:
    """Asks an LLM backend for a schema-v2 recommendation; retries once with the validator's errors."""

    backend: Any
    validator: ValidatorConfig = field(default_factory=ValidatorConfig)
    max_retries: int = 1
    constrain: bool = True
    prompt: str = "v1"
    name: str = "llm"
    last_meta: Optional[Dict[str, Any]] = None

    def messages(self, report, findings, current, history, window, context) -> Tuple[List[Dict[str, str]], List[LeverState]]:
        levers = lever_states(current, findings, history, window, self.validator)
        if self.prompt not in SYSTEM_PROMPTS:
            raise ValueError(f"unknown prompt version {self.prompt!r} ({', '.join(PROMPT_VERSIONS)})")
        user = build_user_prompt(report, findings, levers, window, context.get("recent"), self.prompt)
        return [{"role": "system", "content": SYSTEM_PROMPTS[self.prompt]}, {"role": "user", "content": user}], levers

    def _check(self, obj: Dict[str, Any], findings, current, history, window, context,
               n_waypoints: int, levers: Optional[List[LeverState]] = None) -> Tuple[Dict[str, Any], List[str]]:
        changes = obj.get("changes") or {}
        step_errors: List[str] = []
        if self.prompt == "v3":
            if not isinstance(changes, Mapping):
                changes, step_errors = {}, ["changes must be an object"]
            else:
                changes, step_errors = steps_to_values(changes, levers or [])
        doc = {"schema_version": SCHEMA, "source_model": context.get("source_model") or "supervisor",
               "track": context.get("track", ""), "changes": changes,
               "grounds": {k: v for k, v in (obj.get("grounds") or {}).items()
                           if self.prompt != "v3" or k in changes}}
        if isinstance(obj.get("rationale"), str):
            doc["rationale"] = obj["rationale"][:1000]
        if step_errors:
            return doc, step_errors
        try:
            rec = parse_recommendation(doc, n_waypoints)
        except SchemaError as exc:
            return doc, list(exc.errors)
        result = validate_proposal(rec, current, window, history, self.validator, findings=findings)
        return doc, list(result.errors)

    def propose(self, report, findings, current, history, window, context):
        messages, levers = self.messages(report, findings, current, history, window, context)
        schema = response_schema(levers, self.constrain, steps=self.prompt == "v3")
        n_wp = int(report.get("n_waypoints") or context.get("n_waypoints") or 10_000)
        attempts: List[Dict[str, Any]] = []
        doc: Optional[Dict[str, Any]] = None
        self.last_meta = {"backend": getattr(self.backend, "name", "?"), "model": getattr(self.backend, "model", "?"),
                          "prompt": self.prompt, "attempts": attempts}
        for attempt in range(1 + self.max_retries):
            reply = self.backend.chat(messages, schema)
            rec: Dict[str, Any] = {"latency_s": round(reply.latency_s, 3), "prompt_tokens": reply.prompt_tokens,
                                   "completion_tokens": reply.completion_tokens, "text": reply.text[:4000],
                                   "thinking_chars": len(reply.thinking or "")}
            try:
                obj = extract_json(reply.text)
            except ValueError as exc:
                rec["errors"] = [str(exc)]
                attempts.append(rec)
                feedback = ["the reply did not contain a JSON object"]
            else:
                doc, errors = self._check(obj, findings, current, history, window, context, n_wp, levers)
                rec["errors"] = errors
                if self.prompt == "v3":
                    rec["steps"] = obj.get("changes")            # what the model asked for, before conversion
                attempts.append(rec)
                if not errors:
                    break
                feedback = errors
            if attempt < self.max_retries:
                messages = messages + [
                    {"role": "assistant", "content": reply.text},
                    {"role": "user", "content": "That proposal was rejected by the validator:\n- "
                     + "\n- ".join(feedback) + "\nReturn a corrected JSON object, or empty changes."}]
        self.last_meta["latency_s"] = round(sum(a["latency_s"] for a in attempts), 3)
        if doc is None:
            raise LLMError("; ".join(attempts[-1]["errors"]))
        return doc
