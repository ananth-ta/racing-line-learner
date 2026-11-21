"""Offline proposer benchmark: score LLMs (and the rule-based tuner) on saved telemetry windows.

A *case* is one supervisor decision point, rebuilt exactly: the window's
telemetry report and findings, the lever values in force, the per-key
history (cooldowns, rollbacks) and the recent decisions, replayed from the
run's ``proposals.jsonl``. Runs without a supervisor log (plain telemetry
reports) give cases with the report's own config and no history.

Each proposal is scored on:

* ``json_ok``: the reply held a JSON object;
* ``accepted``: it passed the schema and the stateful validator (first try,
  and after the one retry);
* ``abstained``: it proposed no change (scored against whether any lever was
  available at all);
* ``direction_ok``: every change moves its lever the way its cited finding
  calls for (:data:`EXPECTED_DIRECTION`; a lever/finding pair with no stated
  direction does not count against it);
* ``useful``: on cases where some lever was available, an accepted change with every scorable move in the
  right direction. This is the headline: ``accepted`` alone rewards abstaining, and the validator checks
  legality, not sense, so a backwards change passes it (only the judge would catch it, a window later);
* ``rules_overlap``: Jaccard overlap of changed keys with the rule-based tuner;
* ``step_fraction``: mean size of the accepted moves as a fraction of each lever's per-window cap (descriptive:
  levers only act in narrow bands, so a timid 1%-of-cap nudge is legal but has no measurable effect);
* latency and token counts.

Usage::

    python -m racing_line_learner.in_loop.bench build --runs runs/cota_rules_s0 --out bench/cases.json
    python -m racing_line_learner.in_loop.bench run --cases bench/cases.json --proposer rules --out bench/rules
    python -m racing_line_learner.in_loop.bench run --cases bench/cases.json --proposer llm \\
        --backend ollama --model qwen3:8b --think off --repeats 3 --out bench/qwen3_8b_nothink
"""

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ..schema import WHITELIST_V2
from .llm import PROMPT_VERSIONS, LLMProposer, lever_states, make_backend, recent_decisions
from .validator import (OUTCOME_PENDING, OUTCOME_ROLLED_BACK, KeyHistory, ValidatorConfig)

#: Direction a lever should move to address a finding (+1 raise, -1 lower). Pairs not listed are not scored.
EXPECTED_DIRECTION: Dict[str, Dict[str, int]] = {
    "weaving": {"jitter_coef": +1, "steering_penalty_w": +1},
    "throttle_jitter": {"throttle_change_coef": +1},
    "crash_hotspot": {"overspeed_coef": +1, "grip_margin": -1, "spawn_corner_mix": +1, "jitter_coef": +1,
                      "steering_penalty_w": +1},
    "speed_pinned_at_max": {"overspeed_coef": +1, "grip_margin": -1, "time_cost": -1},
    "speed_pinned_at_floor": {"time_cost": +1, "progress_coef": +1},
    "stagnation_dominant": {"progress_coef": +1, "ent_coef": +1},
    "corkscrew": {"steering_penalty_w": +1, "jitter_coef": +1, "progress_coef": +1},
    "steering_saturated": {"steering_penalty_w": +1, "jitter_coef": +1},
    "under_driven_segment": {"time_cost": +1, "progress_coef": +1, "lap_bonus": +1, "grip_margin": +1},
    "inconsistent_line": {"spawn_corner_mix": +1, "jitter_coef": +1},
    "probe_failures": {"spawn_corner_mix": +1, "overspeed_coef": +1, "grip_margin": -1},
}

DEFAULT_ENT_COEF = 0.01


# --------------------------------------------------------------------------- #
# Cases
# --------------------------------------------------------------------------- #

def _levers_from_config(config: Dict[str, Any]) -> Dict[str, float]:
    out = {k: float(config[k]) for k in WHITELIST_V2 if k in config}
    out.setdefault("ent_coef", float(config.get("ent_coef", DEFAULT_ENT_COEF)))
    return out


def _replay_state(log_path: Path, base: Dict[str, float], window: int):
    """Lever values and key history the supervisor held when proposing in ``window``."""
    current = dict(base)
    history: Dict[str, Dict[str, Any]] = {}
    before_of: Dict[int, Dict[str, float]] = {}
    keys_of: Dict[int, List[str]] = {}
    if not log_path.exists():
        return current, history
    for line in log_path.read_text().splitlines():
        e = json.loads(line)
        w = int(e["window"])
        if e.get("type") == "proposal":
            if w >= window:
                break
            if e.get("status") == "accepted":
                applied = e.get("applied") or {}
                before_of[w] = {k: current.get(k) for k in applied}
                keys_of[w] = sorted(applied)
                current.update(applied)
                for k in applied:
                    history[k] = {"window": w, "outcome": OUTCOME_PENDING, "judged_window": None}
        elif e.get("type") == "judgement":
            if w > window:
                break
            cw = int(e["change_window"])
            for k in keys_of.get(cw, e.get("keys", [])):
                history[k] = {"window": cw, "outcome": e["outcome"], "judged_window": w}
            if e["outcome"] == OUTCOME_ROLLED_BACK:
                current.update({k: v for k, v in before_of.get(cw, {}).items() if v is not None})
    return current, history


def build_cases(run_dirs: Sequence[str], redetect: bool = False) -> List[Dict[str, Any]]:
    """One case per saved telemetry window of each run (windows with a report).

    ``redetect`` reruns today's detectors on each saved report instead of using the findings saved with it, so
    cases pick up detectors added since the report was written.
    """
    if redetect:
        from .detectors import detect
    cases = []
    for run in run_dirs:
        run = Path(run)
        reports = sorted((run / "telemetry").glob("w*.json"))
        if not reports:
            continue
        first = json.loads(reports[0].read_text())
        base = _levers_from_config(first.get("config", {}))
        log = run / "proposals.jsonl"
        for path in reports:
            report = json.loads(path.read_text())
            if redetect:
                report["findings"] = detect(report)
            window = int(report.get("window", 0))
            if log.exists():
                current, history = _replay_state(log, base, window)
            else:
                current, history = _levers_from_config(report.get("config", {})), {}
            cases.append({"id": f"{run.name}/w{window:04d}", "run": run.name, "window": window,
                          "track": report.get("track", ""), "report": report,
                          "findings": report.get("findings", []), "current": current, "history": history,
                          "recent": recent_decisions(log, before_window=window, base=base) if log.exists() else []})
    return cases


def _history(case) -> Dict[str, KeyHistory]:
    return {k: KeyHistory(**v) for k, v in case["history"].items()}


def _context(case) -> Dict[str, Any]:
    return {"source_model": "bench", "track": case["track"], "recent": case.get("recent", []),
            "n_waypoints": case["report"].get("n_waypoints")}


# --------------------------------------------------------------------------- #
# Snapshots: telemetry reports from saved checkpoints (runs without a supervisor)
# --------------------------------------------------------------------------- #

_STEPS_RE = re.compile(r"_(\d+)_steps\.zip$")


def snapshot(checkpoint: str, out_root: str, rollout_steps: int = 50_000, probe_runs: int = 8,
             probe_max_sim_steps: int = 30_000, window_steps: int = 250_000, seed: int = 0) -> Path:
    """Write one telemetry report (``<out_root>/<run>_<steps>/telemetry/wNNNN.json``) for a checkpoint.

    The training window is stood in for by ``rollout_steps`` stochastic policy steps under the run's own training
    config (random starts, exploration noise); the probe is the supervisor's deterministic probe.
    """
    from stable_baselines3 import PPO

    from ..config import EnvConfig, save_json
    from ..env.tracks import load_track
    from ..env.wrapper import make_env
    from .detectors import detect
    from .supervisor import probe_policy
    from .telemetry import TelemetryCollector

    ckpt = Path(checkpoint)
    run_dir = ckpt.parent.parent
    spec = json.loads((run_dir / "run.json").read_text())
    m = _STEPS_RE.search(ckpt.name)
    steps = int(m.group(1)) if m else 0
    track, cfg = load_track(spec["track"]), EnvConfig.from_dict(spec["env_config"])
    model = PPO.load(str(ckpt), device="cpu")
    env = make_env(track, cfg, seed=seed)
    collector = TelemetryCollector(track, cfg)
    try:
        obs = env.reset()
        for _ in range(rollout_steps):
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done, info = env.step(action)
            collector.update(info, float(reward))
            if done:
                obs = env.reset()
    finally:
        env.close()
    probe = probe_policy(model, track, cfg, probe_runs, probe_max_sim_steps, seed)
    window = max(1, steps // window_steps)
    report = collector.report(window=window, timesteps=steps, probe=probe)
    report["config"]["ent_coef"] = float(model.ent_coef)
    report["findings"] = detect(report)
    report["source"] = {"checkpoint": str(ckpt), "offline_snapshot": True, "rollout_steps": rollout_steps,
                        "probe_runs": probe_runs, "seed": seed}
    out = Path(out_root) / f"{run_dir.name}_{steps}" / "telemetry" / f"w{window:04d}.json"
    save_json(report, out)
    return out


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def direction_ok(raw: Optional[Dict[str, Any]], current: Dict[str, float]) -> Optional[bool]:
    """Whether every change moves the way one of its cited findings calls for (None if nothing is scorable)."""
    if not raw or not raw.get("changes"):
        return None
    verdicts = []
    for key, new in raw["changes"].items():
        if not isinstance(new, (int, float)) or key not in current:
            verdicts.append(False)
            continue
        sign = (new > current[key]) - (new < current[key])
        wanted = [EXPECTED_DIRECTION.get(g, {}).get(key) for g in (raw.get("grounds") or {}).get(key, [])]
        wanted = [d for d in wanted if d is not None]
        if wanted:
            verdicts.append(sign in wanted)
    return all(verdicts) if verdicts else None


def step_fraction(raw: Optional[Dict[str, Any]], current: Dict[str, float],
                  validator: ValidatorConfig = ValidatorConfig()) -> Optional[float]:
    """Mean |move| / per-window cap over the capped levers a proposal changes (None if none are capped)."""
    from .validator import max_step
    fr = []
    for key, new in ((raw or {}).get("changes") or {}).items():
        cap = max_step(key, validator, 2)
        if cap and key in current and isinstance(new, (int, float)):
            fr.append(abs(new - current[key]) / cap)
    return round(sum(fr) / len(fr), 3) if fr else None


def _rules_reference(case, validator: ValidatorConfig) -> Optional[Dict[str, Any]]:
    from .supervisor import RuleBasedProposer
    return RuleBasedProposer(validator).propose(case["report"], case["findings"], dict(case["current"]),
                                                _history(case), case["window"], _context(case))


def score(case, raw, meta, error, validator: ValidatorConfig, rules_raw) -> Dict[str, Any]:
    from ..schema import SchemaError, parse_recommendation
    from .validator import validate_proposal
    attempts = (meta or {}).get("attempts") or []
    accepted, errors = False, []
    abstain_none = raw is None and not error           # a proposer returning None chose "no change"
    if abstain_none:
        accepted = True
    elif raw is not None:
        try:
            rec = parse_recommendation(raw, int(case["report"].get("n_waypoints") or 10_000))
            res = validate_proposal(rec, case["current"], case["window"], _history(case), validator,
                                    findings=case["findings"])
            accepted, errors = res.accepted, res.errors
        except SchemaError as exc:
            errors = exc.errors
    elif error:
        errors = [error]
    changed = set((raw or {}).get("changes") or {})
    ref = set((rules_raw or {}).get("changes") or {})
    available = any(lv.available for lv in lever_states(case["current"], case["findings"], _history(case),
                                                        case["window"], validator))
    return {
        "case": case["id"], "json_ok": raw is not None or abstain_none, "accepted": accepted,
        "accepted_first_try": accepted and (len(attempts) <= 1),
        "attempts": len(attempts) or (0 if error else 1),
        "abstained": (raw is not None or abstain_none) and not changed, "any_lever_available": available,
        "n_changes": len(changed), "changes": (raw or {}).get("changes"), "grounds": (raw or {}).get("grounds"),
        "direction_ok": direction_ok(raw, case["current"]) if accepted else None,
        "useful": (accepted and bool(changed) and direction_ok(raw, case["current"]) is not False) if available else None,
        "step_fraction": step_fraction(raw, case["current"], validator) if accepted else None,
        "rules_changes": sorted(ref),
        "rules_overlap": (len(changed & ref) / len(changed | ref)) if (changed | ref) else 1.0,
        "latency_s": (meta or {}).get("latency_s"),
        "completion_tokens": sum(a.get("completion_tokens") or 0 for a in attempts) or None,
        "errors": errors, "rationale": (raw or {}).get("rationale"),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def share(key, rows_=rows):
        vals = [r[key] for r in rows_ if r[key] is not None]
        return round(sum(bool(v) for v in vals) / len(vals), 3) if vals else None
    lat = sorted(r["latency_s"] for r in rows if r["latency_s"] is not None)
    toks = [r["completion_tokens"] for r in rows if r["completion_tokens"]]
    actionable = [r for r in rows if r["any_lever_available"]]
    steps = [r["step_fraction"] for r in rows if r.get("step_fraction") is not None]
    return {
        "n": len(rows), "actionable": len(actionable), "useful": share("useful"),
        "json_ok": share("json_ok"), "accepted": share("accepted"),
        "accepted_first_try": share("accepted_first_try"), "direction_ok": share("direction_ok"),
        "abstain_when_actionable": share("abstained", actionable),
        "rules_overlap": round(statistics.mean(r["rules_overlap"] for r in rows), 3) if rows else None,
        "step_fraction_mean": round(statistics.mean(steps), 3) if steps else None,
        "latency_p50_s": round(statistics.median(lat), 2) if lat else None,
        "latency_max_s": round(lat[-1], 2) if lat else None,
        "completion_tokens_mean": round(statistics.mean(toks)) if toks else None,
    }


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #

def run(cases: List[Dict[str, Any]], proposer, repeats: int = 1,
        validator: ValidatorConfig = ValidatorConfig(), log=print) -> List[Dict[str, Any]]:
    rows = []
    for rep in range(repeats):
        for case in cases:
            if hasattr(proposer, "last_meta"):
                proposer.last_meta = None
            t0 = time.time()
            raw, error = None, None
            try:
                raw = proposer.propose(case["report"], case["findings"], dict(case["current"]), _history(case),
                                       case["window"], _context(case))
            except Exception as exc:               # LLMError, HTTP errors, timeouts: score the failure, keep going
                error = f"{type(exc).__name__}: {exc}"
            meta = getattr(proposer, "last_meta", None) or {"latency_s": round(time.time() - t0, 3)}
            row = score(case, raw, meta, error, validator, _rules_reference(case, validator))
            row["repeat"] = rep
            rows.append(row)
            log(f"[{rep}] {case['id']}: accepted={row['accepted']} changes={row['changes']} "
                f"latency={row['latency_s']}s {'; '.join(row['errors'])[:200]}")
    return rows


def _proposer_from_args(args, validator):
    if args.proposer == "rules":
        from .supervisor import RuleBasedProposer
        return RuleBasedProposer(validator)
    from ..train.ppo import think_value
    think = think_value(args.think)
    kwargs: Dict[str, Any] = {"temperature": args.temperature, "timeout": args.timeout, "seed": args.seed}
    if args.backend == "ollama":
        kwargs.update(think=think, num_ctx=args.num_ctx)
    backend = make_backend(args.backend, args.model, args.url, **kwargs)
    return LLMProposer(backend, validator=validator, max_retries=args.retries, constrain=not args.unconstrained,
                       prompt=args.prompt)


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="python -m racing_line_learner.in_loop.bench", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="Build benchmark cases from run directories.")
    b.add_argument("--runs", nargs="+", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--redetect", action="store_true", help="Rerun the current detectors on each saved report.")
    sn = sub.add_parser("snapshot", help="Telemetry report(s) from saved checkpoints (needs the simulator and SB3).")
    sn.add_argument("--checkpoints", nargs="+", required=True)
    sn.add_argument("--out", required=True, help="Root folder; one run-like folder per checkpoint.")
    sn.add_argument("--rollout_steps", type=int, default=50_000)
    sn.add_argument("--probe_runs", type=int, default=8)
    sn.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run", help="Score a proposer on saved cases.")
    r.add_argument("--cases", required=True)
    r.add_argument("--out", required=True, help="Output directory (results.jsonl, summary.json).")
    r.add_argument("--proposer", default="llm", choices=("llm", "rules"))
    r.add_argument("--backend", default="ollama", choices=("ollama", "openai"))
    r.add_argument("--model", default=None)
    r.add_argument("--url", default=None)
    r.add_argument("--think", default="default", choices=("default", "on", "off", "low", "medium", "high"))
    r.add_argument("--temperature", type=float, default=0.2)
    r.add_argument("--timeout", type=float, default=600.0)
    r.add_argument("--num_ctx", type=int, default=8192)
    r.add_argument("--retries", type=int, default=1)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--repeats", type=int, default=1)
    r.add_argument("--prompt", default="v1", choices=PROMPT_VERSIONS,
                   help="Prompt version (v2: measured magnitudes; v3: v2 with direction-and-step replies).")
    r.add_argument("--unconstrained", action="store_true", help="Ablation: send a loose schema (any keys, values).")
    r.add_argument("--limit", type=int, default=None, help="Only the first N cases (smoke test).")
    args = p.parse_args(argv)

    if args.cmd == "build":
        cases = build_cases(args.runs, redetect=args.redetect)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(cases, indent=1))
        print(f"{len(cases)} cases -> {args.out}")
        return

    if args.cmd == "snapshot":
        for ck in args.checkpoints:
            t0 = time.time()
            out = snapshot(ck, args.out, args.rollout_steps, args.probe_runs, seed=args.seed)
            findings = [f["id"] for f in json.loads(out.read_text())["findings"]]
            print(f"{ck} -> {out} ({time.time() - t0:.0f}s) findings={findings}", flush=True)
        return

    if args.proposer == "llm" and not args.model:
        p.error("--proposer llm needs --model")
    validator = ValidatorConfig()
    cases = json.loads(Path(args.cases).read_text())[: args.limit]
    proposer = _proposer_from_args(args, validator)
    rows = run(cases, proposer, args.repeats, validator)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    summary = {"proposer": args.proposer, "backend": args.backend, "model": args.model, "think": args.think,
               "prompt": args.prompt,
               "constrained": not args.unconstrained, "repeats": args.repeats, "cases": len(cases),
               **summarize(rows)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
