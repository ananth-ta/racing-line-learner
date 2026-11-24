"""In-loop supervisor: every ``window_steps`` training steps, look, judge, propose, guard, apply.

At each window boundary (acted on at the end of a PPO rollout, so a rollout
never mixes two configs):

1. **Look.** Build the telemetry report for the window, run a small
   deterministic *probe* (``probe_runs`` jittered out-laps) for completion, lap
   time and smoothness, and run the detectors.
2. **Judge** the change made at the previous boundary, if any. With
   ``judge_mode="before_after"`` the probe before the change is compared with
   the probe after it. With ``judge_mode="control"`` a *control branch* (the
   same checkpoint trained for the same window without the change, in a
   separate process, see :mod:`racing_line_learner.in_loop.control`) is probed
   and compared instead, so ordinary learning progress is not credited to the
   change. A regression beyond the probe's own noise margin is **rolled back**:
   to the control branch when there is one, otherwise to the checkpoint taken
   just before the change (rolled-back steps still count toward the budget).
3. **Propose.** Ask the proposer (rules, LLM, replay, stub) for a schema-v2
   recommendation.
4. **Guard.** Parse (schema v2) and validate (step cap, cooldowns, grounding).
   Rejected proposals are logged and nothing changes.
5. **Apply.** Checkpoint, then push the new config to every env
   (``set_config``) and ``ent_coef`` to the model.

Everything is appended to ``proposals.jsonl`` and the state (current lever
values, per-key history, pending judgement) is written atomically to
``supervisor_state.json``, so a restarted run continues where it stopped and
never re-applies a decision.
"""

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..config import EnvConfig, save_json
from ..env.tracks import Track
from ..schema import WHITELIST_V2, SchemaError, parse_recommendation
from .detectors import LEVERS, detect
from .telemetry import TelemetryCollector, summarize_probe
from .validator import (OUTCOME_IMPROVED, OUTCOME_NEUTRAL, OUTCOME_PENDING, OUTCOME_ROLLED_BACK, KeyHistory,
                        ValidatorConfig, step_unit, validate_proposal)

SCHEMA = 2
ALGO_KEYS = ("ent_coef",)


# --------------------------------------------------------------------------- #
# Judging a change
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class JudgeConfig:
    """When a change counts as a regression or an improvement (probe-based, noise-aware)."""

    # Sized from the lever check: between 100k checkpoints, even an unchanged run swings 39.7-43.6 s and 0-8/8 laps.
    completion_drop: float = 0.26      # completion may fall by at most this much (two probe runs in eight)
    lap_time_margin_s: float = 1.5     # minimum lap-time margin; widened to 2x the probes' own spread
    target_gain: float = 0.10          # a targeted smoothness metric improving by this fraction counts as improved
    # Pace first: once the policy completes at least this share of probe laps, a clear lap-time loss is rolled back
    # even if completion rose. Below it the policy is not yet reliable, and reliability may cost some pace.
    reliable_completion: float = 0.5


#: How a change is judged: against the probe before it, or against a control branch trained without it.
JUDGE_MODES = ("before_after", "control")

#: Which probe smoothness metric a finding's fix should move (lower is better).
TARGET_METRIC = {
    # The jitter penalty shrinks the size of steering corrections, not their count (lever check 2026-11-15).
    "weaving": "steer_change_per_step",
    "throttle_jitter": "throttle_change_per_step",
}


def judge(before: Dict[str, Any], after: Dict[str, Any], targets: Sequence[str] = (),
          config: JudgeConfig = JudgeConfig()) -> Dict[str, Any]:
    """Compare two probe summaries. Returns ``{"outcome", "reason"}``; outcome is improved / neutral / rolled_back."""
    cb, ca = before.get("completion_rate") or 0.0, after.get("completion_rate") or 0.0
    if ca < cb - config.completion_drop:
        return {"outcome": OUTCOME_ROLLED_BACK, "reason": f"completion fell {cb:.2f} -> {ca:.2f}"}
    tb, ta = before.get("lap_time_mean"), after.get("lap_time_mean")
    spread = max(before.get("lap_time_std") or 0.0, after.get("lap_time_std") or 0.0)
    margin = max(config.lap_time_margin_s, 2 * spread)
    if tb is not None and ta is not None and ta > tb + margin and (ca <= cb or cb >= config.reliable_completion):
        also = f", although completion rose {cb:.2f} -> {ca:.2f}" if ca > cb else ""
        return {"outcome": OUTCOME_ROLLED_BACK,
                "reason": f"lap time worse {tb:.2f} -> {ta:.2f} s (margin {margin:.2f}){also}"}
    if tb is not None and ta is not None and ta < tb - margin and ca >= cb - 1e-9:
        return {"outcome": OUTCOME_IMPROVED, "reason": f"lap time better {tb:.2f} -> {ta:.2f} s"}
    if ca > cb + config.completion_drop:
        return {"outcome": OUTCOME_IMPROVED, "reason": f"completion rose {cb:.2f} -> {ca:.2f}"}
    smb, sma = before.get("smoothness") or {}, after.get("smoothness") or {}
    for fid in targets:
        metric = TARGET_METRIC.get(fid)
        vb, va = smb.get(metric), sma.get(metric)
        if metric and vb and va is not None and va <= vb * (1 - config.target_gain):
            return {"outcome": OUTCOME_IMPROVED, "reason": f"{metric} {vb} -> {va} with lap time and completion held"}
    return {"outcome": OUTCOME_NEUTRAL, "reason": "no clear change"}


# --------------------------------------------------------------------------- #
# Proposers
# --------------------------------------------------------------------------- #

class Proposer:
    """Turns a window's telemetry into a raw schema-v2 recommendation dict (or ``None`` for no change)."""

    name = "base"

    def propose(self, report: Dict[str, Any], findings: List[Dict[str, Any]], current: Dict[str, float],
                history: Dict[str, KeyHistory], window: int, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class StubProposer(Proposer):
    """Never changes anything: the supervisor runs (telemetry, probe, logging) as a no-intervention baseline."""

    name = "stub"

    def propose(self, *args, **kwargs):
        return None


#: Rule-based tuner: for each finding, the lever to move and in which direction (+1 raise, -1 lower).
#: Ordered by priority. Step size is the validator's per-window cap, so a rule never proposes an illegal move.
RULES = (
    ("probe_failures", "spawn_corner_mix", +1),     # reliability first: corner practice changes no reward weight
    ("weaving", "jitter_coef", +1),
    ("throttle_jitter", "throttle_change_coef", +1),
    ("crash_hotspot", "overspeed_coef", +1),
    ("speed_pinned_at_max", "overspeed_coef", +1),
    ("inconsistent_line", "spawn_corner_mix", +1),
    ("stagnation_dominant", "progress_coef", +1),
    ("under_driven_segment", "time_cost", +1),
)


class RuleBasedProposer(Proposer):
    """The baseline the LLM must beat: fixed finding -> lever rules, one cap-sized step, at most two changes."""

    name = "rules"

    def __init__(self, validator: ValidatorConfig = ValidatorConfig(), max_changes: int = 2):
        self.validator = validator
        self.max_changes = max_changes

    def propose(self, report, findings, current, history, window, context):
        present = {f["id"] for f in findings}
        changes: Dict[str, float] = {}
        grounds: Dict[str, List[str]] = {}
        for fid, key, direction in RULES:
            if len(changes) >= self.max_changes:
                break
            if fid not in present or key in changes or key not in current:
                continue
            h = history.get(key)
            if h is not None and _locked(h, window, self.validator):
                continue
            spec = WHITELIST_V2[key]
            step = step_unit(key, self.validator, SCHEMA)
            new = min(max(current[key] + direction * step, spec.low), spec.high)
            if abs(new - current[key]) < 1e-9:
                continue                                    # already at its limit
            changes[key] = round(new, 6)
            grounds[key] = [fid]
        if not changes:
            return None
        return {"schema_version": SCHEMA, "source_model": context.get("source_model", "supervisor"),
                "track": context.get("track", ""), "changes": changes, "grounds": grounds,
                "rationale": "rule-based: " + ", ".join(f"{g[0]} -> {k}" for k, g in grounds.items())}


class ReplayProposer(Proposer):
    """Re-applies the proposals logged by an earlier run, window by window (no LLM needed to reproduce a run)."""

    name = "replay"

    def __init__(self, log_path: str):
        self.by_window: Dict[int, Dict[str, Any]] = {}
        for line in Path(log_path).read_text().splitlines():
            entry = json.loads(line)
            if entry.get("type") == "proposal" and entry.get("raw") is not None:
                self.by_window[int(entry["window"])] = entry["raw"]

    def propose(self, report, findings, current, history, window, context):
        return self.by_window.get(window)


def _locked(h: KeyHistory, window: int, cfg: ValidatorConfig) -> bool:
    if h.outcome == OUTCOME_ROLLED_BACK:
        return window < (h.judged_window or h.window) + cfg.rollback_cooldown_windows + 1
    if h.outcome == OUTCOME_IMPROVED:
        return False
    return h.outcome == OUTCOME_PENDING or window < h.window + cfg.cooldown_windows + 1


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #

@dataclass
class SupervisorState:
    window: int = -1                                   # last window decided
    current: Dict[str, float] = field(default_factory=dict)
    history: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    pending: Optional[Dict[str, Any]] = None           # change awaiting judgement
    counts: Dict[str, int] = field(default_factory=lambda: {"proposals": 0, "accepted": 0, "rejected": 0,
                                                             "rolled_back": 0, "improved": 0, "neutral": 0})

    def key_history(self) -> Dict[str, KeyHistory]:
        return {k: KeyHistory(**v) for k, v in self.history.items()}

    def save(self, path: Path) -> None:
        save_json(asdict(self), path)

    @classmethod
    def load(cls, path: Path) -> "SupervisorState":
        return cls(**json.loads(Path(path).read_text()))


def lever_values(config: EnvConfig, ent_coef: float) -> Dict[str, float]:
    """Current values of every schema-v2 lever."""
    out = {k: float(getattr(config, k)) for k in WHITELIST_V2 if hasattr(config, k)}
    out["ent_coef"] = float(ent_coef)
    return out


# --------------------------------------------------------------------------- #
# The callback
# --------------------------------------------------------------------------- #

def _make_callback_base():
    from stable_baselines3.common.callbacks import BaseCallback
    return BaseCallback


class SupervisorCallback(_make_callback_base()):
    """SB3 callback running the look / judge / propose / guard / apply loop. See the module docstring."""

    def __init__(self, track: Track, base_config: EnvConfig, proposer: Proposer, out_dir: str,
                 window_steps: int = 250_000, probe_runs: int = 8, probe_max_sim_steps: int = 30000,
                 validator: ValidatorConfig = ValidatorConfig(), judge_config: JudgeConfig = JudgeConfig(),
                 source_model: str = "", seed: int = 0, verbose: int = 0, judge_mode: str = "before_after",
                 control_timeout_s: float = 4 * 3600):
        super().__init__(verbose)
        if base_config.reward_mode != "laptime":
            raise ValueError("the supervisor uses schema v2 levers: train with a laptime-mode preset such as v2")
        self.track, self.base_config, self.proposer = track, base_config, proposer
        self.out_dir = Path(out_dir)
        self.window_steps, self.probe_runs, self.probe_max_sim_steps = window_steps, probe_runs, probe_max_sim_steps
        self.validator, self.judge_config = validator, judge_config
        self.source_model, self.seed = source_model, seed
        self.state_path = self.out_dir / "supervisor_state.json"
        self.log_path = self.out_dir / "proposals.jsonl"
        self.ckpt_dir = self.out_dir / "supervisor_checkpoints"
        self.collector = TelemetryCollector(track, base_config)
        self.state = SupervisorState.load(self.state_path) if self.state_path.exists() else SupervisorState()
        self._boundary_due = False
        if judge_mode not in JUDGE_MODES:
            raise ValueError(f"judge_mode must be one of {JUDGE_MODES}, got {judge_mode!r}")
        self.judge_mode, self.control_timeout_s = judge_mode, control_timeout_s
        self._control_proc: Optional[subprocess.Popen] = None

    # ---------------- helpers ----------------
    def _config(self) -> EnvConfig:
        return self._config_from(self.state.current)

    def _config_from(self, values: Dict[str, float]) -> EnvConfig:
        return replace(self.base_config, **{k: v for k, v in values.items() if k not in ALGO_KEYS})

    def _apply_current(self) -> None:
        cfg = self._config()
        self.training_env.env_method("set_config", cfg)
        if "ent_coef" in self.state.current:
            self.model.ent_coef = float(self.state.current["ent_coef"])
        self.collector.env_config = cfg

    def _log(self, entry: Dict[str, Any]) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=float) + "\n")

    def _probe(self) -> Dict[str, Any]:
        return probe_policy(self.model, self.track, self._config(), self.probe_runs, self.probe_max_sim_steps, self.seed)

    # ---------------- SB3 hooks ----------------
    def _on_training_start(self) -> None:
        if not self.state.current:
            self.state.current = lever_values(self.base_config, float(self.model.ent_coef))
        self._apply_current()      # also restores the supervised config after a restart

    def _on_training_end(self) -> None:
        if self._control_proc is not None and self._control_proc.poll() is None:
            self._control_proc.kill()                  # nothing left to judge it against
        self._control_proc = None

    def _on_step(self) -> bool:
        for i, (info, r) in enumerate(zip(self.locals.get("infos", []), self.locals.get("rewards", []))):
            self.collector.update(info, float(r), env_index=i)
        if self.num_timesteps // self.window_steps > max(self.state.window, -1) and self.num_timesteps >= self.window_steps:
            self._boundary_due = True
        return True

    def _on_rollout_end(self) -> None:
        if not self._boundary_due:
            return
        self._boundary_due = False
        window = self.num_timesteps // self.window_steps
        if window <= self.state.window:
            return                                     # already decided (e.g. resumed after this boundary)
        t0 = time.time()
        probe = self._probe()
        report = self.collector.report(window=window, timesteps=self.num_timesteps, probe=probe)
        findings = detect(report)
        report["findings"] = findings
        self.out_dir.mkdir(parents=True, exist_ok=True)
        save_json(report, self.out_dir / "telemetry" / f"w{window:04d}.json")
        rolled_back_to = self._judge_pending(window, probe)
        # After a rollback the policy is the restored checkpoint, so the next change must be judged against that
        # checkpoint's own probe (taken when it was saved), not against the probe of the rolled-back policy.
        baseline = rolled_back_to if rolled_back_to is not None else probe
        total = int(getattr(self.model, "_total_timesteps", 0) or 0)
        if total and (window + 1) * self.window_steps > total:
            # A change made now could neither be trained nor judged before the run ends.
            self._log({"type": "proposal", "window": window, "timesteps": self.num_timesteps,
                       "proposer": self.proposer.name, "status": "skipped", "errors": [],
                       "reason": "final window: a change could not be trained or judged before the run ends"})
        else:
            self._propose_and_apply(window, report, findings, baseline)
        self.state.window = window
        self.state.save(self.state_path)
        self.collector.reset()
        if self.verbose:
            print(f"[supervisor] window {window} done in {time.time() - t0:.1f}s; current {self.state.current}")

    # ---------------- judge / propose ----------------
    def _judge_pending(self, window: int, probe: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Judge the pending change. Returns the restored checkpoint's probe if it was rolled back, else ``None``."""
        p = self.state.pending
        if not p:
            return None
        control_path, control_error = self._wait_control(p) if p.get("control") else (None, None)
        if control_path is not None:
            from stable_baselines3 import PPO
            control = PPO.load(control_path, device=self.model.device)
            reference = probe_policy(control, self.track, self._config_from(p["values_before"]), self.probe_runs,
                                     self.probe_max_sim_steps, self.seed)
            mode = "control"
        else:
            reference = p["probe_before"]
            mode = "before_after" + (f" (control branch failed: {control_error})" if control_error else "")
        verdict = judge(reference, probe, p.get("targets", ()), self.judge_config)
        outcome = verdict["outcome"]
        if outcome == OUTCOME_ROLLED_BACK:
            # The control branch is the policy trained this window without the change: switching to it keeps the
            # window's training. Without one, fall back to the checkpoint taken before the change.
            restore = control_path if control_path is not None else p["checkpoint"]
            self.model.set_parameters(restore, exact_match=True, device=self.model.device)
            self.state.current = dict(p["values_before"])
            self._apply_current()
        for key in p["keys"]:
            self.state.history[key] = asdict(KeyHistory(window=p["window"], outcome=outcome, judged_window=window))
        self.state.counts[outcome] = self.state.counts.get(outcome, 0) + 1
        entry = {"type": "judgement", "window": window, "change_window": p["window"], "keys": p["keys"],
                 "outcome": outcome, "reason": verdict["reason"], "judge": mode,
                 "probe_before": _probe_brief(p["probe_before"]), "probe_after": _probe_brief(probe)}
        if control_path is not None:
            entry["probe_control"] = _probe_brief(reference)
        self._log(entry)
        self.state.pending = None
        return reference if outcome == OUTCOME_ROLLED_BACK else None

    # ---------------- control branch ----------------
    def _launch_control(self, window: int, checkpoint: str, values_before: Dict[str, float]) -> Dict[str, Any]:
        """Start training ``checkpoint`` for this window under the unchanged config, in a separate process."""
        stem = self.ckpt_dir / f"control_w{window:04d}"
        config_path, out, log = stem.with_suffix(".json"), stem.with_suffix(".zip"), stem.with_suffix(".log")
        save_json(self._config_from(values_before).to_dict(), config_path)
        # SB3 stops at the first rollout at or past this count, exactly where the main run's next boundary falls.
        until = (window + 1) * self.window_steps
        cmd = [sys.executable, "-m", "racing_line_learner.in_loop.control", "--checkpoint", checkpoint,
               "--config", str(config_path), "--track", self.track.name, "--until", str(until),
               "--n_envs", str(int(getattr(self.training_env, "num_envs", 1))),
               "--seed", str(self.seed + 1000 * window + 7), "--out", str(out)]
        if "ent_coef" in values_before:
            cmd += ["--ent_coef", str(values_before["ent_coef"])]
        env = dict(os.environ)
        src = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        with open(log, "w") as fh:
            self._control_proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env)
        return {"model": str(out), "log": str(log), "until": until, "pid": self._control_proc.pid}

    def _wait_control(self, pending: Dict[str, Any]):
        """Wait for the pending change's control branch. Returns ``(model path, None)`` or ``(None, reason)``."""
        info = pending["control"]
        proc, self._control_proc = self._control_proc, None
        if proc is not None:
            try:
                code = proc.wait(timeout=self.control_timeout_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                return None, f"timed out after {self.control_timeout_s:.0f} s"
            if code != 0:
                return None, f"exit code {code} (log: {info['log']})"
        if Path(info["model"]).is_file():
            return info["model"], None
        # No process handle: the run was restarted while the branch was training, and the branch never finished.
        return None, "no control model (run restarted before the control branch finished)"

    def _propose_and_apply(self, window: int, report, findings, probe) -> None:
        history = self.state.key_history()
        from .llm import recent_decisions
        context = {"source_model": self.source_model, "track": self.track.name, "n_waypoints": self.track.n_waypoints,
                   "recent": recent_decisions(self.log_path, before_window=window)}
        t0 = time.time()
        error = None
        if hasattr(self.proposer, "last_meta"):
            self.proposer.last_meta = None
        try:
            raw = self.proposer.propose(report, findings, dict(self.state.current), history, window, context)
        except Exception as exc:                       # a failing proposer must never stop training
            raw, error = None, f"{type(exc).__name__}: {exc}"
        latency = time.time() - t0
        entry = {"type": "proposal", "window": window, "timesteps": self.num_timesteps, "proposer": self.proposer.name,
                 "raw": raw, "latency_s": round(latency, 3), "findings": [f["id"] for f in findings],
                 "probe": _probe_brief(probe)}
        if getattr(self.proposer, "last_meta", None):
            entry["llm"] = self.proposer.last_meta
        self.state.counts["proposals"] += 1
        if raw is None:
            entry.update(status="none" if error is None else "error", errors=[error] if error else [])
            self._log(entry)
            return
        try:
            rec = parse_recommendation(raw, self.track.n_waypoints, expected_track=self.track.name)
            if rec.schema_version != SCHEMA:
                raise SchemaError([f"schema_version must be {SCHEMA} for the in-loop supervisor"])
        except SchemaError as exc:
            self.state.counts["rejected"] += 1
            entry.update(status="rejected", errors=exc.errors)
            self._log(entry)
            return
        result = validate_proposal(rec, self.state.current, window, history, self.validator, findings=findings)
        if not result.accepted:
            self.state.counts["rejected"] += 1
            entry.update(status="rejected", errors=result.errors)
            self._log(entry)
            return
        if not rec.changes:
            entry.update(status="none", errors=[])
            self._log(entry)
            return
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt = str(self.ckpt_dir / f"pre_w{window:04d}.zip")
        self.model.save(ckpt)
        before = dict(self.state.current)
        self.state.current.update(rec.changes)
        self._apply_current()
        targets = sorted({g for gs in rec.grounds.values() for g in gs})
        self.state.pending = {"window": window, "keys": sorted(rec.changes), "values_before": before,
                              "checkpoint": ckpt, "probe_before": probe, "targets": targets}
        if self.judge_mode == "control":
            self.state.pending["control"] = self._launch_control(window, ckpt, before)
        for key in rec.changes:
            self.state.history[key] = asdict(KeyHistory(window=window, outcome=OUTCOME_PENDING))
        self.state.counts["accepted"] += 1
        entry.update(status="accepted", errors=[], applied=rec.changes, grounds={k: list(v) for k, v in rec.grounds.items()},
                     values_before={k: before.get(k) for k in rec.changes})
        self._log(entry)


def probe_policy(model, track: Track, config: EnvConfig, probe_runs: int = 8, max_sim_steps: int = 30000,
                 seed: int = 0) -> Dict[str, Any]:
    """Deterministic jittered out-laps under ``config``: completion, lap time, smoothness, per-segment line spread."""
    from ..env.wrapper import make_env
    from ..eval.laps import eval_config
    cfg = replace(eval_config(config, "line", laps=1), max_episode_steps=max_sim_steps)
    env = make_env(track, cfg, seed=seed)
    collector = TelemetryCollector(track, cfg)
    rng = np.random.RandomState(seed)
    base = track.spawn_pose(track.outlap_start_waypoint())
    runs = []
    try:
        for k in range(probe_runs):
            pose = base if k == 0 else (base[0] + rng.normal(0, 0.05), base[1] + rng.normal(0, 0.05),
                                        base[2] + rng.normal(0, 0.02))
            obs = env.reset(pose=pose)
            info: Dict[str, Any] = {}
            for _ in range(max_sim_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                collector.update(info, reward)
                if done:
                    break
            runs.append({"spawn": None, "termination": info.get("termination"),
                         "lap_time": info.get("lap_time"), "waypoint": info.get("current_waypoint_idx")})
    finally:
        env.close()
    probe = summarize_probe(runs)
    laps = [r["lap_time"] for r in runs if r["termination"] == "lap_complete" and r["lap_time"] is not None]
    probe["lap_time_std"] = round(float(np.std(laps)), 3) if len(laps) > 1 else None
    rep = collector.report()
    probe["smoothness"] = rep["smoothness"]
    probe["segments"] = [{k: s[k] for k in ("name", "waypoints", "is_corner", "lateral_std", "pass_time_s")}
                         for s in rep["segments"]]
    return probe


def _probe_brief(p: Dict[str, Any]) -> Dict[str, Any]:
    sm = p.get("smoothness") or {}
    return {"completion_rate": p.get("completion_rate"), "lap_time_mean": p.get("lap_time_mean"),
            "lap_time_std": p.get("lap_time_std"), "steer_reversals_per_s": sm.get("steer_reversals_per_s"),
            "steer_change_per_step": sm.get("steer_change_per_step"),
            "throttle_change_per_step": sm.get("throttle_change_per_step"), "lateral_std_m": sm.get("lateral_std_m")}


def make_proposer(name: str, replay_log: Optional[str] = None, validator: ValidatorConfig = ValidatorConfig(),
                  llm: Optional[Dict[str, Any]] = None) -> Proposer:
    """Build a proposer. ``llm`` (for ``name="llm"``): backend, model, url, think, temperature, timeout, retries, seed,
    prompt (v1/v2/v3)."""
    if name == "llm":
        from .llm import LLMProposer, make_backend
        cfg = dict(llm or {})
        if not cfg.get("model"):
            raise ValueError("--supervisor llm needs --llm_model")
        retries = int(cfg.pop("retries", 1))
        constrain = bool(cfg.pop("constrain", True))
        prompt = cfg.pop("prompt", "v1")          # runs recorded before prompt versions existed used v1
        backend = make_backend(cfg.pop("backend", "ollama"), cfg.pop("model"), cfg.pop("url", None),
                               **{k: v for k, v in cfg.items() if v is not None})
        return LLMProposer(backend, validator=validator, max_retries=retries, constrain=constrain, prompt=prompt)
    if name == "stub":
        return StubProposer()
    if name == "rules":
        return RuleBasedProposer(validator)
    if name == "replay":
        if not replay_log:
            raise ValueError("--supervisor replay needs --replay_log")
        return ReplayProposer(replay_log)
    raise ValueError(f"unknown proposer {name!r} (stub, rules, replay, llm)")
