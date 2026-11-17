"""LLM proposer (prompt, schema, backends, retry), benchmark harness, and the supervisor wired to an LLM. No server needed."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from racing_line_learner.config import get_preset
from racing_line_learner.in_loop import bench
from racing_line_learner.in_loop.llm import (LLMError, LLMProposer, LLMReply, OllamaBackend, OpenAIBackend,
                                             build_user_prompt, extract_json, lever_states, make_backend,
                                             recent_decisions, response_schema)
from racing_line_learner.in_loop.validator import OUTCOME_IMPROVED, OUTCOME_ROLLED_BACK, KeyHistory
from racing_line_learner.schema import WHITELIST_V2
from tests.sim import HAVE_SB3, requires_sb3

V2 = get_preset("v2")
CURRENT = {k: float(getattr(V2, k)) for k in WHITELIST_V2 if hasattr(V2, k)}
CURRENT["ent_coef"] = 0.01
WEAVING = {"id": "weaving", "severity": "medium", "message": "steering reverses 9/s", "waypoints": None, "facts": {}}
THROTTLE = {"id": "throttle_jitter", "severity": "medium", "message": "speed command jitters", "waypoints": None,
            "facts": {}}
REPORT = {"window": 3, "timesteps": 750_000, "track": "cota", "n_waypoints": 150,
          "probe": {"episodes": 8, "completion_rate": 0.75, "lap_time_mean": 43.9, "lap_time_std": 0.3,
                    "lap_time_best": 43.5, "smoothness": {"steer_reversals_per_s": 9.0}, "failures_at": []},
          "terminations": {"collision": {"count": 9, "share": 0.9}}, "speed": {"mean": 8.0}}
CTX = {"source_model": "m", "track": "cota", "n_waypoints": 150}


class FakeBackend:
    """Returns canned replies in order and records the messages it was sent."""

    name, model = "fake", "fake-model"

    def __init__(self, *texts):
        self.texts, self.calls = list(texts), []

    def chat(self, messages, schema):
        self.calls.append({"messages": messages, "schema": schema})
        return LLMReply(text=self.texts.pop(0), prompt_tokens=100, completion_tokens=20, latency_s=0.5)


def reply(changes, grounds, rationale="r"):
    return json.dumps({"changes": changes, "grounds": grounds, "rationale": rationale})


class TestLeverStatesAndPrompt(unittest.TestCase):
    def test_availability_reach_and_locks(self):
        hist = {"jitter_coef": KeyHistory(window=1, outcome=OUTCOME_ROLLED_BACK, judged_window=2)}
        states = {s.key: s for s in lever_states(CURRENT, [WEAVING, THROTTLE], hist, window=3)}
        self.assertFalse(states["jitter_coef"].available)                    # rolled back: locked until window 5
        self.assertIn("rolled back", states["jitter_coef"].lock_reason)
        self.assertTrue(states["steering_penalty_w"].available)             # weaving justifies it
        self.assertFalse(states["time_cost"].available)                     # no finding justifies it
        tc = states["throttle_change_coef"]
        self.assertEqual((tc.reach_low, tc.reach_high), (0.0, 0.25))         # 25% of 0..1 from 0
        self.assertEqual(states["throttle_change_coef"].justified_by, ("throttle_jitter",))

    def test_value_outside_range_is_locked(self):
        cur = dict(CURRENT, throttle_change_coef=2.0)                      # e.g. an old lever-check run
        st = {s.key: s for s in lever_states(cur, [THROTTLE], {}, window=3)}["throttle_change_coef"]
        self.assertFalse(st.available)
        self.assertIn("outside the allowed range", st.lock_reason)

    def test_prompt_mentions_findings_levers_and_recent(self):
        levers = lever_states(CURRENT, [WEAVING], {}, window=3)
        recent = [{"window": 2, "applied": {"jitter_coef": 12.5}, "before": {"jitter_coef": 0.05},
                   "grounds": {"jitter_coef": ["weaving"]}, "outcome": "improved", "reason": "smaller corrections"}]
        text = build_user_prompt(REPORT, [WEAVING], levers, 3, recent)
        for needle in ("completed 6/8", "weaving [medium]", "jitter_coef: 0.05", "available", "no current finding",
                       "window 2: jitter_coef 0.05 -> 12.5", "smaller corrections"):
            self.assertIn(needle, text)

    def test_schema_constrained_and_loose(self):
        levers = lever_states(CURRENT, [WEAVING], {}, window=3)
        sch = response_schema(levers)
        props = sch["properties"]["changes"]["properties"]
        self.assertEqual(sorted(props), ["jitter_coef", "steering_penalty_w"])
        self.assertAlmostEqual(props["jitter_coef"]["maximum"], 0.05 + 0.25 * (50 - 0.01), places=4)
        self.assertEqual(sch["properties"]["grounds"]["properties"]["jitter_coef"]["items"]["enum"], ["weaving"])
        self.assertNotIn("properties", response_schema(levers, constrain=False)["properties"]["changes"])


class TestExtractJson(unittest.TestCase):
    def test_variants(self):
        self.assertEqual(extract_json('{"a": 1}'), {"a": 1})
        self.assertEqual(extract_json('<think>maybe {"x": 0}</think>\n```json\n{"a": 2}\n```'), {"a": 2})
        self.assertEqual(extract_json('Here you go: {"a": {"b": 3}} done'), {"a": {"b": 3}})
        with self.assertRaises(ValueError):
            extract_json("no json {oops")


class TestBackends(unittest.TestCase):
    def test_ollama_payload_and_reply(self):
        seen = {}

        def transport(url, payload, timeout, headers):
            seen.update(url=url, payload=payload)
            return {"message": {"content": "{}", "thinking": "hmm"}, "prompt_eval_count": 7, "eval_count": 3}

        b = OllamaBackend(model="qwen3:8b", think=False, seed=1, transport=transport)
        r = b.chat([{"role": "user", "content": "hi"}], {"type": "object"})
        self.assertEqual(seen["url"], "http://localhost:11434/api/chat")
        self.assertEqual(seen["payload"]["format"], {"type": "object"})
        self.assertIs(seen["payload"]["think"], False)
        self.assertFalse(seen["payload"]["stream"])
        self.assertEqual(seen["payload"]["options"]["seed"], 1)
        self.assertEqual((r.text, r.thinking, r.prompt_tokens, r.completion_tokens), ("{}", "hmm", 7, 3))

    def test_openai_payload_and_reply(self):
        seen = {}

        def transport(url, payload, timeout, headers):
            seen.update(url=url, payload=payload, headers=headers)
            return {"choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}

        b = make_backend("openai", "m", "http://h:8080/v1/", think=True, api_key="k", transport=transport)
        r = b.chat([], {"type": "object"})
        self.assertIsInstance(b, OpenAIBackend)
        self.assertEqual(seen["url"], "http://h:8080/v1/chat/completions")
        self.assertEqual(seen["payload"]["response_format"]["json_schema"]["schema"], {"type": "object"})
        self.assertEqual(seen["headers"], {"Authorization": "Bearer k"})
        self.assertEqual((r.prompt_tokens, r.completion_tokens), (5, 2))
        with self.assertRaises(ValueError):
            make_backend("nope", "m")


class TestLLMProposer(unittest.TestCase):
    def propose(self, backend, findings=(WEAVING,), history=None, window=3):
        return LLMProposer(backend).propose(REPORT, list(findings), dict(CURRENT), history or {}, window, CTX)

    def test_valid_reply_becomes_schema_v2(self):
        p = LLMProposer(FakeBackend(reply({"jitter_coef": 10.0}, {"jitter_coef": ["weaving"]})))
        doc = p.propose(REPORT, [WEAVING], dict(CURRENT), {}, 3, CTX)
        self.assertEqual(doc["schema_version"], 2)
        self.assertEqual(doc["changes"], {"jitter_coef": 10.0})
        self.assertEqual(doc["track"], "cota")
        self.assertEqual(len(p.last_meta["attempts"]), 1)
        self.assertEqual(p.last_meta["attempts"][0]["errors"], [])

    def test_retry_feeds_back_validator_errors(self):
        backend = FakeBackend(reply({"time_cost": 0.12}, {"time_cost": ["weaving"]}),       # ungrounded
                              reply({"steering_penalty_w": 2.5}, {"steering_penalty_w": ["weaving"]}))
        p = LLMProposer(backend)
        doc = p.propose(REPORT, [WEAVING], dict(CURRENT), {}, 3, CTX)
        self.assertEqual(doc["changes"], {"steering_penalty_w": 2.5})
        self.assertEqual(len(p.last_meta["attempts"]), 2)
        self.assertTrue(p.last_meta["attempts"][0]["errors"])
        feedback = backend.calls[1]["messages"][-1]["content"]
        self.assertIn("rejected by the validator", feedback)
        self.assertIn("time_cost", feedback)

    def test_still_invalid_after_retry_is_returned_for_the_supervisor_to_reject(self):
        bad = reply({"jitter_coef": 49.0}, {"jitter_coef": ["weaving"]})                     # beyond the step cap
        p = LLMProposer(FakeBackend(bad, bad))
        doc = p.propose(REPORT, [WEAVING], dict(CURRENT), {}, 3, CTX)
        self.assertEqual(doc["changes"], {"jitter_coef": 49.0})
        self.assertIn("exceeds per-window cap", " ".join(p.last_meta["attempts"][-1]["errors"]))

    def test_no_json_raises(self):
        with self.assertRaises(LLMError):
            self.propose(FakeBackend("I would raise the jitter penalty.", "still no json"))

    def test_empty_changes_is_a_valid_abstention(self):
        doc = self.propose(FakeBackend(reply({}, {})))
        self.assertEqual(doc["changes"], {})


def _write_run(tmp):
    """A fake supervised run: window 1 change accepted, rolled back in window 2; reports for windows 1-3."""
    run = Path(tmp) / "fake_run"
    (run / "telemetry").mkdir(parents=True)
    config = {k: getattr(V2, k) for k in WHITELIST_V2 if hasattr(V2, k)}
    for w in (1, 2, 3):
        rep = dict(REPORT, window=w, config=config, findings=[WEAVING, THROTTLE])
        (run / "telemetry" / f"w{w:04d}.json").write_text(json.dumps(rep))
    log = [
        {"type": "proposal", "window": 1, "status": "accepted", "applied": {"jitter_coef": 12.5},
         "grounds": {"jitter_coef": ["weaving"]}},
        {"type": "judgement", "window": 2, "change_window": 1, "keys": ["jitter_coef"], "outcome": "rolled_back",
         "reason": "completion fell"},
        {"type": "proposal", "window": 2, "status": "accepted", "applied": {"throttle_change_coef": 0.25},
         "grounds": {"throttle_change_coef": ["throttle_jitter"]}, "values_before": {"throttle_change_coef": 0.0}},
        {"type": "judgement", "window": 3, "change_window": 2, "keys": ["throttle_change_coef"], "outcome": "improved",
         "reason": "smoother"},
    ]
    (run / "proposals.jsonl").write_text("\n".join(json.dumps(e) for e in log) + "\n")
    return run


class TestBench(unittest.TestCase):
    def test_cases_replay_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = {c["window"]: c for c in bench.build_cases([str(_write_run(tmp))])}
        self.assertEqual(sorted(cases), [1, 2, 3])
        self.assertEqual(cases[1]["history"], {})
        c2 = cases[2]                                           # deciding window 2: after the rollback
        self.assertAlmostEqual(c2["current"]["jitter_coef"], 0.05)
        self.assertEqual(c2["history"]["jitter_coef"]["outcome"], OUTCOME_ROLLED_BACK)
        self.assertEqual(c2["recent"][0]["before"], {"jitter_coef": 0.05})
        c3 = cases[3]
        self.assertAlmostEqual(c3["current"]["throttle_change_coef"], 0.25)
        self.assertEqual(c3["history"]["throttle_change_coef"]["outcome"], OUTCOME_IMPROVED)
        self.assertEqual([d["window"] for d in c3["recent"]], [1, 2])

    def test_direction(self):
        cur = {"jitter_coef": 1.0, "grip_margin": 1.0}
        self.assertTrue(bench.direction_ok({"changes": {"jitter_coef": 2.0}, "grounds": {"jitter_coef": ["weaving"]}}, cur))
        self.assertFalse(bench.direction_ok({"changes": {"jitter_coef": 0.5}, "grounds": {"jitter_coef": ["weaving"]}}, cur))
        self.assertFalse(bench.direction_ok({"changes": {"grip_margin": 1.2},
                                             "grounds": {"grip_margin": ["crash_hotspot"]}}, cur))
        self.assertIsNone(bench.direction_ok({"changes": {}}, cur))

    @unittest.skipUnless(HAVE_SB3, "rule-based reference needs stable_baselines3")
    def test_run_scores_llm_and_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = bench.build_cases([str(_write_run(tmp))])
        good = reply({"steering_penalty_w": 2.5}, {"steering_penalty_w": ["weaving"]})
        llm = LLMProposer(FakeBackend(good, "garbage", "garbage", good))
        rows = bench.run(cases, llm, log=lambda *_: None)
        self.assertEqual([r["accepted"] for r in rows], [True, False, True])
        self.assertFalse(rows[1]["json_ok"])
        self.assertTrue(rows[0]["direction_ok"])
        self.assertEqual(rows[0]["rules_changes"], ["jitter_coef", "throttle_change_coef"])
        s = bench.summarize(rows)
        self.assertEqual((s["n"], s["accepted"], s["json_ok"]), (3, 0.667, 0.667))
        self.assertEqual(rows[0]["useful"], True)
        self.assertEqual(rows[1]["useful"], False)
        from racing_line_learner.in_loop.supervisor import RuleBasedProposer
        rule_rows = bench.run(cases, RuleBasedProposer(), log=lambda *_: None)
        self.assertTrue(all(r["accepted"] for r in rule_rows))
        self.assertEqual(bench.summarize(rule_rows)["rules_overlap"], 1.0)

    @unittest.skipUnless(HAVE_SB3, "rule-based reference needs stable_baselines3")
    def test_useful_metric_ranks_bad_proposers_low(self):
        with tempfile.TemporaryDirectory() as tmp:
            cases = bench.build_cases([str(_write_run(tmp))])

        def score(text):
            rows = bench.run(cases, LLMProposer(FakeBackend(*[text] * 10), max_retries=0), log=lambda *_: None)
            return bench.summarize(rows)["useful"]

        right = reply({"steering_penalty_w": 2.5}, {"steering_penalty_w": ["weaving"]})
        backwards = reply({"steering_penalty_w": 1.5}, {"steering_penalty_w": ["weaving"]})   # legal, but backwards
        self.assertEqual(score(right), 1.0)
        self.assertEqual(score(backwards), 0.0)
        self.assertEqual(score(reply({}, {})), 0.0)                                           # always abstains

    def test_recent_decisions_window_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = _write_run(tmp) / "proposals.jsonl"
            self.assertEqual(recent_decisions(log, before_window=1), [])
            d = recent_decisions(log, before_window=2)
            self.assertEqual((len(d), d[0]["outcome"]), (1, "rolled_back"))


@requires_sb3
class TestSnapshot(unittest.TestCase):
    def test_snapshot_from_checkpoint(self):
        from racing_line_learner.train.ppo import main
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            main(["--run_name", "snap", "--output_dir", tmp, "--preset", "v2", "--timesteps", "1024",
                  "--n_steps", "512", "--batch_size", "128", "--checkpoint_freq", "1024"])
            ckpt = sorted((Path(tmp) / "runs" / "snap" / "checkpoints").glob("*_steps.zip"))[-1]
            out = bench.snapshot(str(ckpt), str(Path(tmp) / "snaps"), rollout_steps=300, probe_runs=1,
                                 probe_max_sim_steps=300)
            rep = json.loads(out.read_text())
            cases = bench.build_cases([str(out.parent.parent)])
        self.assertEqual(rep["window"], 1)
        self.assertTrue(rep["source"]["offline_snapshot"])
        self.assertIn("findings", rep)
        self.assertEqual(rep["probe"]["episodes"], 1)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["history"], {})
        self.assertAlmostEqual(cases[0]["current"]["ent_coef"], rep["config"]["ent_coef"])


class TestCli(unittest.TestCase):
    def test_train_args_record_llm_spec(self):
        from racing_line_learner.train.ppo import build_run_spec, parse_args
        spec = build_run_spec(parse_args(["--run_name", "r", "--preset", "v2", "--supervisor", "llm",
                                          "--llm_model", "qwen3:8b", "--llm_think", "off"]))
        self.assertEqual(spec["supervisor"]["llm"]["model"], "qwen3:8b")
        self.assertIs(spec["supervisor"]["llm"]["think"], False)
        rules = build_run_spec(parse_args(["--run_name", "r", "--preset", "v2", "--supervisor", "rules"]))
        self.assertNotIn("llm", rules["supervisor"])            # older supervised runs still resume
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--run_name", "r", "--supervisor", "llm"])

    def test_make_proposer_llm(self):
        if not HAVE_SB3:
            self.skipTest("supervisor module needs stable_baselines3")
        from racing_line_learner.in_loop.supervisor import make_proposer
        p = make_proposer("llm", llm={"backend": "ollama", "model": "m", "think": True, "retries": 0,
                                      "temperature": 0.1, "timeout": 5.0, "seed": 0, "url": None})
        self.assertEqual((p.name, p.max_retries, p.backend.think, p.backend.model), ("llm", 0, True, "m"))


@requires_sb3
class TestSupervisedTrainingWithLLM(unittest.TestCase):
    def test_llm_proposals_logged(self):
        from racing_line_learner.in_loop import supervisor as sv
        from racing_line_learner.train.ppo import main
        good = reply({"steering_penalty_w": 2.2}, {"steering_penalty_w": ["weaving"]})
        backend = FakeBackend(*[good] * 20)
        old_make, old_detect = sv.make_proposer, sv.detect
        sv.make_proposer = lambda name, replay_log=None, validator=None, llm=None: LLMProposer(backend)
        sv.detect = lambda report: old_detect(report) + [dict(WEAVING)]
        args = ["--preset", "v2", "--timesteps", "4096", "--n_envs", "2", "--n_steps", "512", "--batch_size", "128",
                "--checkpoint_freq", "4096", "--window_steps", "2048", "--probe_runs", "1",
                "--probe_max_sim_steps", "300", "--supervisor", "llm", "--llm_model", "fake"]
        try:
            with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                main(["--run_name", "svllm", "--output_dir", tmp] + args)
                log = [json.loads(l) for l in (Path(tmp) / "runs" / "svllm" / "proposals.jsonl").read_text().splitlines()]
        finally:
            sv.make_proposer, sv.detect = old_make, old_detect
        first = [e for e in log if e["type"] == "proposal"][0]
        self.assertEqual(first["status"], "accepted")
        self.assertEqual(first["llm"]["model"], "fake-model")
        self.assertEqual(first["values_before"], {"steering_penalty_w": 2.0})
        self.assertIn("Findings", backend.calls[0]["messages"][1]["content"])


if __name__ == "__main__":
    unittest.main()
