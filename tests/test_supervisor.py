"""Step 3: the in-loop supervisor (judge, rule-based proposer, replay, state, end-to-end with rollback and resume)."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from racing_line_learner.config import get_preset
from racing_line_learner.env.tracks import load_track
from racing_line_learner.in_loop import supervisor as sv
from racing_line_learner.in_loop.supervisor import (JudgeConfig, ReplayProposer, RuleBasedProposer, SupervisorState,
                                                    judge, lever_values)
from racing_line_learner.in_loop.validator import (OUTCOME_IMPROVED, OUTCOME_NEUTRAL, OUTCOME_PENDING,
                                                   OUTCOME_ROLLED_BACK, KeyHistory, ValidatorConfig,
                                                   validate_proposal)
from racing_line_learner.schema import WHITELIST_V2, parse_recommendation
from tests.sim import requires_sb3

COTA = load_track("cota")
V2 = get_preset("v2")


def probe(c=0.75, t=43.0, std=0.1, rev=9.0, thr=0.8):
    return {"completion_rate": c, "lap_time_mean": t, "lap_time_std": std,
            "smoothness": {"steer_reversals_per_s": rev, "throttle_change_per_step": thr}}


class TestJudge(unittest.TestCase):
    def test_completion_drop_rolls_back(self):
        self.assertEqual(judge(probe(c=1.0), probe(c=0.5))["outcome"], OUTCOME_ROLLED_BACK)

    def test_slower_lap_rolls_back_beyond_noise_only(self):
        self.assertEqual(judge(probe(t=43.0), probe(t=44.2))["outcome"], OUTCOME_NEUTRAL)       # within 1.5 s
        self.assertEqual(judge(probe(t=43.0), probe(t=45.0))["outcome"], OUTCOME_ROLLED_BACK)
        noisy = judge(probe(t=43.0, std=1.0), probe(t=45.0, std=1.0))                           # margin 2.0 s
        self.assertEqual(noisy["outcome"], OUTCOME_NEUTRAL)

    def test_improvements(self):
        self.assertEqual(judge(probe(t=43.0), probe(t=41.0))["outcome"], OUTCOME_IMPROVED)
        self.assertEqual(judge(probe(c=0.25), probe(c=0.75))["outcome"], OUTCOME_IMPROVED)

    def test_pace_first_once_reliable(self):
        # Reliable before: a clear lap-time loss is rolled back even though completion rose.
        out = judge(probe(c=0.5, t=43.0), probe(c=1.0, t=46.0))
        self.assertEqual(out["outcome"], OUTCOME_ROLLED_BACK)
        self.assertIn("although completion rose", out["reason"])
        # Not yet reliable: the reliability gain counts, the pace loss is accepted.
        self.assertEqual(judge(probe(c=0.12, t=44.4), probe(c=0.75, t=46.6))["outcome"], OUTCOME_IMPROVED)
        self.assertEqual(judge(probe(c=0.5, t=43.0), probe(c=1.0, t=44.0))["outcome"], OUTCOME_IMPROVED)  # within margin
        before, after = probe(), probe()
        before["smoothness"]["steer_change_per_step"], after["smoothness"]["steer_change_per_step"] = 0.075, 0.056
        self.assertEqual(judge(before, after, targets=["weaving"])["outcome"], OUTCOME_IMPROVED)  # size, not count
        self.assertEqual(judge(before, after)["outcome"], OUTCOME_NEUTRAL)                       # not the target
        self.assertEqual(judge(probe(rev=9.0), probe(rev=7.5), targets=["weaving"])["outcome"], OUTCOME_NEUTRAL)

    def test_missing_lap_times(self):
        self.assertEqual(judge({"completion_rate": 0.0}, {"completion_rate": 0.0})["outcome"], OUTCOME_NEUTRAL)


class TestRuleBasedProposer(unittest.TestCase):
    CURRENT = lever_values(V2, 0.01)

    def propose(self, ids, history=None, window=3, current=None):
        findings = [{"id": i} for i in ids]
        raw = RuleBasedProposer().propose({}, findings, current or dict(self.CURRENT), history or {}, window,
                                          {"source_model": "m", "track": "cota"})
        return raw, findings

    def test_grounded_cap_sized_and_valid(self):
        raw, findings = self.propose(["weaving", "throttle_jitter", "inconsistent_line"])
        self.assertEqual(set(raw["changes"]), {"jitter_coef", "throttle_change_coef"})        # priority order, max 2
        self.assertEqual(raw["grounds"], {"jitter_coef": ["weaving"], "throttle_change_coef": ["throttle_jitter"]})
        rec = parse_recommendation(raw, COTA.n_waypoints, expected_track="cota")
        res = validate_proposal(rec, self.CURRENT, 3, {}, ValidatorConfig(), findings=findings)
        self.assertTrue(res.accepted, res.errors)

    def test_respects_cooldowns_and_limits(self):
        hist = {"jitter_coef": KeyHistory(window=2, outcome=OUTCOME_PENDING)}
        raw, _ = self.propose(["weaving", "throttle_jitter"], history=hist)
        self.assertNotIn("jitter_coef", raw["changes"])
        at_max = dict(self.CURRENT, jitter_coef=WHITELIST_V2["jitter_coef"].high)
        raw, _ = self.propose(["weaving"], current=at_max)
        self.assertIsNone(raw)

    def test_improved_key_can_move_again(self):
        hist = {"jitter_coef": KeyHistory(window=2, outcome=OUTCOME_IMPROVED, judged_window=3)}
        raw, _ = self.propose(["weaving"], history=hist)
        self.assertIn("jitter_coef", raw["changes"])

    def test_probe_failures_come_first(self):
        raw, _ = self.propose(["weaving", "probe_failures", "throttle_jitter"])
        self.assertEqual(list(raw["changes"])[0], "spawn_corner_mix")
        self.assertEqual(raw["grounds"]["spawn_corner_mix"], ["probe_failures"])
        self.assertEqual(len(raw["changes"]), 2)

    def test_nothing_to_do(self):
        self.assertIsNone(self.propose([])[0])


class TestReplayAndState(unittest.TestCase):
    def test_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "proposals.jsonl"
            log.write_text("\n".join(json.dumps(e) for e in [
                {"type": "proposal", "window": 1, "raw": {"changes": {"jitter_coef": 0.1}}},
                {"type": "judgement", "window": 2},
                {"type": "proposal", "window": 2, "raw": None}]) + "\n")
            rp = ReplayProposer(str(log))
            self.assertEqual(rp.propose({}, [], {}, {}, 1, {}), {"changes": {"jitter_coef": 0.1}})
            self.assertIsNone(rp.propose({}, [], {}, {}, 2, {}))

    def test_state_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            st = SupervisorState(window=3, current={"jitter_coef": 0.1},
                                 history={"jitter_coef": {"window": 3, "outcome": "pending", "judged_window": None}})
            st.save(Path(tmp) / "s.json")
            again = SupervisorState.load(Path(tmp) / "s.json")
            self.assertEqual(again, st)
            self.assertEqual(again.key_history()["jitter_coef"].outcome, OUTCOME_PENDING)

    def test_rejects_legacy_preset(self):
        from racing_line_learner.config import EnvConfig
        with self.assertRaises(ValueError):
            sv.SupervisorCallback(COTA, EnvConfig(), sv.StubProposer(), "/tmp/x")


class AlwaysJitter(sv.Proposer):
    name = "always_jitter"

    def propose(self, report, findings, current, history, window, context):
        return {"schema_version": 2, "source_model": "m", "track": "cota",
                "changes": {"jitter_coef": round(current["jitter_coef"] + 0.1, 6)}, "grounds": {"jitter_coef": ["weaving"]}}


@requires_sb3
class TestSupervisedTraining(unittest.TestCase):
    ARGS = ["--preset", "v2", "--timesteps", "8192", "--n_envs", "2", "--n_steps", "512", "--batch_size", "128",
            "--checkpoint_freq", "4096", "--window_steps", "2048", "--probe_runs", "1", "--probe_max_sim_steps", "300"]

    def train(self, tmp, *extra):
        from racing_line_learner.train.ppo import main
        with contextlib.redirect_stdout(io.StringIO()) as out:
            main(["--run_name", "sv", "--output_dir", tmp] + self.ARGS + list(extra))
        run = Path(tmp) / "runs" / "sv"
        log = [json.loads(l) for l in (run / "proposals.jsonl").read_text().splitlines()]
        return run, log, out.getvalue()

    def test_rollback_and_resume(self):
        old_judge, old_make = sv.judge, sv.make_proposer
        sv.judge = lambda before, after, targets=(), config=None: {"outcome": OUTCOME_ROLLED_BACK, "reason": "forced"}
        sv.make_proposer = lambda name, replay_log=None, validator=None, llm=None: AlwaysJitter()
        # Every proposal cites "weaving"; make sure the finding is present so grounding passes.
        old_detect = sv.detect
        sv.detect = lambda report: old_detect(report) + [{"id": "weaving", "severity": "medium", "message": "",
                                                          "waypoints": None, "facts": {}}]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                run, log, _ = self.train(tmp, "--supervisor", "rules")
                state = json.loads((run / "supervisor_state.json").read_text())
                props = [e for e in log if e["type"] == "proposal"]
                judged = [e for e in log if e["type"] == "judgement"]
                self.assertEqual([e["window"] for e in props], [1, 2, 3, 4])
                self.assertEqual(props[0]["status"], "accepted")
                self.assertTrue(all(j["outcome"] == OUTCOME_ROLLED_BACK for j in judged))
                # After a rollback the key is locked for 2 more windows, so the next proposals are rejected.
                self.assertEqual(props[1]["status"], "rejected")
                self.assertIn("rolled back", " ".join(props[1]["errors"]))
                self.assertAlmostEqual(state["current"]["jitter_coef"], 0.05)              # restored
                self.assertGreaterEqual(state["counts"]["rolled_back"], 1)
                self.assertTrue((run / "supervisor_checkpoints" / "pre_w0001.zip").exists())
                # A change proposed right after a rollback is judged against the restored checkpoint's probe.
                after_rb = [e for e in log if e["type"] == "proposal" and e["window"] == 2][0]
                first = [e for e in log if e["type"] == "proposal" and e["window"] == 1][0]
                self.assertEqual(after_rb["probe"], first["probe"])

                # Resume after a "disconnect": drop the final checkpoint, rerun the same command.
                final = sorted((run / "checkpoints").glob("*.zip"))[-1]
                final.unlink()
                n_before = len(log)
                _, log2, out = self.train(tmp, "--supervisor", "rules")
                self.assertIn("Resuming sv", out)
                self.assertEqual(len(log2), n_before)                                   # no decision re-applied
        finally:
            sv.judge, sv.make_proposer, sv.detect = old_judge, old_make, old_detect

    def test_stub_changes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, log, _ = self.train(tmp, "--supervisor", "stub")
            state = json.loads((run / "supervisor_state.json").read_text())
            self.assertEqual(state["current"], lever_values(V2, 0.01))
            self.assertTrue(all(e["status"] == "none" for e in log if e["type"] == "proposal"))
            self.assertEqual(len(list((run / "telemetry").glob("w*.json"))), 4)


if __name__ == "__main__":
    unittest.main()
