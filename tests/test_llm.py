"""Tests for src/post_loop/llm.py. No GPU or model needed.

Run from the repo root:  python -m unittest tests.test_llm
"""
import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.post_loop.llm import (
    CAUSES,
    DummyBackend,
    OpenAICompatBackend,
    build_json_schema,
    build_messages,
    recommend,
    score,
    summarize_analysis,
)

ANALYSIS = {
    "source_model": "m1",
    "track": "cota",
    "summary": {"laps": 3, "terminations": {"missed_waypoint": 2, "crash": 1},
                "completion_rate": 0.0, "mean_waypoints_cleared": 120.0},
    "track_info": {"num_waypoints": 150, "duplicate_waypoints": [[70, 71]]},
    "crash_segments": [{"waypoints": [44, 44], "crashes": 1, "laps_reaching": 3,
                        "crash_rate": 0.33, "mean_speed": 6.5, "blind_fraction": 0.0,
                        "max_curvature": 0.09}],
    "crashes": [{"lap": 2, "mode": "perturbed", "waypoint": 44, "speed": 6.5,
                 "steer_raw": 0.157, "steer_smoothed": 0.066, "corner_radius_m": 10.79,
                 "heading_change_deg": 192.0,
                 "visibility": {"blind": False, "visible_m": 1.48, "required_brake_m": 2.22}}],
    "stuck": [],
    "missed_waypoints": [
        {"lap": 0, "waypoint": 146, "closest_approach_m": 1.75, "capture_radius_m": 1.5,
         "distance_driven_after_m": 129.9, "mean_speed_last_2s": 6.5},
        {"lap": 1, "waypoint": 146, "closest_approach_m": 1.61, "capture_radius_m": 1.5,
         "distance_driven_after_m": 129.9, "mean_speed_last_2s": 6.5},
    ],
}
EXPECTED = {146: "misplaced_waypoint", 44: "spin"}


class TestPromptAndSchema(unittest.TestCase):
    def test_summary_groups_missed_waypoints_and_copies_numbers(self):
        s = summarize_analysis(ANALYSIS)
        self.assertEqual(len(s["missed_waypoints"]), 1)
        g = s["missed_waypoints"][0]
        self.assertEqual(g["laps"], [0, 1])
        self.assertEqual((g["closest_approach_m_min"], g["closest_approach_m_max"]), (1.61, 1.75))
        self.assertEqual(s["crashes"][0]["heading_change_deg"], 192.0)

    def test_schema_pins_run_and_causes(self):
        sch = build_json_schema(ANALYSIS)
        self.assertEqual(sch["properties"]["track"]["enum"], ["cota"])
        self.assertEqual(sch["properties"]["source_model"]["enum"], ["m1"])
        diag = sch["properties"]["diagnoses"]["items"]["properties"]
        self.assertEqual(diag["cause"]["enum"], list(CAUSES))
        self.assertEqual(diag["waypoints"]["items"]["maximum"], 149)
        self.assertEqual(sch["properties"]["changes"]["properties"]["speed_floor"]["minimum"], 6.5)

    def test_prompt_contains_data(self):
        msgs = build_messages(ANALYSIS)
        self.assertIn("speed_floor: current 6.5, allowed 6.5 to 8.0", msgs[0]["content"])
        self.assertIn('"heading_change_deg": 192.0', msgs[1]["content"])

    def test_prompt_lists_every_failure(self):
        user = build_messages(ANALYSIS)[1]["content"]
        self.assertIn("- crash at waypoint 44 (lap 2)", user)
        self.assertIn("- missed waypoint 146 (laps [0, 1])", user)
        self.assertIn("EVERY failure", build_messages(ANALYSIS)[0]["content"])

    def test_default_url_is_8081(self):
        self.assertEqual(OpenAICompatBackend().base_url, "http://127.0.0.1:8081/v1")


class TestDummyBackend(unittest.TestCase):
    def test_dummy_gets_known_answers(self):
        r = recommend(ANALYSIS, DummyBackend())
        self.assertIsNotNone(r["recommendation"])
        self.assertEqual(len(r["attempts"]), 1)
        self.assertTrue(all(s["correct"] for s in score(r["recommendation"], EXPECTED).values()))


class TestScore(unittest.TestCase):
    def rec(self, *diags):
        return {"diagnoses": [{"waypoints": w, "cause": c} for w, c in diags]}

    def test_range_and_wrap(self):
        s = score(self.rec(([140, 5], "misplaced_waypoint"), ([40, 48], "spin")), EXPECTED)
        self.assertTrue(s[146]["correct"] and s[44]["correct"])

    def test_extra_cause_is_wrong(self):
        s = score(self.rec(([44, 44], "spin"), ([44, 44], "late_braking")), {44: "spin"})
        self.assertFalse(s[44]["correct"])

    def test_missing_is_wrong(self):
        self.assertFalse(score(None, {44: "spin"})[44]["correct"])


class FakeServer:
    """Tiny OpenAI-style server that replays canned replies and records requests."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append((self.path, body))
                out = {"choices": [{"message": {"content": outer.replies.pop(0)},
                                    "finish_reason": "stop"}],
                       "usage": {"completion_tokens": 7},
                       "timings": {"predicted_ms": 1.0, "prompt_ms": 2.0}}
                data = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class TestOpenAICompatBackend(unittest.TestCase):
    GOOD = json.dumps({"schema_version": 1, "source_model": "m1", "track": "cota",
                       "diagnoses": [{"waypoints": [44, 44], "cause": "spin",
                                      "evidence": {}, "verified": False}],
                       "changes": {}, "curriculum": {}, "speed_plan": {}, "rationale": "x"})
    # Valid JSON for the grammar, but out of range: llama.cpp doesn't enforce number ranges.
    BAD = GOOD.replace('"changes": {}', '"changes": {"target_speed": 15.0}')

    def test_request_uses_json_schema_mode(self):
        srv = FakeServer([self.GOOD])
        try:
            r = recommend(ANALYSIS, OpenAICompatBackend(srv.url, model="qwen"))
        finally:
            srv.close()
        path, body = srv.requests[0]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(body["model"], "qwen")
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertEqual(body["response_format"]["json_schema"]["schema"], build_json_schema(ANALYSIS))
        self.assertIsNotNone(r["recommendation"])
        self.assertEqual(r["attempts"][0]["usage"]["completion_tokens"], 7)

    def test_out_of_range_reply_is_retried_with_errors(self):
        srv = FakeServer([self.BAD, self.GOOD])
        try:
            r = recommend(ANALYSIS, OpenAICompatBackend(srv.url))
        finally:
            srv.close()
        self.assertEqual(len(r["attempts"]), 2)
        self.assertIn("changes.target_speed: 15.0 out of range", r["attempts"][0]["errors"][0])
        retry_msgs = srv.requests[1][1]["messages"]
        self.assertEqual(retry_msgs[-2]["role"], "assistant")
        self.assertIn("out of range", retry_msgs[-1]["content"])
        self.assertEqual(r["recommendation"]["changes"], {})

    def test_gives_up_after_retries(self):
        srv = FakeServer([self.BAD] * 3)
        try:
            r = recommend(ANALYSIS, OpenAICompatBackend(srv.url), max_retries=2)
        finally:
            srv.close()
        self.assertIsNone(r["recommendation"])
        self.assertEqual(len(r["attempts"]), 3)


if __name__ == "__main__":
    unittest.main()
