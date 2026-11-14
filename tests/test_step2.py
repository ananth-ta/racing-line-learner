"""Step 2: corner segments, smoothness telemetry and detectors, new reward terms, schema v2 and grounding."""
import json
import unittest
from dataclasses import replace

import numpy as np

from racing_line_learner.config import EnvConfig, get_preset
from racing_line_learner.env.rewards import laptime_reward
from racing_line_learner.env.tracks import load_track
from racing_line_learner.in_loop import detectors
from racing_line_learner.in_loop.detectors import LEVERS, allowed_levers, detect
from racing_line_learner.in_loop.telemetry import TelemetryCollector
from racing_line_learner.in_loop.validator import ValidatorConfig, default_values, validate_proposal
from racing_line_learner.schema import (WHITELIST, WHITELIST_V2, SchemaError, parse_recommendation,
                                        recommendation_json_schema)
from tests.sim import requires_sim

COTA = load_track("cota")
V2 = get_preset("v2")


class TestCornerSegments(unittest.TestCase):
    def test_cover_every_waypoint_once_in_lap_order(self):
        for name in ("cota", "sepang", "sepang_wide"):
            t = load_track(name)
            segs = t.corner_segments()
            idx = [i for s in segs for i in s["waypoints"]]
            self.assertEqual(sorted(idx), list(range(t.n_waypoints)), name)
            self.assertIn(0, segs[0]["waypoints"], name)                       # first segment holds the line
            kinds = [s["kind"] for s in segs]
            self.assertTrue(all(a != b for a, b in zip(kinds, kinds[1:])), name)  # corners and straights alternate
            self.assertTrue(all(len(s["waypoints"]) >= 3 for s in segs if s["kind"] == "straight"), name)

    def test_known_cota_corners(self):
        names = {s["name"]: s["waypoints"] for s in COTA.corner_segments()}
        corners = [w for n, w in names.items() if n.startswith("C")]
        for apex in (18, 70, 111, 146):                                          # the hairpins we know
            self.assertTrue(any(apex in w for w in corners), apex)
        self.assertTrue(any(80 in w and len(w) > 20 for n, w in names.items() if n.startswith("S")))  # back straight

    def test_corner_practice_spawns_before_tight_corners(self):
        grip = COTA.grip_speed_limit()
        spawns = COTA.corner_approach_waypoints(5)
        self.assertEqual(len(set(spawns)), 5)
        for w in spawns:   # a tight corner within ~6 waypoints ahead of each spawn point
            ahead = [(w + o) % COTA.n_waypoints for o in range(1, 7)]
            self.assertLess(grip[ahead].min(), 6.0)

    def test_sectors_validation(self):
        from racing_line_learner.env.tracks import Track
        with self.assertRaises(ValueError):
            Track("x", COTA.map_path, ".png", COTA.waypoints, sectors=(100, 50))
        t = Track("x", COTA.map_path, ".png", COTA.waypoints, sectors=(50, 100))
        self.assertEqual(t.sectors, (50, 100))


class TestNewRewardTerms(unittest.TestCase):
    def r(self, cfg, **kw):
        args = dict(velocity=10.0, raw_steer=0.0, prev_raw_steer=0.0, progress_m=0.1, waypoints_cleared=0,
                    collision=False, stagnated=False, lap_complete=False)
        args.update(kw)
        return laptime_reward(cfg, **args)

    def test_off_by_default(self):
        base, _ = self.r(V2)
        self.assertEqual(base, self.r(V2, throttle_change=3.0, overspeed=2.0)[0])

    def test_throttle_and_overspeed(self):
        cfg = replace(V2, throttle_change_coef=0.2, overspeed_coef=0.5)
        base, _ = self.r(cfg)
        r, terms = self.r(cfg, throttle_change=3.0, overspeed=2.0)
        self.assertAlmostEqual(terms["throttle"], -0.6)
        self.assertAlmostEqual(terms["overspeed"], -2.0)
        self.assertAlmostEqual(r, base - 2.6)
        self.assertEqual(self.r(cfg, throttle_change=3.0, collision=True)[1]["throttle"], 0.0)  # terminal replaces

    def test_config_validation(self):
        for bad in ({"throttle_change_coef": -1}, {"grip_margin": 0.1}, {"spawn_corner_k": 0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(V2, **bad)
        with self.assertRaises(ValueError):
            replace(EnvConfig(), overspeed_coef=0.5)   # legacy mode has no such term


def info(wp, steer=0.0, cmd=10.0, lat=0.0, kappa=0.0, term=None, speed=10.0):
    return {"current_waypoint_idx": wp, "speed": speed, "speed_command": cmd, "raw_steering": steer, "steering": steer,
            "lateral": lat, "curvature": kappa, "termination": term, "reward_terms": {"progress": 1.0}}


class TestSmoothnessTelemetry(unittest.TestCase):
    def straight_run(self, steer_fn, cmd_fn, lat_fn, n=100):
        c = TelemetryCollector(COTA, V2)
        for i in range(n):
            c.update(info(85, steer_fn(i), cmd_fn(i), lat_fn(i)), 1.0)          # waypoint 85: back straight
        return c.report()

    def test_weaving_and_jitter_measured(self):
        r = self.straight_run(lambda i: 0.1 * (-1) ** i, lambda i: 12 + 1.0 * (-1) ** i, lambda i: 0.4 * np.sin(i / 3))
        sm = r["smoothness"]
        self.assertAlmostEqual(sm["straight_share"], 1.0)
        self.assertGreater(sm["steer_reversals_per_s"], 20)                       # flips every 0.04 s step
        self.assertAlmostEqual(sm["throttle_change_per_step"], 2.0, places=1)
        self.assertGreater(sm["lateral_std_m"], 0.25)
        ids = [f["id"] for f in detect(r)]
        self.assertIn("weaving", ids)
        self.assertIn("throttle_jitter", ids)

    def test_smooth_driving_not_flagged(self):
        r = self.straight_run(lambda i: 0.01 * np.sin(i / 40), lambda i: 12 + i * 0.05, lambda i: 0.05 * np.sin(i / 50))
        ids = [f["id"] for f in detect(r)]
        self.assertNotIn("weaving", ids)
        self.assertNotIn("throttle_jitter", ids)

    def test_corners_do_not_count_as_straights(self):
        c = TelemetryCollector(COTA, V2)
        for i in range(50):
            c.update(info(111, 0.3 * (-1) ** i, kappa=0.3), 1.0)
        self.assertEqual(c.report()["smoothness"]["steer_reversals_per_s"], None)

    def test_probe_smoothness_preferred(self):
        r = self.straight_run(lambda i: 0.1 * (-1) ** i, lambda i: 12.0, lambda i: 0.4 * np.sin(i / 3))
        r["probe"] = {"smoothness": {"steer_reversals_per_s": 1.0, "lateral_std_m": 0.05, "throttle_change_per_step": 0.0},
                      "segments": []}
        self.assertNotIn("weaving", [f["id"] for f in detect(r)])                # deterministic driving is smooth

    def test_inconsistent_line_in_a_corner(self):
        c = TelemetryCollector(COTA, V2)
        for i in range(60):
            c.update(info(111, lat=1.0 * (-1) ** i, kappa=0.3), 1.0)
        f = [x for x in detect(c.report()) if x["id"] == "inconsistent_line"]
        self.assertTrue(f)
        lo, hi = f[0]["waypoints"]
        self.assertTrue(lo <= 110 <= hi, f[0]["waypoints"])      # the car is between waypoints 110 and 111

    def test_sector_proxy_and_names(self):
        r = self.straight_run(lambda i: 0.0, lambda i: 12.0, lambda i: 0.0, n=25)
        seg = next(s for s in r["segments"] if s["waypoints"][0] <= 84 <= s["waypoints"][1])
        self.assertTrue(seg["name"].startswith("S"))
        self.assertAlmostEqual(seg["pass_time_s"], 1.0)                           # 25 steps x 0.04 s, one pass


class TestLevers(unittest.TestCase):
    def test_every_lever_exists_in_a_whitelist(self):
        known = set(WHITELIST) | set(WHITELIST_V2) | {"curriculum"}
        for fid, keys in LEVERS.items():
            self.assertTrue(set(keys) <= known, (fid, set(keys) - known))

    def test_allowed_levers(self):
        out = allowed_levers([{"id": "weaving"}, {"id": "throttle_jitter"}])
        self.assertEqual(out["jitter_coef"], ["weaving"])
        self.assertEqual(out["throttle_change_coef"], ["throttle_jitter"])


def v2_doc(changes, grounds=None, **kw):
    d = {"schema_version": 2, "source_model": "m", "track": "cota", "changes": changes,
         "grounds": grounds if grounds is not None else {k: ["weaving"] for k in changes}}
    d.update(kw)
    return d


class TestSchemaV2(unittest.TestCase):
    def test_defaults_match_v2_preset(self):
        for key, spec in WHITELIST_V2.items():
            if hasattr(V2, key):
                self.assertEqual(getattr(V2, key), spec.default, key)
            self.assertTrue(spec.low <= spec.default <= spec.high, key)

    def test_parse_and_roundtrip(self):
        rec = parse_recommendation(v2_doc({"jitter_coef": 0.1}), COTA.n_waypoints)
        self.assertEqual((rec.schema_version, rec.grounds), (2, {"jitter_coef": ("weaving",)}))
        again = parse_recommendation(json.loads(json.dumps(rec.to_dict())), COTA.n_waypoints)
        self.assertEqual(rec, again)
        self.assertNotIn("curriculum", rec.to_dict())

    def test_rejections(self):
        cases = [
            (v2_doc({"speed_floor": 7.0}), "not a tunable parameter"),             # v1-only key
            (v2_doc({"jitter_coef": 0.9}), "out of range"),
            (v2_doc({"jitter_coef": 0.1}, grounds={}), "must cite"),
            (v2_doc({"jitter_coef": 0.1}, grounds={"jitter_coef": []}), "non-empty list"),
            (v2_doc({"jitter_coef": 0.1}, grounds={"jitter_coef": ["weaving"], "time_cost": ["x"]}), "not one of the proposed"),
            (v2_doc({}, curriculum={"spawn_waypoints": [3], "spawn_mix": 0.1}), "curriculum: unknown key"),
        ]
        for doc, frag in cases:
            with self.subTest(frag=frag), self.assertRaises(SchemaError) as ctx:
                parse_recommendation(doc, COTA.n_waypoints)
            self.assertIn(frag, "\n".join(ctx.exception.errors))

    def test_v1_unchanged(self):
        rec = parse_recommendation({"schema_version": 1, "source_model": "m", "track": "cota",
                                    "changes": {"target_speed": 9.0}}, COTA.n_waypoints)
        self.assertEqual((rec.schema_version, rec.grounds), (1, {}))
        self.assertIn("curriculum", rec.to_dict())
        self.assertNotIn("grounds", rec.to_dict())

    def test_json_schema_versions(self):
        s1, s2 = recommendation_json_schema(150), recommendation_json_schema(150, version=2)
        self.assertIn("curriculum", s1["properties"])
        self.assertIn("grounds", s2["properties"])
        self.assertIn("grounds", s2["required"])
        self.assertEqual(set(s2["properties"]["changes"]["properties"]), set(WHITELIST_V2))
        json.dumps(s2)


class TestGrounding(unittest.TestCase):
    CURRENT = default_values(2)
    FINDINGS = [{"id": "weaving"}, {"id": "throttle_jitter"}]

    def check(self, changes, grounds, findings=FINDINGS, config=ValidatorConfig()):
        rec = parse_recommendation(v2_doc(changes, grounds), COTA.n_waypoints)
        return validate_proposal(rec, self.CURRENT, window=3, findings=findings, config=config)

    def test_grounded_change_accepted(self):
        res = self.check({"jitter_coef": 0.1}, {"jitter_coef": ["weaving"]})
        self.assertTrue(res.accepted, res.errors)

    def test_hallucinated_finding_rejected(self):
        res = self.check({"jitter_coef": 0.1}, {"jitter_coef": ["corkscrew"]})
        self.assertFalse(res.accepted)
        self.assertIn("not in the current report", res.errors[0])

    def test_wrong_lever_for_finding_rejected(self):
        res = self.check({"time_cost": 0.12}, {"time_cost": ["weaving"]})
        self.assertFalse(res.accepted)
        self.assertIn("no cited finding is addressed by this lever", res.errors[0])

    def test_missing_findings_rejected_and_ablation(self):
        self.assertFalse(self.check({"jitter_coef": 0.1}, {"jitter_coef": ["weaving"]}, findings=None).accepted)
        off = ValidatorConfig(require_grounding=False)
        self.assertTrue(self.check({"time_cost": 0.12}, {"time_cost": ["weaving"]}, config=off).accepted)

    def test_v2_step_cap_uses_v2_ranges(self):
        res = self.check({"jitter_coef": 0.2}, {"jitter_coef": ["weaving"]})      # cap = 0.25 x 0.49 = 0.1225
        self.assertFalse(res.accepted)
        self.assertIn("exceeds per-window cap", res.errors[0])


@requires_sim
class TestWrapperStep2(unittest.TestCase):
    def make(self, **kw):
        from racing_line_learner.env.wrapper import make_env
        env = make_env("cota", replace(V2, spawn_random=0.0, **kw), seed=4)
        self.addCleanup(env.close)
        return env

    def test_info_has_line_position(self):
        env = self.make()
        env.reset(spawn_idx=80)
        _, _, _, inf = env.step(np.array([0.0, 0.0], dtype=np.float32))
        self.assertLess(abs(inf["lateral"]), 0.5)                                 # spawned on the line
        self.assertLess(abs(inf["curvature"]), 0.05)                              # back straight

    def test_throttle_term_charged_once_per_policy_step(self):
        env = self.make(throttle_change_coef=0.2)
        env.reset(spawn_idx=80)
        env.step(np.array([0.0, 0.0], dtype=np.float32))                          # 5 m/s
        _, _, _, inf = env.step(np.array([0.0, 1.0], dtype=np.float32))           # 20 m/s: +15 m/s
        self.assertAlmostEqual(inf["reward_terms"]["throttle"], -0.2 * 15.0, places=5)

    def test_overspeed_term_in_a_tight_corner(self):
        env = self.make(overspeed_coef=1.0, grip_margin=0.8)
        obs = env.reset(spawn_idx=COTA.corner_approach_waypoints(1)[0])
        total = 0.0
        for _ in range(40):
            obs, _, done, inf = env.step(np.array([np.clip(obs[env.obs_dim * 2 + 22] * np.pi / 0.4189, -1, 1), 1.0],
                                                  dtype=np.float32))
            total += inf["reward_terms"]["overspeed"]
            if done:
                break
        self.assertLess(total, 0.0)                                               # full throttle into it is penalised

    def test_corner_practice_spawns(self):
        env = self.make(spawn_corner_mix=1.0)
        spawns = {(env.reset(), env.spawn_idx)[1] for _ in range(30)}
        self.assertTrue(spawns <= set(COTA.corner_approach_waypoints(5)))
        self.assertGreater(len(spawns), 2)


if __name__ == "__main__":
    unittest.main()
