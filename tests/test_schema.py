import copy
import json
import math
import os
import tempfile
import unittest

from racing_line_learner.schema import (
    SCHEMA_VERSION,
    WHITELIST,
    Recommendation,
    SchemaError,
    dump_recommendation,
    load_recommendation,
    parse_recommendation,
    recommendation_json_schema,
)

N_WP = 150  # COTA waypoint count; tests also vary this to check bounds come from the track.

EXAMPLE = {
    "schema_version": 1,
    "source_model": "ppo_cota_run3_5M",
    "track": "cota",
    "diagnoses": [
        {
            "waypoints": [108, 114],
            "cause": "blind_corner",
            "evidence": {"crash_rate": 0.8, "mean_speed": 9.4, "visible_m": 6.1, "required_brake_m": 8.7},
            "verified": True,
        }
    ],
    "changes": {"target_speed": 8.5},
    "curriculum": {"spawn_waypoints": [104], "spawn_mix": 0.3},
    "rationale": "short text, logged only",
}


def doc(**overrides):
    d = copy.deepcopy(EXAMPLE)
    for k, v in overrides.items():
        if v is None:
            d.pop(k, None)
        else:
            d[k] = v
    return d


class SchemaTestCase(unittest.TestCase):
    def assertRejected(self, data, *fragments, **kwargs):
        kwargs.setdefault("n_waypoints", N_WP)
        with self.assertRaises(SchemaError) as ctx:
            parse_recommendation(data, **kwargs)
        joined = "\n".join(ctx.exception.errors)
        for frag in fragments:
            self.assertIn(frag, joined)
        return ctx.exception


class TestValidDocuments(SchemaTestCase):
    def test_example_parses(self):
        rec = parse_recommendation(EXAMPLE, N_WP)
        self.assertEqual(rec.track, "cota")
        self.assertEqual(rec.changes, {"target_speed": 8.5})
        self.assertEqual(rec.curriculum.spawn_waypoints, (104,))
        self.assertAlmostEqual(rec.curriculum.spawn_mix, 0.3)
        self.assertEqual(rec.diagnoses[0].waypoints, (108, 114))
        self.assertTrue(rec.diagnoses[0].verified)
        self.assertEqual(rec.n_changes, 2)

    def test_minimal_document_is_noop(self):
        rec = parse_recommendation({"schema_version": 1, "source_model": "m", "track": "cota"}, N_WP)
        self.assertTrue(rec.is_noop())
        self.assertEqual(rec.diagnoses, ())

    def test_speed_plan_string_keys_become_ints(self):
        rec = parse_recommendation(doc(changes={}, curriculum=None, speed_plan={"104": 8.0, "108": 6.8}), N_WP)
        self.assertEqual(rec.speed_plan, {104: 8.0, 108: 6.8})

    def test_ints_accepted_as_numbers(self):
        rec = parse_recommendation(doc(changes={"waypoint_bonus": 60}, curriculum=None), N_WP)
        self.assertEqual(rec.changes["waypoint_bonus"], 60.0)
        self.assertIsInstance(rec.changes["waypoint_bonus"], float)

    def test_range_endpoints_inclusive(self):
        for key, spec in WHITELIST.items():
            for value in (spec.low, spec.high):
                with self.subTest(key=key, value=value):
                    parse_recommendation(doc(changes={key: value}, curriculum=None), N_WP)

    def test_whitelist_defaults_within_range(self):
        for key, spec in WHITELIST.items():
            with self.subTest(key=key):
                self.assertTrue(spec.low <= spec.default <= spec.high)

    def test_wrapping_diagnosis_range_allowed(self):
        rec = parse_recommendation(doc(diagnoses=[{"waypoints": [145, 3], "cause": "start_finish"}]), N_WP)
        self.assertEqual(rec.diagnoses[0].waypoints, (145, 3))
        self.assertFalse(rec.diagnoses[0].verified)

    def test_roundtrip_through_file(self):
        rec = parse_recommendation(doc(changes={}, speed_plan={"104": 8.0}), N_WP)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "recommendations.json")
            dump_recommendation(rec, path)
            again = load_recommendation(path, N_WP)
        self.assertEqual(rec, again)


class TestRejections(SchemaTestCase):
    def test_unknown_top_level_key(self):
        self.assertRejected(doc(model_weights=[1, 2]), "model_weights: unknown key")

    def test_missing_required(self):
        self.assertRejected(doc(track=None), "track: required key missing")

    def test_bad_version(self):
        self.assertRejected(doc(schema_version=2), "schema_version")
        self.assertRejected(doc(schema_version="1"), "schema_version")

    def test_unknown_parameter(self):
        self.assertRejected(doc(changes={"learning_rate": 0.001}, curriculum=None), "changes.learning_rate")

    def test_out_of_range_parameter(self):
        self.assertRejected(doc(changes={"speed_floor": 6.0}, curriculum=None), "out of range")
        self.assertRejected(doc(changes={"ent_coef": 0.5}, curriculum=None), "out of range")

    def test_non_numeric_bool_nan(self):
        self.assertRejected(doc(changes={"target_speed": "fast"}, curriculum=None), "expected a number")
        self.assertRejected(doc(changes={"target_speed": True}, curriculum=None), "expected a number")
        self.assertRejected(doc(changes={"target_speed": math.nan}, curriculum=None), "finite")
        self.assertRejected(doc(changes={"target_speed": math.inf}, curriculum=None), "finite")

    def test_change_cap(self):
        self.assertRejected(
            doc(changes={"target_speed": 8.0, "jitter_coef": 0.1}),  # 2 changes + curriculum = 3
            "3 changes proposed",
        )
        parse_recommendation(doc(changes={"target_speed": 8.0, "jitter_coef": 0.1}), N_WP, max_changes=3)

    def test_waypoint_bounds_come_from_track(self):
        self.assertRejected(doc(curriculum={"spawn_waypoints": [150], "spawn_mix": 0.2}), "out of range [0, 149]")
        # Same index is fine on a longer track, and rejected on a shorter one.
        parse_recommendation(doc(curriculum={"spawn_waypoints": [150], "spawn_mix": 0.2}), 200)
        self.assertRejected(doc(), "out of range [0, 99]", n_waypoints=100)  # diagnosis [108, 114]

    def test_curriculum_consistency(self):
        self.assertRejected(doc(curriculum={"spawn_waypoints": [10], "spawn_mix": 0.0}), "never fire")
        self.assertRejected(doc(curriculum={"spawn_waypoints": [], "spawn_mix": 0.2}), "requires at least one")
        self.assertRejected(doc(curriculum={"spawn_waypoints": [10, 10], "spawn_mix": 0.2}), "duplicate")
        self.assertRejected(doc(curriculum={"spawn_waypoints": [10], "spawn_mix": 0.6}), "spawn_mix")
        self.assertRejected(doc(curriculum={"spawn_waypoint": [10], "spawn_mix": 0.2}), "unknown key")

    def test_speed_plan_bounds(self):
        self.assertRejected(doc(changes={}, speed_plan={"104": 6.0}), "speed_plan.104")
        self.assertRejected(doc(changes={}, speed_plan={"104": 25.0}), "speed_plan.104")
        self.assertRejected(doc(changes={}, speed_plan={"abc": 8.0}), "key must be a waypoint index")
        self.assertRejected(doc(changes={}, speed_plan={"-1": 8.0}), "out of range")

    def test_speed_plan_respects_new_and_current_floor(self):
        # A recommendation raising the floor to 7.5 cannot plan 7.0 anywhere.
        self.assertRejected(
            doc(changes={"speed_floor": 7.5}, curriculum=None, speed_plan={"104": 7.0}), "[7.5, 20.0]"
        )
        # The floor currently in force also applies.
        self.assertRejected(doc(changes={}, curriculum=None, speed_plan={"104": 7.0}), "[7.2", speed_floor=7.2)

    def test_bad_diagnosis(self):
        self.assertRejected(doc(diagnoses=[{"waypoints": [1], "cause": "x"}]), "expected [start, end]")
        self.assertRejected(doc(diagnoses=[{"waypoints": [1, 2], "cause": ""}]), "cause")
        self.assertRejected(doc(diagnoses=[{"waypoints": [1, 2], "cause": "x", "verified": "yes"}]), "verified")
        self.assertRejected(doc(diagnoses=[{"waypoints": [1, 2], "cause": "x", "evidence": {"a": "b"}}]), "evidence.a")
        self.assertRejected(doc(diagnoses=[{"waypoints": [1, 2], "cause": "x", "fix": 1}]), "fix: unknown key")

    def test_expected_track(self):
        self.assertRejected(doc(), "expected 'heldout'", expected_track="heldout")

    def test_rationale(self):
        self.assertRejected(doc(rationale=5), "rationale")
        self.assertRejected(doc(rationale="x" * 5000), "exceeds limit")

    def test_reports_all_errors_at_once(self):
        err = self.assertRejected(
            doc(changes={"target_speed": 99, "bogus": 1}, extra=1, rationale=3),
        )
        self.assertGreaterEqual(len(err.errors), 4)

    def test_non_object_root_and_bad_json(self):
        self.assertRejected([1, 2], "<root>")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.json")
            with open(path, "w") as f:
                f.write("{not json")
            with self.assertRaises(SchemaError):
                load_recommendation(path, N_WP)

    def test_invalid_n_waypoints(self):
        with self.assertRaises(ValueError):
            parse_recommendation(EXAMPLE, 0)


class TestJsonSchema(unittest.TestCase):
    def test_schema_is_serializable_and_matches_whitelist(self):
        schema = recommendation_json_schema(N_WP)
        json.dumps(schema)
        props = schema["properties"]["changes"]["properties"]
        self.assertEqual(set(props), set(WHITELIST))
        self.assertEqual(props["target_speed"]["maximum"], WHITELIST["target_speed"].high)
        spawn = schema["properties"]["curriculum"]["properties"]["spawn_waypoints"]["items"]
        self.assertEqual(spawn["maximum"], N_WP - 1)
        self.assertEqual(schema["properties"]["schema_version"]["enum"], [SCHEMA_VERSION])


if __name__ == "__main__":
    unittest.main()
