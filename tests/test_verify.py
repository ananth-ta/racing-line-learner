"""Tests for src/post_loop/verify.py.

Run from the repo root:  python -m unittest tests.test_verify

RealAnswersTest checks the 8 saved Qwen answers against the real recording;
it is skipped when that run folder isn't there (it's on Drive, not in git).
"""
import copy
import json
import unittest
from pathlib import Path

from src.post_loop.verify import check_number, verify, waypoint_range

ANALYSIS = {
    "source_model": "m1",
    "track": "cota",
    "summary": {"laps": 3, "terminations": {"missed_waypoint": 2, "crash": 1},
                "completion_rate": 0.0, "mean_waypoints_cleared": 120.0},
    "track_info": {"num_waypoints": 150, "duplicate_waypoints": [[70, 71]]},
    "crash_segments": [{"waypoints": [44, 44], "crashes": 1, "laps_reaching": 3,
                        "crash_rate": 0.33, "mean_speed": 6.5, "blind_fraction": 0.0,
                        "max_curvature": 0.0927, "laps": [2]}],
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

#: A correct answer, shaped like the real Qwen replies.
GOOD = {
    "schema_version": 1,
    "source_model": "m1",
    "track": "cota",
    "diagnoses": [
        {"waypoints": [44, 44], "cause": "spin", "verified": False,
         "evidence": {"lap": 2.0, "heading_change_deg": 192.0, "visible_m": 1.48,
                      "required_brake_m": 2.22, "blind": 0.0, "crash_rate": 0.33}},
        {"waypoints": [146, 146], "cause": "misplaced_waypoint", "verified": False,
         "evidence": {"waypoint": 146.0, "capture_radius_m": 1.5,
                      "closest_approach_m_min": 1.61, "closest_approach_m_max": 1.75,
                      "distance_driven_after_m_min": 129.9}},
    ],
    "changes": {},
    "curriculum": {"spawn_waypoints": [], "spawn_mix": 0.0},
    "speed_plan": {},
    "rationale": "Spin at 44; waypoint 146 is misplaced.",
}


def edited(fn):
    """A copy of GOOD changed by ``fn``."""
    rec = copy.deepcopy(GOOD)
    fn(rec)
    return rec


class GoodAnswersTest(unittest.TestCase):
    def test_good_answer_passes_and_is_marked_verified(self):
        r = verify(GOOD, ANALYSIS)
        self.assertTrue(r.ok, r.errors)
        self.assertEqual(r.warnings, [])
        self.assertTrue(all(d["verified"] for d in r.recommendation["diagnoses"]))
        self.assertFalse(GOOD["diagnoses"][0]["verified"], "input must not be modified")

    def test_valid_changes_pass(self):
        rec = edited(lambda r: r.update(changes={"steering_penalty_w": 2.5},
                                        speed_plan={"40": 7.0}))
        self.assertTrue(verify(rec, ANALYSIS).ok)

    def test_rounding_is_tolerated(self):
        rec = edited(lambda r: r["diagnoses"][0]["evidence"].update(heading_change_deg=192.004,
                                                                    visible_m=1.475))
        self.assertTrue(verify(rec, ANALYSIS).ok)

    def test_mean_of_recorded_values_is_accepted(self):
        rec = edited(lambda r: r["diagnoses"][1]["evidence"].update(closest_approach_m=1.68))
        self.assertTrue(verify(rec, ANALYSIS).ok)

    def test_wide_range_covering_the_failure_passes(self):
        rec = edited(lambda r: r["diagnoses"][0].update(waypoints=[40, 48]))
        self.assertTrue(verify(rec, ANALYSIS).ok)

    def test_range_wrapping_past_the_finish_line(self):
        self.assertEqual(waypoint_range(148, 1, 150), {148, 149, 0, 1})
        rec = edited(lambda r: r["diagnoses"][1].update(waypoints=[140, 2]))
        self.assertTrue(verify(rec, ANALYSIS).ok)

    def test_unexplained_failure_is_a_warning_not_an_error(self):
        rec = edited(lambda r: r["diagnoses"].pop(0))          # drop the wp 44 spin
        r = verify(rec, ANALYSIS)
        self.assertTrue(r.ok, r.errors)
        self.assertEqual(r.warnings, ["no diagnosis covers the crash at waypoint 44 (lap 2)"])


class WrongAnswersTest(unittest.TestCase):
    """Deliberately wrong answers: each must be rejected, for the right reason."""

    def assertRejected(self, rec, *fragments):
        r = verify(rec, ANALYSIS)
        self.assertFalse(r.ok, "should have been rejected")
        self.assertIsNone(r.recommendation)
        text = "\n".join(r.errors)
        for frag in fragments:
            self.assertIn(frag, text)
        return r

    def test_made_up_number(self):
        rec = edited(lambda r: r["diagnoses"][0]["evidence"].update(heading_change_deg=150.0))
        self.assertRejected(rec, "diagnoses[0].evidence.heading_change_deg",
                            "150.0 does not match the recorded heading_change_deg (192)")

    def test_made_up_min_max(self):
        rec = edited(lambda r: r["diagnoses"][1]["evidence"].update(closest_approach_m_max=1.9))
        self.assertRejected(rec, "closest_approach_m_max", "max of the recorded closest_approach_m")

    def test_number_from_another_failure(self):
        # 192.0 is real, but it belongs to the wp 44 crash, not to wp 146.
        rec = edited(lambda r: r["diagnoses"][1]["evidence"].update(heading_change_deg=192.0))
        self.assertRejected(rec, "diagnoses[1].evidence.heading_change_deg", "field not found")

    def test_field_not_in_the_data(self):
        rec = edited(lambda r: r["diagnoses"][0]["evidence"].update(tyre_grip=0.8))
        self.assertRejected(rec, "evidence.tyre_grip", "field not found")

    def test_waypoint_that_does_not_exist(self):
        rec = edited(lambda r: r["diagnoses"][1].update(waypoints=[150, 150]))
        self.assertRejected(rec, "waypoint index 150 out of range [0, 149]")

    def test_spawn_waypoint_that_does_not_exist(self):
        rec = edited(lambda r: r.update(curriculum={"spawn_waypoints": [200], "spawn_mix": 0.2}))
        self.assertRejected(rec, "curriculum.spawn_waypoints[0]", "out of range")

    def test_waypoint_where_nothing_failed(self):
        rec = edited(lambda r: r["diagnoses"].append(
            {"waypoints": [100, 100], "cause": "late_braking", "evidence": {}, "verified": False}))
        self.assertRejected(rec, "diagnoses[2]: nothing failed at waypoints [100, 100]")

    def test_setting_outside_ananths_range(self):
        rec = edited(lambda r: r.update(changes={"steering_penalty_w": 5.0}))
        self.assertRejected(rec, "changes.steering_penalty_w: 5.0 out of range [0.5, 3.0]")

    def test_speed_floor_below_ananths_range(self):
        rec = edited(lambda r: r.update(changes={"speed_floor": 5.0}))
        self.assertRejected(rec, "changes.speed_floor: 5.0 out of range [6.5, 8.0]")

    def test_speed_plan_below_speed_floor(self):
        rec = edited(lambda r: r.update(speed_plan={"44": 5.0}))
        self.assertRejected(rec, "speed_plan.44: 5.0 out of range [6.5, 20.0]")

    def test_setting_not_on_ananths_list(self):
        rec = edited(lambda r: r.update(changes={"learning_rate": 0.001}))
        self.assertRejected(rec, "changes.learning_rate: not a tunable parameter")

    def test_more_than_two_changes(self):
        rec = edited(lambda r: r.update(changes={"steering_penalty_w": 2.5, "speed_floor": 7.0,
                                                 "ent_coef": 0.02}))
        self.assertRejected(rec, "3 changes proposed, at most 2 allowed")

    def test_two_changes_plus_curriculum_is_too_many(self):
        rec = edited(lambda r: r.update(changes={"steering_penalty_w": 2.5, "speed_floor": 7.0},
                                        curriculum={"spawn_waypoints": [40], "spawn_mix": 0.2}))
        self.assertRejected(rec, "3 changes proposed, at most 2 allowed")

    def test_wrong_track_or_model(self):
        self.assertRejected(edited(lambda r: r.update(track="vegas")), "track:")
        self.assertRejected(edited(lambda r: r.update(source_model="other")), "source_model:")


class CheckNumberTest(unittest.TestCase):
    def test_bool_counts_as_0_or_1(self):
        pool = {"blind": [0.0]}
        self.assertIsNone(check_number("blind", 0.0, pool))
        self.assertIsNotNone(check_number("blind", 1.0, pool))


RUN = Path("post_loop_runs/ppo_final_3_5000000_steps__cota__20261112-001137")
REAL_FILES = ["llm_t0.json", "llm_t07.json"]


@unittest.skipUnless((RUN / "analysis.json").exists(), "real recording not available")
class RealAnswersTest(unittest.TestCase):
    """The 8 saved Qwen2.5-7B answers (3 at temp 0, 5 at temp 0.7) on the pre-fix run."""

    def test_all_real_answers_pass(self):
        analysis = json.loads((RUN / "analysis.json").read_text())
        n = 0
        for name in REAL_FILES:
            for i, trial in enumerate(json.loads((RUN / name).read_text())["trials"]):
                with self.subTest(file=name, trial=i + 1):
                    r = verify(trial["recommendation"], analysis)
                    self.assertTrue(r.ok, r.errors)
                    self.assertEqual(r.warnings, [])
                n += 1
        self.assertEqual(n, 8)


if __name__ == "__main__":
    unittest.main()
