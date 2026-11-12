import unittest

from racing_line_learner.in_loop.validator import (
    CAPPED_KEYS,
    CURRICULUM_KEY,
    OUTCOME_IMPROVED,
    OUTCOME_NEUTRAL,
    OUTCOME_PENDING,
    OUTCOME_ROLLED_BACK,
    KeyHistory,
    ValidatorConfig,
    default_values,
    max_step,
    touched_keys,
    validate_proposal,
)
from racing_line_learner.schema import WHITELIST, parse_recommendation

N_WP = 150


def rec(changes=None, curriculum=None, speed_plan=None):
    data = {"schema_version": 1, "source_model": "m", "track": "cota", "changes": changes or {}}
    if curriculum is not None:
        data["curriculum"] = curriculum
    if speed_plan is not None:
        data["speed_plan"] = speed_plan
    return parse_recommendation(data, N_WP)


CURRENT = default_values()


class TestStepCap(unittest.TestCase):
    def test_cap_is_quarter_of_range(self):
        self.assertAlmostEqual(max_step("target_speed", ValidatorConfig()), 1.5)    # range 6..12
        self.assertAlmostEqual(max_step("speed_floor", ValidatorConfig()), 0.375)   # range 6.5..8
        self.assertAlmostEqual(max_step("waypoint_bonus", ValidatorConfig()), 20.0)  # range 20..100

    def test_within_cap_accepted(self):
        res = validate_proposal(rec({"target_speed": 8.5}), CURRENT, window=3)  # 10 -> 8.5 = exactly 1.5
        self.assertTrue(res.accepted, res.errors)

    def test_over_cap_rejected(self):
        res = validate_proposal(rec({"target_speed": 7.0}), CURRENT, window=3)  # 3.0 > 1.5
        self.assertFalse(res.accepted)
        self.assertIn("exceeds per-window cap", res.errors[0])

    def test_every_capped_key_enforced_both_directions(self):
        for key in CAPPED_KEYS:
            spec = WHITELIST[key]
            limit = max_step(key, ValidatorConfig())
            for target in (spec.low, spec.high):
                if abs(target - spec.default) <= limit:
                    continue
                with self.subTest(key=key, target=target):
                    self.assertFalse(validate_proposal(rec({key: target}), CURRENT, window=1).accepted)

    def test_ent_coef_and_curriculum_uncapped(self):
        self.assertIsNone(max_step("ent_coef", ValidatorConfig()))
        res = validate_proposal(rec({"ent_coef": 0.05}), CURRENT, window=1)  # full-range jump
        self.assertTrue(res.accepted, res.errors)
        res = validate_proposal(rec(curriculum={"spawn_waypoints": [104], "spawn_mix": 0.5}), CURRENT, window=1)
        self.assertTrue(res.accepted, res.errors)

    def test_cap_disabled_for_ablation(self):
        res = validate_proposal(rec({"target_speed": 6.0}), CURRENT, window=1,
                                config=ValidatorConfig(max_step_fraction=None))
        self.assertTrue(res.accepted, res.errors)

    def test_cap_measured_from_current_not_default(self):
        current = dict(CURRENT, target_speed=8.5)
        self.assertTrue(validate_proposal(rec({"target_speed": 7.0}), current, window=5).accepted)

    def test_missing_current_value(self):
        res = validate_proposal(rec({"target_speed": 9.0}), {}, window=1)
        self.assertIn("no current value", res.errors[0])


class TestCooldown(unittest.TestCase):
    def check(self, outcome, window, judged=None, key="target_speed", changes=None, config=ValidatorConfig()):
        hist = {key: KeyHistory(window=3, outcome=outcome, judged_window=judged)}
        return validate_proposal(rec(changes or {"target_speed": 9.0}), CURRENT, window, hist, config)

    def test_pending_change_blocks_next_window(self):
        res = self.check(OUTCOME_PENDING, window=4)
        self.assertFalse(res.accepted)
        self.assertIn("awaiting judgement", res.errors[0])

    def test_improved_change_may_be_nudged_again_at_once(self):
        self.assertTrue(self.check(OUTCOME_IMPROVED, window=4, judged=4).accepted)

    def test_neutral_change_waits_one_window(self):
        self.assertFalse(self.check(OUTCOME_NEUTRAL, window=4, judged=4).accepted)
        self.assertTrue(self.check(OUTCOME_NEUTRAL, window=5, judged=4).accepted)

    def test_rolled_back_extended_lock(self):
        # Changed in w3, rolled back in w4, locked for 2 more windows: free again in w7.
        for w in (5, 6):
            with self.subTest(window=w):
                self.assertFalse(self.check(OUTCOME_ROLLED_BACK, window=w, judged=4).accepted)
        self.assertTrue(self.check(OUTCOME_ROLLED_BACK, window=7, judged=4).accepted)

    def test_never_judged_change_stays_blocked(self):
        res = self.check(OUTCOME_PENDING, window=9)
        self.assertFalse(res.accepted)
        self.assertIn("never judged", res.errors[0])

    def test_other_keys_unaffected(self):
        res = self.check(OUTCOME_PENDING, window=4, changes={"jitter_coef": 0.06})
        self.assertTrue(res.accepted, res.errors)

    def test_curriculum_pseudo_key(self):
        hist = {CURRICULUM_KEY: KeyHistory(window=3)}
        r = rec(curriculum={"spawn_waypoints": [10], "spawn_mix": 0.2})
        self.assertEqual(touched_keys(r), [CURRICULUM_KEY])
        self.assertFalse(validate_proposal(r, CURRENT, 4, hist).accepted)

    def test_cooldown_disabled_for_ablation(self):
        cfg = ValidatorConfig(cooldown_windows=0, rollback_cooldown_windows=0)
        self.assertTrue(self.check(OUTCOME_PENDING, window=4, config=cfg).accepted)
        self.assertTrue(self.check(OUTCOME_ROLLED_BACK, window=4, judged=4, config=cfg).accepted)

    def test_history_from_future_rejected(self):
        self.assertFalse(self.check(OUTCOME_IMPROVED, window=3).accepted)

    def test_reports_all_errors(self):
        hist = {"target_speed": KeyHistory(window=3)}
        res = validate_proposal(rec({"target_speed": 6.0}), CURRENT, 4, hist)  # over cap AND on cooldown
        self.assertEqual(len(res.errors), 2)


class TestConfig(unittest.TestCase):
    def test_bad_config(self):
        with self.assertRaises(ValueError):
            ValidatorConfig(max_step_fraction=0.0)
        with self.assertRaises(ValueError):
            ValidatorConfig(cooldown_windows=-1)
        with self.assertRaises(ValueError):
            KeyHistory(window=1, outcome="great")


if __name__ == "__main__":
    unittest.main()
