import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from racing_line_learner.train.ppo import (
    RunLinearSchedule,
    build_run_spec,
    find_latest_checkpoint,
    main,
    parse_args,
)
from tests.sim import requires_sb3


class TestHelpers(unittest.TestCase):
    def test_schedule_matches_original_for_fresh_run(self):
        s = RunLinearSchedule(3e-4, 0, 1000)
        for p in (1.0, 0.5, 0.1, 0.0):
            self.assertAlmostEqual(s(p), 3e-4 * p)

    def test_schedule_restarts_for_finetune(self):
        s = RunLinearSchedule(3e-4, start_timesteps=5000, run_timesteps=1000)
        self.assertAlmostEqual(s(1 - 5000 / 6000), 3e-4)   # first step of the fine-tune
        self.assertAlmostEqual(s(1 - 5500 / 6000), 1.5e-4)
        self.assertAlmostEqual(s(0.0), 0.0)

    def test_find_latest_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            for name in ("ppo_cota_a_100_steps.zip", "ppo_cota_a_2000_steps.zip", "ppo_cota_ab_9000_steps.zip",
                         "other.zip"):
                (d / name).write_bytes(b"")
            steps, path = find_latest_checkpoint(d, "ppo_cota_a")
            self.assertEqual(steps, 2000)
            self.assertEqual(find_latest_checkpoint(d / "missing", "x"), None)

    def test_args(self):
        a = parse_args(["--run_name", "r1"])
        self.assertEqual((a.track, a.preset, a.timesteps, a.resume, a.device), ("cota", "legacy", 5_000_000, "auto", "cpu"))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--run_name", "../escape"])

    def test_run_spec_applies_recommendation(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = Path(tmp) / "rec.json"
            rec.write_text(json.dumps({"schema_version": 1, "source_model": "m", "track": "cota",
                                       "changes": {"target_speed": 9.0, "ent_coef": 0.02}}))
            spec = build_run_spec(parse_args(["--run_name", "r", "--config", str(rec)]))
        self.assertEqual(spec["env_config"]["target_speed"], 9.0)
        self.assertEqual(spec["hparams"]["ent_coef"], 0.02)
        self.assertEqual(spec["recommendation"]["changes"]["target_speed"], 9.0)


@requires_sb3
class TestTrainingSmoke(unittest.TestCase):
    """Tiny real training runs: fresh, auto-resume after a 'disconnect', fine-tune from a model."""

    ARGS = ["--timesteps", "512", "--n_steps", "256", "--batch_size", "64", "--checkpoint_freq", "256",
            "--preset", "laptime"]

    def run_main(self, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            path = main(list(self.ARGS) + list(extra))
        return path, out.getvalue()

    def test_fresh_resume_and_finetune(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = ["--run_name", "smoke", "--output_dir", tmp]
            final, log = self.run_main(*base)
            run_dir = Path(tmp) / "runs" / "smoke"
            self.assertIn("from scratch", log)
            self.assertEqual(final.name, "ppo_cota_smoke_512_steps.zip")
            self.assertTrue((run_dir / "run.json").exists())

            _, log = self.run_main(*base)
            self.assertIn("already complete", log)

            # Simulate a disconnect after the first checkpoint.
            final.unlink()
            final2, log = self.run_main(*base)
            self.assertIn("Resuming smoke from ppo_cota_smoke_256_steps.zip", log)
            self.assertEqual(final2.name, "ppo_cota_smoke_512_steps.zip")

            # Changing settings under the same run name is refused.
            with self.assertRaises(SystemExit):
                self.run_main(*base, "--seed", "1")

            # Fine-tune: a new run starting from the finished model ends at 512 + 512 steps.
            ft = ["--run_name", "smoke_ft", "--output_dir", tmp, "--resume", str(final2)]
            final3, log = self.run_main(*ft)
            self.assertIn("Starting smoke_ft from", log)
            self.assertEqual(final3.name, "ppo_cota_smoke_ft_1024_steps.zip")
            meta = json.loads((Path(tmp) / "runs" / "smoke_ft" / "run.json").read_text())
            self.assertEqual(meta["start_timesteps"], 512)


if __name__ == "__main__":
    unittest.main()
