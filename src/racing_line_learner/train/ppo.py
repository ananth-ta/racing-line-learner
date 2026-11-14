"""Resumable PPO training.

Usage::

    python -m racing_line_learner.train --run_name cota_laptime_s0 --preset laptime --seed 0

Each run lives in ``<output root>/runs/<run_name>/``::

    run.json                     # what this run is: track, preset, env config, hyperparameters
    checkpoints/ppo_<track>_<run>_<steps>_steps.zip
    tb/                          # TensorBoard logs

The output root is ``--output_dir``, else ``$RLL_OUTPUT_DIR``, else the repo
root. Re-running the same command after a disconnect resumes from the
newest checkpoint (``--resume auto``, the default). ``--resume PATH``
starts a new run from an existing model (fine-tuning, e.g. with a
post-loop ``--config recommendations.json``); later re-runs of that command
resume the new run's own checkpoints instead.

``--timesteps`` is the number of policy steps *this run* trains for, counted
from its starting model, so fine-tuning a 5M-step model with ``--timesteps
1000000`` ends at 6M. With ``action_repeat`` > 1 each policy step covers
several simulator steps.

``--env key=value`` overrides one :class:`EnvConfig` field of the preset
(repeatable), for ablations: ``--preset v2 --env speed_floor=6.5``.
``--n_envs N`` collects experience from N simulator processes in parallel;
``--n_steps`` is per environment, so a rollout is ``n_steps * n_envs`` steps.
"""

import functools

import argparse
import json
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..config import EnvConfig, apply_recommendation, get_preset, output_root, parse_overrides, save_json
from ..schema import load_recommendation

DEFAULT_HPARAMS: Dict[str, Any] = {
    "learning_rate": 3e-4,   # linearly decayed to 0 over the run
    "n_steps": 2048,
    "batch_size": 256,
    "ent_coef": 0.01,
    "gamma": 0.99,
    "log_std_init": 0.0,     # initial log std of the Gaussian policy (SB3 default; fresh models only)
}

#: Per-preset training defaults (applied before recommendations and CLI flags).
#: v2: a longer horizon (gamma 0.998 at 25 Hz is ~20 s) and smaller initial
#: exploration noise, so braking for corners is both visible and learnable.
PRESET_HPARAMS: Dict[str, Dict[str, Any]] = {
    "v2": {"gamma": 0.998, "log_std_init": -0.7},
}

_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class RunLinearSchedule:
    """Learning rate decaying linearly to 0 over *this run's* steps.

    SB3 passes ``progress_remaining = 1 - num_timesteps / total`` where
    ``total`` is the step count at the end of the run (start + run steps).
    For a run from scratch this is exactly ``initial * progress_remaining``
    (the original schedule); for a fine-tuning run it restarts at
    ``initial`` instead of continuing from the parent run's decayed value.
    Defined at module level so checkpoints can pickle it.
    """

    def __init__(self, initial: float, start_timesteps: int, run_timesteps: int):
        self.initial = float(initial)
        self.end = int(start_timesteps) + int(run_timesteps)
        self.run_timesteps = int(run_timesteps)

    def __call__(self, progress_remaining: float) -> float:
        remaining_steps = progress_remaining * self.end
        return self.initial * min(1.0, max(0.0, remaining_steps / self.run_timesteps))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m racing_line_learner.train", description=__doc__.split("\n\n")[0])
    p.add_argument("--run_name", required=True, help="Run identifier; names the run directory and checkpoints.")
    p.add_argument("--track", default="cota", help="Registered track name (default: cota).")
    p.add_argument("--preset", default="legacy", help="Environment/reward preset: legacy, laptime or v2.")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                   help="Override one EnvConfig field of the preset (repeatable), e.g. speed_floor=6.5.")
    p.add_argument("--n_envs", type=int, default=1, help="Parallel simulator processes (default: 1).")
    p.add_argument("--timesteps", type=int, default=5_000_000, help="Steps to train in this run.")
    p.add_argument("--config", default=None, help="recommendations.json to apply on top of the preset.")
    p.add_argument("--resume", default="auto",
                   help="'auto' (newest checkpoint in this run, else fresh), 'none' (fresh; refuse if "
                        "checkpoints exist), or a model .zip to start this run from.")
    p.add_argument("--output_dir", default=None, help="Output root (default: $RLL_OUTPUT_DIR or repo root).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="Torch device (the small MLP policy trains fastest on cpu).")
    p.add_argument("--checkpoint_freq", type=int, default=100_000, help="Save a checkpoint every N steps.")
    for key, value in DEFAULT_HPARAMS.items():
        p.add_argument(f"--{key}", type=type(value), default=None, help=f"PPO {key} (default: {value}).")
    args = p.parse_args(argv)
    if not _RUN_NAME_RE.match(args.run_name):
        p.error("--run_name may contain only letters, digits, '_', '.', '-'")
    if args.timesteps < 1 or args.checkpoint_freq < 1 or args.n_envs < 1:
        p.error("--timesteps, --checkpoint_freq and --n_envs must be >= 1")
    return args


def model_prefix(track: str, run_name: str) -> str:
    return f"ppo_{track}_{run_name}"


def find_latest_checkpoint(ckpt_dir: Path, prefix: str) -> Optional[Tuple[int, Path]]:
    """Newest ``<prefix>_<steps>_steps.zip`` in ``ckpt_dir`` as ``(steps, path)``."""
    pattern = re.compile(re.escape(prefix) + r"_(\d+)_steps\.zip$")
    found = []
    if ckpt_dir.is_dir():
        for path in ckpt_dir.iterdir():
            m = pattern.match(path.name)
            if m:
                found.append((int(m.group(1)), path))
    return max(found) if found else None


def build_run_spec(args: argparse.Namespace) -> Dict[str, Any]:
    """Everything that defines a run; must match exactly when resuming."""
    from ..env.tracks import load_track

    track = load_track(args.track)
    try:
        overrides = parse_overrides(args.env)
        env_config = replace(get_preset(args.preset), **overrides)
    except ValueError as exc:
        raise SystemExit(f"invalid --preset/--env: {exc}") from None
    algo_overrides: Dict[str, float] = {}
    recommendation = None
    if args.config:
        rec = load_recommendation(args.config, track.n_waypoints, speed_floor=env_config.speed_floor,
                                  expected_track=args.track)
        env_config, algo_overrides = apply_recommendation(env_config, rec)
        recommendation = rec.to_dict()
        if rec.speed_plan:
            print("WARNING: speed_plan is not consumed yet (Phase 3); ignoring it.", file=sys.stderr)

    hparams = dict(DEFAULT_HPARAMS)
    hparams.update(PRESET_HPARAMS.get(args.preset, {}))
    hparams.update(algo_overrides)
    for key in DEFAULT_HPARAMS:
        if getattr(args, key) is not None:
            hparams[key] = getattr(args, key)

    return {
        "run_name": args.run_name,
        "track": args.track,
        "n_waypoints": track.n_waypoints,
        "preset": args.preset,
        "env_overrides": {k: list(v) if isinstance(v, tuple) else v for k, v in overrides.items()},
        "env_config": env_config.to_dict(),
        "hparams": hparams,
        "n_envs": args.n_envs,
        "seed": args.seed,
        "timesteps": args.timesteps,
        "recommendation": recommendation,
    }


#: Defaults for spec keys that older run.json files do not have.
_SPEC_DEFAULTS = {"n_envs": 1, "env_overrides": {}}


def _spec_mismatches(saved: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    """Settings that differ between a saved run and the current command.

    ``env_config`` is compared as an :class:`EnvConfig`, so fields added to
    the config after a run started (with their defaults) do not block
    resuming it.
    """
    out = []
    for key in ("track", "preset", "env_overrides", "n_envs", "seed", "timesteps", "recommendation"):
        if saved.get(key, _SPEC_DEFAULTS.get(key)) != current.get(key, _SPEC_DEFAULTS.get(key)):
            out.append(key)
    # Hyperparameters added later default to their (behaviour-preserving) defaults.
    if dict(DEFAULT_HPARAMS, **saved.get("hparams", {})) != dict(DEFAULT_HPARAMS, **current.get("hparams", {})):
        out.append("hparams")
    try:
        if EnvConfig.from_dict(saved["env_config"]) != EnvConfig.from_dict(current["env_config"]):
            out.append("env_config")
    except (KeyError, ValueError, TypeError):
        out.append("env_config")
    return out


def make_training_env(track: str, env_config: EnvConfig, n_envs: int, seed: int):
    """One wrapped env (SB3 adds Monitor + DummyVecEnv), or ``n_envs`` subprocesses."""
    from ..env.wrapper import make_env

    if n_envs == 1:
        return make_env(track, env_config, seed=seed)
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv

    return make_vec_env(functools.partial(make_env, track, env_config), n_envs=n_envs, seed=seed,
                        vec_env_cls=SubprocVecEnv)


def main(argv: Optional[Sequence[str]] = None) -> Path:
    args = parse_args(argv)

    # Heavy imports after argument parsing so --help is fast.
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    spec = build_run_spec(args)
    run_dir = output_root(args.output_dir) / "runs" / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    tb_dir = run_dir / "tb"
    run_file = run_dir / "run.json"
    prefix = model_prefix(args.track, args.run_name)
    latest = find_latest_checkpoint(ckpt_dir, prefix)

    if args.resume == "none" and latest is not None:
        raise SystemExit(f"{ckpt_dir} already has checkpoints; use a new --run_name or --resume auto")

    requested_init = None if args.resume in ("auto", "none") else str(Path(args.resume).expanduser().resolve())
    if requested_init is not None and not Path(requested_init).is_file():
        raise SystemExit(f"--resume model not found: {requested_init}")

    if run_file.exists():
        saved = json.loads(run_file.read_text())
        mismatched = _spec_mismatches(saved, spec)
        if requested_init is not None and requested_init != saved.get("init_model"):
            mismatched.append("init_model")
        if mismatched:
            raise SystemExit(f"run {args.run_name!r} exists with different settings ({', '.join(mismatched)}); "
                             f"use a new --run_name")
        start_timesteps = int(saved["start_timesteps"])
        init_model = saved.get("init_model")
    else:
        start_timesteps = 0
        init_model = requested_init

    env_config = EnvConfig.from_dict(spec["env_config"])
    env = make_training_env(args.track, env_config, args.n_envs, args.seed)
    hp = spec["hparams"]

    def load(path: str, start: int):
        schedule = RunLinearSchedule(hp["learning_rate"], start, args.timesteps)
        return PPO.load(path, env=env, device=args.device,
                        custom_objects={"learning_rate": schedule, "ent_coef": hp["ent_coef"]})

    if latest is not None:
        steps, path = latest
        print(f"Resuming {args.run_name} from {path.name} ({steps} steps)")
        model = load(str(path), start_timesteps)
    elif init_model is not None:
        print(f"Starting {args.run_name} from {init_model}")
        if not run_file.exists():
            start_timesteps = int(PPO.load(init_model, device=args.device).num_timesteps)
        model = load(init_model, start_timesteps)
    else:
        print(f"Starting {args.run_name} from scratch")
        schedule = RunLinearSchedule(hp["learning_rate"], 0, args.timesteps)
        model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, device=args.device,
                    learning_rate=schedule, n_steps=hp["n_steps"], batch_size=hp["batch_size"],
                    ent_coef=hp["ent_coef"], gamma=hp["gamma"],
                    policy_kwargs={"log_std_init": hp["log_std_init"]})
    model.tensorboard_log = str(tb_dir)

    if not run_file.exists():
        save_json(dict(spec, start_timesteps=start_timesteps, init_model=init_model,
                       created=time.strftime("%Y-%m-%dT%H:%M:%S")), run_file)

    done_steps = int(model.num_timesteps) - start_timesteps
    remaining = args.timesteps - done_steps
    if remaining <= 0:
        final_path = latest[1] if latest is not None else ckpt_dir / f"{prefix}_{model.num_timesteps}_steps.zip"
        print(f"Run {args.run_name} already complete ({done_steps} steps): {final_path}")
        env.close()
        return final_path

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # CheckpointCallback counts vectorised steps (n_envs env steps each).
    callback = CheckpointCallback(save_freq=max(1, args.checkpoint_freq // args.n_envs), save_path=str(ckpt_dir),
                                  name_prefix=prefix)
    print(f"Training {remaining} steps ({done_steps}/{args.timesteps} done) -> {run_dir}")
    model.learn(total_timesteps=remaining, callback=callback, reset_num_timesteps=False, tb_log_name="ppo")

    final_path = ckpt_dir / f"{prefix}_{model.num_timesteps}_steps.zip"
    model.save(str(final_path))
    print(f"Training complete: {final_path}")
    env.close()
    return final_path


if __name__ == "__main__":
    main()
