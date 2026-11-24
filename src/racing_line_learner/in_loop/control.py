"""Control branch for the supervisor's judge: continue a checkpoint for one window *without* the change.

The before/after judge compares the probe taken just before a change with the
probe one window later. The policy keeps learning in between, so ordinary
training progress is credited to the change (in the first arms almost every
change looked like an improvement). The control branch removes that drift: when
a change is applied, this module trains the same checkpoint, in a separate
process, for the same number of steps under the *unchanged* config. At the next
boundary the judge compares the changed branch against the control branch, both
probed with the same deterministic jittered laps.

If the change loses, the supervisor switches to the control branch: the policy
trained without the change, so the window is not thrown away as it is when
rolling back to the pre-change checkpoint.

What this does not remove is the noise between two training runs from the same
checkpoint, so the judge keeps its noise margins.

Run as a module (the supervisor starts it with ``subprocess``)::

    python -m racing_line_learner.in_loop.control --checkpoint pre_w0002.zip --config config.json \\
        --track cota --until 750000 --n_envs 8 --seed 10002 --out control_w0002.zip
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence


def train_control(checkpoint: str, config_path: str, track: str, until: int, n_envs: int, seed: int, out: str,
                  ent_coef: Optional[float] = None, device: str = "cpu") -> Path:
    """Load ``checkpoint``, train it under the config in ``config_path`` until ``until`` total steps, save to ``out``.

    The model keeps its own learning-rate schedule, optimizer state and step counter, exactly as the main run does
    after the same checkpoint. The file is written under a temporary name and renamed, so a partly written control
    model is never mistaken for a finished one.
    """
    from stable_baselines3 import PPO

    from ..config import EnvConfig
    from ..train.ppo import make_training_env

    config = EnvConfig.from_dict(json.loads(Path(config_path).read_text()))
    env = make_training_env(track, config, n_envs, seed)
    model = PPO.load(checkpoint, env=env, device=device)
    if ent_coef is not None:
        model.ent_coef = float(ent_coef)
    remaining = int(until) - int(model.num_timesteps)
    if remaining > 0:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.stem + ".partial.zip")       # SB3 keeps an explicit suffix as is
    model.save(str(tmp))
    os.replace(str(tmp), str(out_path))
    env.close()
    return out_path


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(prog="python -m racing_line_learner.in_loop.control", description=__doc__.split("\n\n")[0])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config", required=True, help="EnvConfig as JSON (the config *without* the change).")
    p.add_argument("--track", required=True)
    p.add_argument("--until", type=int, required=True, help="Train until the model's step counter reaches this.")
    p.add_argument("--n_envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ent_coef", type=float, default=None)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    path = train_control(args.checkpoint, args.config, args.track, args.until, args.n_envs, args.seed, args.out,
                         args.ent_coef)
    print(f"control branch saved: {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
