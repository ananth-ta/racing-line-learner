# Intelligent Racing Line Learner

CS 5130 (Fall 2026) team project at Northeastern University, built on our CS 5180 (Spring 2026) project, which compared PPO and a genetic algorithm in F1TENTH racing.

A PPO policy drives the car in the [F1TENTH Gym](https://github.com/f1tenth/f1tenth_gym) simulator. Two bounded reasoning agents work around it:

- **In-loop supervisor** (during training) reads compact telemetry and proposes small, whitelisted changes to reward weights, curriculum or `ent_coef`.
- **Post-loop agent** (after training) diagnoses failures from evaluation laps and writes recommendations for fine-tuning.

The objective is the **fastest completed lap**; finishing the lap is a constraint, not the goal. Both agents write the same `recommendations.json`, validated by [`racing_line_learner.schema`](src/racing_line_learner/schema.py).

## Repository layout

```
src/racing_line_learner/
  config.py          EnvConfig (every reward/physics constant), presets, output root
  schema.py          shared recommendations.json schema + strict loader
  env/tracks.py      track registry, racing-line geometry (arc length, curvature, projection, spawn poses)
  env/rewards.py     reward functions (legacy, laptime), pure and unit-tested
  env/wrapper.py     F1TenthWrapper: observations, action mapping, termination, spawn curriculum
  train/ppo.py       resumable PPO training entry point
  in_loop/validator.py  per-window step cap and cooldown guardrails for in-loop proposals
scripts/             Colab launchers (setup, run)
tests/               unit tests (simulator tests skip if f110_gym is not installed)
maps/                maps and racing lines (COTA is stored as maps/vegas.*)
```

The CS 5180 files (`wrapper.py`, `baseline.py`, `render_*.py`, `ga_*.py`, `models/`) are kept unchanged at the repo root. `wrapper.py` also serves as the reference for the legacy parity test.

## Setup

The stack is pinned and fragile: Python 3.8, gym 0.19.0, stable-baselines3 1.4.0, numpy 1.22.0, f110_gym at commit 4fdb9c7.

```bash
conda create -n f1tenth python=3.8 && conda activate f1tenth
pip install pip==24.0 setuptools==65.5.0 wheel==0.38.4   # newer pip rejects gym 0.19.0
pip install -r requirements.txt
pip install --no-deps --no-build-isolation -e .           # or: export PYTHONPATH=src
```

On Colab, run `bash scripts/colab_setup.sh` once per session.

## Usage

Train (re-running the same command after a disconnect resumes from the newest checkpoint):

```bash
python -m racing_line_learner.train --run_name cota_laptime_s0 --preset laptime --seed 0
```

Each run is written to `<output root>/runs/<run_name>/`: `run.json`, `checkpoints/ppo_<track>_<run>_<steps>_steps.zip` (every 100k steps) and `tb/`. The output root is `--output_dir`, else `$RLL_OUTPUT_DIR`, else the repo root. On Colab, `scripts/colab_run.sh` sets it to `/content/drive/MyDrive/F1Tenth`.

Fine-tune an existing model with recommendations (e.g. from the post-loop agent):

```bash
python -m racing_line_learner.train --run_name cota_ft1 --preset laptime \
    --resume path/to/ppo_cota_..._steps.zip --config recommendations.json --timesteps 1000000
```

Presets:

| Preset | Reward | Notes |
|---|---|---|
| `legacy` | The CS 5180 reward, reproduced exactly | Default; for comparing with old checkpoints |
| `laptime` | Progress along the racing line, minus a per-step time cost, plus a lap bonus | Steering limited to the simulator's ±0.4189 rad; one-lap episodes |

Use the environment directly:

```python
from racing_line_learner.config import get_preset
from racing_line_learner.env.wrapper import make_env

env = make_env("cota", get_preset("laptime"))
obs = env.reset(spawn_idx=100)            # spawn at waypoint 100, facing along the line
obs, reward, done, info = env.step([0.0, 8.0])
info["termination"], info["reward_terms"]   # None | collision | stagnation | lap_complete | time_limit
```

## Tests

```bash
python -m unittest discover -s tests -t .
```

Simulator tests include a parity check: the refactored wrapper with the `legacy` preset gives observations, rewards and dones identical to the original `wrapper.py`.

## Known quirks of the legacy reward

These are preserved in the `legacy` preset and fixed in `laptime`:

- The "speed" reward reads the **yaw rate** (observation index −2), not the velocity, so it pays most for spinning at about 10 rad/s. This is the likely cause of the "corkscrew" exploit.
- Ending the episode, including by finishing it, gives −10.
- The steering action range is ±3.14 rad, but the simulator saturates at ±0.4189 rad.
