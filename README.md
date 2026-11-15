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
  env/laps.py        start/finish line lap timer (flying laps, standing starts)
  train/ppo.py       resumable PPO training entry point
  eval/laps.py       lap-time evaluation protocol (flying, out-lap, standing, robustness)
  in_loop/telemetry.py  window telemetry: crashes, segments, speed vs grip, smoothness, reward breakdown
  in_loop/detectors.py  deterministic findings + which levers address each finding
  in_loop/validator.py  guardrails for proposals: step cap, cooldowns, grounding
  in_loop/analyze.py    offline telemetry report for a saved model
  in_loop/supervisor.py SB3 callback: probe, judge/rollback, propose (rules, replay, stub), validate, apply
scripts/             Colab launchers (setup, run)
tests/               unit tests (simulator tests skip if f110_gym is not installed)
maps/                maps and racing lines (cota = Circuit of the Americas)
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
| `v2` | As `laptime` | Normalised actions and observations, action repeat 4 (25 Hz policy), 2 m/s speed floor, no steering filter, 8-sample curvature lookahead |

Ablations override single fields of a preset, and `--n_envs` runs several simulators in parallel (`--n_steps` is per environment):

```bash
python -m racing_line_learner.train --run_name cota_v2_s0 --preset v2 --n_envs 8 --n_steps 512 --timesteps 1250000
python -m racing_line_learner.train --run_name cota_v2_floor65_s0 --preset v2 --env speed_floor=6.5 --env speed_mid=8.0 --n_envs 8 --n_steps 512 --timesteps 1250000
```

`--timesteps` counts policy steps. With action repeat 4, 1.25M policy steps are 5M simulator steps, the same driving time as a 5M-step `legacy` or `laptime` run.

Use the environment directly:

```python
from racing_line_learner.config import get_preset
from racing_line_learner.env.wrapper import make_env

env = make_env("cota", get_preset("laptime"))
obs = env.reset(spawn_idx=100)            # spawn at waypoint 100, facing along the line
obs, reward, done, info = env.step([0.0, 8.0])
info["termination"], info["reward_terms"]   # None | collision | stagnation | lap_complete | time_limit
```

Evaluate a model's lap times (two-lap flying lap as the headline, plus out-lap, standing-start and 16-start robustness modes, each with jittered repeats):

```bash
python -m racing_line_learner.eval --model path/to/model.zip --preset v2 --tracks cota sepang_wide sepang
```

## In-loop telemetry and levers

`python -m racing_line_learner.in_loop.analyze --model path/to/model.zip --preset v2` prints the report the in-loop
supervisor reads, with its deterministic findings, for example:

```
[high]   67% of collisions occur at waypoints 35-35; approach speed 9.4 m/s exceeds the corner's grip limit of about 7.4 m/s
[medium] on straights the steering reverses 8.4 times per second and the car drifts 0.38 m (std) around the racing line
[medium] on straights the speed command changes by 0.86 m/s per step on average ...
```

Telemetry is grouped by **corner and straight segments derived from the racing line's curvature**
(`Track.corner_segments()`), so it works on any track without hand-placed boundaries. Smoothness is measured on
deterministic probe laps, because training rollouts mostly show exploration noise.

The supervisor proposes changes as `recommendations.json`. **Schema v2** (`schema_version: 2`, for the `v2` preset)
allows only track-independent levers:

| Lever | Default | Range | Addresses |
|---|---|---|---|
| `jitter_coef`, `steering_penalty_w` | 0.05, 2.0 | 0.01–0.5, 0.5–3.0 | weaving, steering saturation |
| `throttle_change_coef` | 0 (off) | 0–0.5 | throttle jitter |
| `overspeed_coef`, `grip_margin` | 0 (off), 1.0 | 0–2, 0.8–1.5 | overspeed into corners |
| `time_cost`, `progress_coef`, `lap_bonus`, `waypoint_bonus` | 0.1, 10, 100, 50 | see `schema.WHITELIST_V2` | pace, stagnation |
| `spawn_random`, `spawn_corner_mix` | 0.75, 0 | 0–0.9, 0–0.5 | coverage, corner practice |
| `ent_coef` | 0.01 | 0.001–0.05 | exploration |

Run training under the supervisor with `--supervisor rules` (the rule-based tuner), `stub` (telemetry and probes, no
changes) or `replay --replay_log <proposals.jsonl>` (re-apply a logged run):

```bash
python -m racing_line_learner.train --run_name cota_rules_s0 --preset v2 --supervisor rules --window_steps 250000 \
    --n_envs 8 --n_steps 512 --timesteps 1250000
```

At each window boundary (at the end of a PPO rollout) it runs a deterministic probe of jittered out-laps, judges the
previous change (rolling it back from a pre-change checkpoint if completion or lap time got worse beyond the probe's
noise), asks the proposer for a change, validates it and applies it to every env. It writes `proposals.jsonl`,
`telemetry/w*.json` and `supervisor_state.json` in the run directory, so a restarted run resumes without repeating a
decision.

Every change must cite in `grounds` the current findings that justify it. The validator rejects a change whose cited
finding is not in the report or does not list that lever (`in_loop.detectors.LEVERS`), caps each move at 25% of its
range per window, and enforces cooldowns after changes and rollbacks. Schema v1 (used by the post-loop agent) is unchanged.

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
