"""Build a telemetry report for a saved model, without training.

    python -m racing_line_learner.in_loop.analyze --model path/to/model.zip --preset laptime

Drives ``--episodes`` stochastic episodes (like training rollouts) from a
mix of the default start and random spawn points to fill the report, then a
deterministic probe from evenly spaced start points. Prints a short summary
with the detector findings; ``--json`` writes the full report.
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..config import get_preset, parse_overrides
from .detectors import detect
from .telemetry import TelemetryCollector, run_episodes, summarize_probe


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m racing_line_learner.in_loop.analyze",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, help="Saved SB3 PPO model (.zip).")
    p.add_argument("--preset", default="laptime", help="Preset the model was trained with.")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="EnvConfig override.")
    p.add_argument("--track", default="cota")
    p.add_argument("--episodes", type=int, default=30, help="Stochastic episodes for the report.")
    p.add_argument("--probe", type=int, default=8, help="Deterministic probe episodes (evenly spaced starts).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="Write the full report here.")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> dict:
    args = parse_args(argv)
    from stable_baselines3 import PPO

    from ..env.wrapper import make_env

    config = replace(get_preset(args.preset), **parse_overrides(args.env))
    # One lap per episode and a hard time limit, so every episode yields a result.
    config = replace(config, max_laps=1, max_episode_steps=config.max_episode_steps or 15000)
    env = make_env(args.track, config, seed=args.seed)
    model = PPO.load(args.model, device="cpu")
    model.set_random_seed(args.seed)

    rng = np.random.RandomState(args.seed)
    n = env.track.n_waypoints
    spawns = [None if rng.rand() < 0.5 else int(rng.randint(n)) for _ in range(args.episodes)]
    collector = TelemetryCollector(env.track, config)
    run_episodes(model, env, collector, spawns, deterministic=False)

    probe_spawns = [None] + [int(i) for i in np.linspace(0, n, args.probe, endpoint=False)[1:]]
    probe = summarize_probe(run_episodes(model, env, None, probe_spawns, deterministic=True))
    env.close()

    report = collector.report(probe=probe)
    report["findings"] = detect(report)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    print_summary(report)
    return report


def print_summary(report: dict, out=sys.stdout) -> None:
    w = lambda s="": print(s, file=out)
    laps, sp, probe = report["laps"], report["speed"], report.get("probe", {})
    w(f"Episodes {report['episodes']}: " + ", ".join(f"{k} {v['count']}" for k, v in report["terminations"].items()))
    w(f"Laps completed {laps['completed']} (rate {laps['completion_rate']}), best {laps['lap_time_best']} s; "
      f"mean speed {sp['mean']} m/s, at floor {sp['pinned_at_floor_share']:.0%}, at max {sp['pinned_at_max_share']:.0%}")
    if probe:
        w(f"Deterministic probe: {probe['completion_rate']:.0%} of {probe['episodes']} laps completed, "
          f"best {probe['lap_time_best']} s, failures at {probe['failures_at']}")
    w("Crash hotspots: " + ("; ".join(f"wp {h['waypoints'][0]}-{h['waypoints'][1]} {h['share']:.0%} "
                                      f"(approach {h['approach_speed']} m/s, grip ~{h['grip_limit']})"
                                      for h in report["crash_hotspots"]) or "none"))
    w("Findings:")
    for f in report["findings"] or [{"severity": "-", "message": "none"}]:
        w(f"  [{f['severity']}] {f['message']}")


if __name__ == "__main__":
    main()
