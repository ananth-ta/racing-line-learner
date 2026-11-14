"""Evaluate a saved model's lap times.

    python -m racing_line_learner.eval --model runs/cota_v2r_s0/checkpoints/....zip --preset v2 \\
        --tracks cota sepang_wide sepang
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from ..config import get_preset, parse_overrides
from .laps import MODES, evaluate


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m racing_line_learner.eval", description="Lap-time evaluation.")
    p.add_argument("--model", required=True)
    p.add_argument("--preset", default="v2", help="Preset the model was trained with.")
    p.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--tracks", nargs="+", default=["cota"])
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    p.add_argument("--repeats", type=int, default=5, help="Jittered repeats for flying/outlap/standing laps.")
    p.add_argument("--n_starts", type=int, default=16, help="Start points for the robustness mode.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", default=None, help="Write full results (with per-run details) here.")
    args = p.parse_args(argv)

    from stable_baselines3 import PPO

    config = replace(get_preset(args.preset), **parse_overrides(args.env))
    model = PPO.load(args.model, device="cpu")
    results = {"model": args.model, "preset": args.preset,
               "tracks": [evaluate(model, t, config, args.modes, args.repeats, args.n_starts, seed=args.seed)
                          for t in args.tracks]}
    print(f"{'track':12s} {'mode':11s} {'laps':>7s} {'mean':>8s} {'std':>6s} {'best':>8s} {'v@line':>7s}  failures")
    for tr in results["tracks"]:
        for mode in args.modes:
            s = tr[mode]
            if "skipped" in s:
                print(f"{tr['track']:12s} {mode:11s} skipped: {s['skipped']}")
                continue
            fmt = lambda v, w: f"{v:>{w}}" if v is not None else f"{'-':>{w}}"
            fails = ", ".join(f"wp{f['waypoint']}x{f['count']}" for f in s["failures_at"][:4]) or "-"
            print(f"{tr['track']:12s} {mode:11s} {s['completed']:>3d}/{s['runs']:<3d} {fmt(s['lap_time_mean'], 8)} "
                  f"{fmt(s['lap_time_std'], 6)} {fmt(s['lap_time_best'], 8)} {fmt(s['line_speed_mean'], 7)}  {fails}")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
