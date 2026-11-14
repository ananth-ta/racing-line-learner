"""Lap timing against the start/finish line (arc length 0 on the racing line).

The simulator counts a lap when the car returns to wherever it spawned.
That makes lap times from different start points incomparable and rules out
flying laps. :class:`LineLapTimer` instead times the interval between two
forward crossings of the start/finish line, like a timing loop:

* **flying lap**: spawn before the line, build speed on the out-lap; the clock
  starts at the first crossing and stops at the next;
* **standing start**: spawn on (or within ``standing_tolerance`` metres of)
  the line; the clock starts at once.

A crossing only completes a lap after at least ``min_lap_fraction`` of the
track length of net forward progress, so driving back and forth over the
line cannot fake a lap.
"""

from typing import List, Optional


class LineLapTimer:
    def __init__(self, track_length: float, min_lap_fraction: float = 0.9, standing_tolerance: float = 2.0):
        if track_length <= 0:
            raise ValueError("track_length must be positive")
        self.length = float(track_length)
        self.min_lap_fraction = min_lap_fraction
        self.standing_tolerance = standing_tolerance
        self.reset(s0=self.length / 2)

    def reset(self, s0: float, t0: float = 0.0) -> None:
        """Start an episode at arc length ``s0`` (in ``[0, length)``) and time ``t0``."""
        self.laps = 0
        self.lap_times: List[float] = []
        signed = s0 if s0 < self.length / 2 else s0 - self.length  # metres past the line (negative: before it)
        if abs(signed) <= self.standing_tolerance:
            self.running, self.t_start, self.progress = True, float(t0), signed
        else:
            self.running, self.t_start, self.progress = False, None, 0.0

    def update(self, s_prev: float, ds: float, t: float) -> Optional[float]:
        """Advance by signed progress ``ds`` from ``s_prev``; return a lap time if one just completed."""
        s_new = s_prev + ds
        crossed_forward = s_new >= self.length
        if self.running:
            self.progress += ds
        if not crossed_forward:
            return None
        past_line = s_new - self.length
        if not self.running:
            self.running, self.t_start, self.progress = True, float(t), past_line
            return None
        if self.progress < self.min_lap_fraction * self.length:
            return None
        lap = float(t) - self.t_start
        self.laps += 1
        self.lap_times.append(lap)
        self.t_start, self.progress = float(t), past_line
        return lap
