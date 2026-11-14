"""Track registry and racing-line geometry.

A track is a simulator map (``<map_path>.png`` + ``<map_path>.yaml``) plus a
closed racing line of waypoints (CSV with an ``x,y`` header). Waypoint
indices are stable identifiers used throughout (telemetry, diagnoses,
spawn curricula), so rows are never dropped or reordered here, even
duplicates: zero-length segments are handled in the geometry instead.

Geometry helpers (arc length, curvature, projection, spawn poses) are
public so the post-loop agent can import them for crash localisation and
braking-feasibility checks.
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from ..config import REPO_ROOT

_EPS = 1e-9


@dataclass(frozen=True)
class TrackSpec:
    """Where a track's files live. Relative paths are resolved against the repo root."""

    name: str
    map_path: str          # without extension; the simulator appends .yaml / map_ext
    waypoints_csv: str
    map_ext: str = ".png"


#: Known tracks. ``cota`` is Circuit of the Americas, stored under the
#: historical (wrong) file name ``maps/vegas``.
TRACKS: Dict[str, TrackSpec] = {
    "cota": TrackSpec(name="cota", map_path="maps/vegas", waypoints_csv="maps/vegas_optimal_line.csv"),
}


def register_track(spec: TrackSpec) -> None:
    """Add (or replace) a track in the registry."""
    TRACKS[spec.name] = spec


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


def load_waypoints_csv(path: Union[str, Path]) -> np.ndarray:
    """Read an ``x,y`` CSV (one header row) into an ``(N, 2)`` float array."""
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if row:
                rows.append((float(row[0]), float(row[1])))
    if len(rows) < 3:
        raise ValueError(f"{path}: need at least 3 waypoints, got {len(rows)}")
    return np.asarray(rows, dtype=np.float64)


@dataclass(frozen=True)
class Projection:
    """Closest point on the racing line to a query position."""

    segment: int      # index i of segment waypoint[i] -> waypoint[i+1]
    s: float          # arc length from waypoint 0, in [0, length)
    lateral: float    # signed distance (left of travel direction is positive)


class Track:
    """A loaded track: map location plus closed racing-line geometry."""

    def __init__(self, name: str, map_path: Path, map_ext: str, waypoints: np.ndarray):
        wp = np.asarray(waypoints, dtype=np.float64)
        if wp.ndim != 2 or wp.shape[1] != 2 or len(wp) < 3:
            raise ValueError("waypoints must be an (N>=3, 2) array")
        self.name = name
        self.map_path = Path(map_path)
        self.map_ext = map_ext
        self.waypoints = wp
        self.n_waypoints = len(wp)

        # Segment i runs from waypoint i to waypoint i+1 (the last one closes the loop).
        self.seg_vec = np.roll(wp, -1, axis=0) - wp
        self.seg_len = np.linalg.norm(self.seg_vec, axis=1)
        if not np.any(self.seg_len > _EPS):
            raise ValueError("racing line has zero length")
        self.cum_s = np.concatenate([[0.0], np.cumsum(self.seg_len)[:-1]])
        self.length = float(self.seg_len.sum())
        self.duplicate_indices = tuple(int(i) for i in np.nonzero(self.seg_len <= _EPS)[0])

    # ------------------------------------------------------------------ #
    # Basic geometry
    # ------------------------------------------------------------------ #

    @property
    def map_stem(self) -> str:
        """Path without extension, as ``gym.make('f110_gym:f110-v0', map=...)`` expects."""
        return str(self.map_path)

    def next_distinct(self, i: int) -> int:
        """First waypoint after ``i`` at a different position."""
        j = (i + 1) % self.n_waypoints
        while np.linalg.norm(self.waypoints[j] - self.waypoints[i]) <= _EPS:
            j = (j + 1) % self.n_waypoints
        return j

    def prev_distinct(self, i: int) -> int:
        """Last waypoint before ``i`` at a different position."""
        j = (i - 1) % self.n_waypoints
        while np.linalg.norm(self.waypoints[j] - self.waypoints[i]) <= _EPS:
            j = (j - 1) % self.n_waypoints
        return j

    def heading_at(self, i: int) -> float:
        """Direction (radians) from waypoint ``i`` toward the next distinct waypoint."""
        d = self.waypoints[self.next_distinct(i)] - self.waypoints[i]
        return math.atan2(d[1], d[0])

    def spawn_pose(self, i: int) -> Tuple[float, float, float]:
        """``(x, y, theta)`` at waypoint ``i``, facing along the racing line."""
        self._check_index(i)
        x, y = self.waypoints[i]
        return float(x), float(y), self.heading_at(i)

    def curvature(self) -> np.ndarray:
        """Signed curvature (1/m) at each waypoint, from the circle through its distinct neighbours.

        Positive means the line turns left. Duplicate waypoints get the
        curvature of their distinct neighbours, never a division by zero.
        """
        k = np.zeros(self.n_waypoints)
        for i in range(self.n_waypoints):
            a = self.waypoints[self.prev_distinct(i)]
            b = self.waypoints[i]
            c = self.waypoints[self.next_distinct(i)]
            ab, bc, ca = np.linalg.norm(b - a), np.linalg.norm(c - b), np.linalg.norm(a - c)
            denom = ab * bc * ca
            if denom <= _EPS:
                continue
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            k[i] = 2.0 * cross / denom
        return k

    def smoothed_curvature(self) -> np.ndarray:
        """Curvature averaged over each waypoint and its two neighbours (cached).

        Curvature from waypoints ~3 m apart is noisy; this is what the
        lookahead observation uses.
        """
        if not hasattr(self, "_smooth_k"):
            k = self.curvature()
            self._smooth_k = (np.roll(k, 1) + k + np.roll(k, -1)) / 3.0
        return self._smooth_k

    def curvature_at(self, s) -> np.ndarray:
        """Smoothed curvature at arc length(s) ``s``, linearly interpolated between waypoints."""
        k = self.smoothed_curvature()
        xp = np.append(self.cum_s, self.length)
        fp = np.append(k, k[0])
        return np.interp(np.mod(s, self.length), xp, fp)

    # ------------------------------------------------------------------ #
    # Projection / progress
    # ------------------------------------------------------------------ #

    def project(self, xy, hint: Optional[int] = None, window: int = 15) -> Projection:
        """Closest point on the racing line to ``xy``.

        With ``hint`` (the previous segment), only segments within ``window``
        of it are searched, so the projection cannot jump to another part of
        the track that happens to pass nearby.
        """
        p = np.asarray(xy, dtype=np.float64)
        n = self.n_waypoints
        if hint is None or window * 2 + 1 >= n:
            candidates = range(n)
        else:
            candidates = ((hint + o) % n for o in range(-window, window + 1))
        best = None
        for i in candidates:
            length = self.seg_len[i]
            if length <= _EPS:
                continue
            v = self.seg_vec[i]
            rel = p - self.waypoints[i]
            t = min(max(float(np.dot(rel, v)) / (length * length), 0.0), 1.0)
            closest = self.waypoints[i] + t * v
            d = float(np.linalg.norm(p - closest))
            if best is None or d < best[0]:
                cross = v[0] * rel[1] - v[1] * rel[0]
                best = (d, i, t, math.copysign(d, cross) if d > 0 else 0.0)
        _, i, t, lateral = best
        s = (self.cum_s[i] + t * self.seg_len[i]) % self.length
        return Projection(segment=int(i), s=float(s), lateral=float(lateral))

    def progress_delta(self, s_from: float, s_to: float) -> float:
        """Signed arc-length change from ``s_from`` to ``s_to``, unwrapped across the start line."""
        d = s_to - s_from
        half = self.length / 2.0
        if d > half:
            d -= self.length
        elif d < -half:
            d += self.length
        return d

    def waypoint_ahead(self, s: float) -> int:
        """Index of the first waypoint strictly ahead of arc length ``s``."""
        idx = int(np.searchsorted(self.cum_s, s % self.length, side="right"))
        return idx % self.n_waypoints

    def segment_ranges(self, n_segments: int) -> Tuple[Tuple[int, int], ...]:
        """Split waypoint indices into ``n_segments`` contiguous ``[start, end)`` ranges (for telemetry)."""
        if not 1 <= n_segments <= self.n_waypoints:
            raise ValueError(f"n_segments must be in [1, {self.n_waypoints}]")
        edges = np.linspace(0, self.n_waypoints, n_segments + 1).round().astype(int)
        return tuple((int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]))

    def _check_index(self, i: int) -> None:
        if not 0 <= i < self.n_waypoints:
            raise IndexError(f"waypoint {i} out of range [0, {self.n_waypoints - 1}]")


def load_track(track: Union[str, TrackSpec]) -> Track:
    """Load a registered track by name, or from an explicit :class:`TrackSpec`."""
    if isinstance(track, str):
        try:
            spec = TRACKS[track]
        except KeyError:
            raise ValueError(f"unknown track {track!r}; registered: {sorted(TRACKS)}") from None
    else:
        spec = track
    map_path = _resolve(spec.map_path)
    for ext in (".yaml", spec.map_ext):
        if not Path(str(map_path) + ext).exists():
            raise FileNotFoundError(f"track {spec.name!r}: missing {map_path}{ext}")
    return Track(spec.name, map_path, spec.map_ext, load_waypoints_csv(_resolve(spec.waypoints_csv)))
