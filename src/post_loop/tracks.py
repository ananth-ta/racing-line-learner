"""Step 0: track settings.

Each track name maps to its map image and waypoint CSV. To add the held-out
track later, run it through maps/linemapper.py and add one entry to TRACKS.
"""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Track:
    name: str
    map_path: str        # without extension, as gym.make(map=...) expects
    map_ext: str
    waypoints_csv: str
    start_pose: tuple    # (x, y, theta) used by every run

    @property
    def map_image(self) -> Path:
        return REPO_ROOT / f"{self.map_path}{self.map_ext}"

    @property
    def map_yaml(self) -> Path:
        return REPO_ROOT / f"{self.map_path}.yaml"


TRACKS = {
    # Circuit of the Americas. The files are mislabeled "vegas" in the repo.
    "cota": Track(
        name="cota",
        map_path="maps/vegas",
        map_ext=".png",
        waypoints_csv="maps/vegas_optimal_line.csv",
        start_pose=(0.0, 0.0, 0.0),
    ),
}


def get_track(name: str) -> Track:
    if name not in TRACKS:
        raise ValueError(f"Unknown track '{name}'. Known tracks: {sorted(TRACKS)}")
    return TRACKS[name]


def load_waypoints(track: Track) -> np.ndarray:
    """Return waypoints as an (N, 2) array of x, y in metres, in CSV order."""
    rows = []
    with open(REPO_ROOT / track.waypoints_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if row:
                rows.append((float(row[0]), float(row[1])))
    return np.array(rows, dtype=np.float64)
