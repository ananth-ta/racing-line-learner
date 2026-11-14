"""Generate a racing line (waypoint CSV) from a simulator map.

    python maps/linemapper.py --map maps/sepang --direction cw
    python maps/linemapper.py --map maps/cota --out /tmp/cota_check.csv   # the original COTA settings

Method (unchanged from the original COTA script, now for any map):

1. Find the drivable track. Maps come in two styles: a white track on black
   (COTA), or thin black boundary lines on white (the f1tenth_racetracks
   maps, e.g. Sepang), where the track is the white ring *between* the
   lines. The track is taken as the free-space component with the largest
   perimeter^2 / area, i.e. the long, narrow one.
2. Medial axis (skeleton), ordered into a loop by nearest neighbour.
3. Periodic B-spline through it, sampled every ``--spacing`` metres.
4. Apex shift: each point moves toward the inside of the corner by
   ``apex_gain * curvature`` metres, capped at 80% of the half-width.

Then the loop is oriented to ``--direction`` and rotated so waypoint 0 sits
mid-way along the longest straight (a stand-in for the start/finish line),
unless ``--keep_start``.

Needs scikit-image, which is deliberately *not* in the pinned training
environment; run this in a separate tools env (see README).
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

#: Apex shift in metres per unit curvature (1/m). The original COTA script used
#: 200 px per (1/px) at 0.05 m/px, which is 200 * 0.05^2 = 0.5 m^2.
DEFAULT_APEX_GAIN_M2 = 0.5


def load_map(stem: str):
    meta = yaml.safe_load(Path(stem + ".yaml").read_text())
    img = cv2.imread(stem + ".png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(stem + ".png")
    return img, float(meta["resolution"]), float(meta["origin"][0]), float(meta["origin"][1])


def track_mask(img: np.ndarray) -> np.ndarray:
    """Boolean mask of the drivable track (see module docstring, step 1)."""
    free = (img > 200).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    best, best_score = None, -1.0
    for i in range(1, n):
        mask = (labels == i).astype(np.uint8)
        area = float(stats[i][cv2.CC_STAT_AREA])
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        score = perimeter * perimeter / area
        if score > best_score:
            best, best_score = i, score
    return labels == best


def ordered_skeleton(mask: np.ndarray) -> np.ndarray:
    """Skeleton pixels of the track, ordered into a loop by nearest neighbour."""
    py, px = np.where(skeletonize(mask))
    points = np.column_stack((px, py)).astype(float)
    tree = cKDTree(points)
    ordered = [0]
    unvisited = set(range(1, len(points)))
    cur = 0
    while unvisited:
        k, found = 10, False
        while not found and k <= len(points):
            _, idx = tree.query(points[cur], k=k)
            for j in np.atleast_1d(idx):
                if j in unvisited:
                    ordered.append(j)
                    unvisited.remove(j)
                    cur, found = j, True
                    break
            k *= 2
        if not found:
            break
    return points[ordered]


def racing_line(stem: str, spacing: float, apex_gain: float, direction: str, keep_start: bool):
    img, res, ox, oy = load_map(stem)
    mask = track_mask(img)
    half_width_px = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    pts = ordered_skeleton(mask)[::5]

    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=200, per=True)
    dense = np.array(splev(np.linspace(0, 1, 4000, endpoint=False), tck)).T
    length_m = np.linalg.norm(np.diff(np.vstack([dense, dense[:1]]), axis=0), axis=1).sum() * res
    n = max(10, int(round(length_m / spacing)))
    u = np.linspace(0, 1, n, endpoint=False)
    x, y = splev(u, tck)
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)

    waypoints = []
    for i in range(n):
        norm = math.hypot(dx[i], dy[i])
        nx, ny = -dy[i] / norm, dx[i] / norm                       # normal, pixel frame
        kappa_m = (dx[i] * ddy[i] - dy[i] * ddx[i]) / norm ** 3 / res  # curvature in 1/m
        cap = half_width_px[int(y[i]), int(x[i])] * 0.8
        shift_px = np.clip(apex_gain * kappa_m / res, -cap, cap)
        opx, opy = x[i] + nx * shift_px, y[i] + ny * shift_px
        waypoints.append((opx * res + ox, (img.shape[0] - opy) * res + oy))  # pixel -> metres (y up)
    wp = np.array(waypoints)

    # Orientation: shoelace area > 0 means counter-clockwise in the (y-up) world frame.
    area = 0.5 * np.sum(wp[:, 0] * np.roll(wp[:, 1], -1) - np.roll(wp[:, 0], -1) * wp[:, 1])
    if (direction == "ccw") != (area > 0):
        wp = wp[::-1]

    if not keep_start:
        seg = np.roll(wp, -1, axis=0) - wp
        ang = np.arctan2(seg[:, 1], seg[:, 0])
        turn = np.abs((ang - np.roll(ang, 1) + np.pi) % (2 * np.pi) - np.pi)  # heading change at each waypoint
        straight = turn < 0.05
        best_len, best_mid, run = 0, 0, 0
        for i in range(2 * n):                      # twice round, so runs can wrap
            run = run + 1 if straight[i % n] else 0
            if run > best_len:
                best_len, best_mid = run, (i - run // 2) % n
        wp = np.roll(wp, -best_mid, axis=0)
    return wp, length_m


def main():
    p = argparse.ArgumentParser(description="Generate a racing line CSV from a simulator map.")
    p.add_argument("--map", required=True, help="Map path without extension, e.g. maps/sepang")
    p.add_argument("--out", default=None, help="Output CSV (default: <map>_optimal_line.csv)")
    p.add_argument("--spacing", type=float, default=3.11, help="Waypoint spacing in metres (COTA: 3.11)")
    p.add_argument("--apex_gain", type=float, default=DEFAULT_APEX_GAIN_M2, help="Apex shift, m per 1/m curvature")
    p.add_argument("--direction", choices=("cw", "ccw"), default="ccw", help="Driving direction")
    p.add_argument("--keep_start", action="store_true", help="Do not move waypoint 0 to the longest straight")
    args = p.parse_args()

    wp, length = racing_line(args.map, args.spacing, args.apex_gain, args.direction, args.keep_start)
    out = args.out or args.map + "_optimal_line.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y"])
        w.writerows(wp.tolist())
    print(f"{len(wp)} waypoints ({length:.0f} m centre line, {args.direction}) -> {out}")


if __name__ == "__main__":
    main()
