"""Draw a racing line and its waypoint indices on the map, for visual checking.

    python maps/verify_raceline.py                       # COTA (defaults)
    python maps/verify_raceline.py --map maps/sepang     # -> maps/sepang_annotated.png

Waypoint 0 is drawn larger, with an arrow showing the driving direction.
"""

import argparse

import cv2
import numpy as np
import pandas as pd
import yaml


def annotate_map(stem: str, csv_path: str, out_path: str) -> None:
    meta = yaml.safe_load(open(stem + ".yaml"))
    res, ox, oy = float(meta["resolution"]), float(meta["origin"][0]), float(meta["origin"][1])
    img = cv2.imread(stem + ".png")
    if img is None:
        raise FileNotFoundError(stem + ".png")
    height = img.shape[0]
    df = pd.read_csv(csv_path)

    points = []
    for index, row in df.iterrows():
        # Metres -> pixels (inverse of the YAML transform; image y points down).
        px = int((row["x"] - ox) / res)
        py = int(height - ((row["y"] - oy) / res))
        points.append((px, py))
        cv2.circle(img, (px, py), radius=3, color=(0, 0, 255), thickness=-1)
        cv2.putText(img, str(index), (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    for i in range(len(points)):
        cv2.line(img, points[i], points[(i + 1) % len(points)], color=(0, 255, 0), thickness=2)
    cv2.circle(img, points[0], radius=8, color=(255, 0, 255), thickness=2)
    cv2.arrowedLine(img, points[0], points[2], color=(255, 0, 255), thickness=2, tipLength=0.4)

    cv2.imwrite(out_path, img)
    print(f"{len(points)} waypoints drawn -> {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--map", default="maps/cota", help="Map path without extension")
    p.add_argument("--csv", default=None, help="Waypoint CSV (default: <map>_optimal_line.csv)")
    p.add_argument("--out", default=None, help="Output image (default: <map>_annotated.png)")
    a = p.parse_args()
    annotate_map(a.map, a.csv or a.map + "_optimal_line.csv", a.out or a.map + "_annotated.png")


if __name__ == "__main__":
    main()
