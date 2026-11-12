import math
import unittest

import numpy as np

from racing_line_learner.env.tracks import TRACKS, Track, TrackSpec, load_track


def square(n_per_side=4, side=10.0, duplicate=None):
    pts = []
    corners = [(0, 0), (side, 0), (side, side), (0, side)]
    for k in range(4):
        a, b = np.array(corners[k]), np.array(corners[(k + 1) % 4])
        for j in range(n_per_side):
            pts.append(a + (b - a) * j / n_per_side)
    if duplicate is not None:
        pts.insert(duplicate + 1, pts[duplicate].copy())
    return Track("square", map_path="unused", map_ext=".png", waypoints=np.array(pts))


class TestGeometry(unittest.TestCase):
    def test_length_and_arc(self):
        t = square()
        self.assertAlmostEqual(t.length, 40.0)
        self.assertAlmostEqual(t.cum_s[4], 10.0)

    def test_projection(self):
        t = square()
        p = t.project((5.0, 1.0))
        self.assertEqual(p.segment, 1)
        self.assertAlmostEqual(p.s, 5.0)
        self.assertAlmostEqual(p.lateral, 1.0)       # left of travel (+x heading, +y is left)
        self.assertAlmostEqual(t.project((5.0, -1.0)).lateral, -1.0)

    def test_projection_hint_limits_search(self):
        t = square(n_per_side=10)
        # True answer is on the last side (segments 30-39); a hint on the second side keeps it there.
        far = t.project((0.0, 5.0), hint=15, window=2)
        self.assertIn(far.segment, range(13, 18))
        self.assertIn(t.project((0.0, 5.0)).segment, range(30, 40))

    def test_progress_delta_wraps(self):
        t = square()
        self.assertAlmostEqual(t.progress_delta(39.5, 0.5), 1.0)
        self.assertAlmostEqual(t.progress_delta(0.5, 39.5), -1.0)
        self.assertAlmostEqual(t.progress_delta(10.0, 12.0), 2.0)

    def test_waypoint_ahead(self):
        t = square()
        self.assertEqual(t.waypoint_ahead(0.0), 1)
        self.assertEqual(t.waypoint_ahead(9.99), 4)
        self.assertEqual(t.waypoint_ahead(39.9), 0)

    def test_duplicate_waypoint_handled(self):
        t = square(duplicate=5)
        self.assertEqual(t.n_waypoints, 17)               # row kept, indices stable
        self.assertEqual(t.duplicate_indices, (5,))
        self.assertEqual(t.next_distinct(5), 7)
        self.assertTrue(np.all(np.isfinite(t.curvature())))
        self.assertAlmostEqual(t.heading_at(5), t.heading_at(6))
        t.project(t.waypoints[5])                         # no zero division

    def test_curvature_sign_and_magnitude(self):
        # Counter-clockwise circle of radius 5: curvature +0.2 everywhere.
        ang = np.linspace(0, 2 * math.pi, 60, endpoint=False)
        t = Track("circle", "unused", ".png", np.c_[5 * np.cos(ang), 5 * np.sin(ang)])
        np.testing.assert_allclose(t.curvature(), 0.2, rtol=1e-2)
        t_cw = Track("circle_cw", "unused", ".png", np.c_[5 * np.cos(-ang), 5 * np.sin(-ang)])
        self.assertTrue(np.all(t_cw.curvature() < 0))

    def test_spawn_pose(self):
        t = square()
        x, y, theta = t.spawn_pose(4)
        self.assertEqual((x, y), (10.0, 0.0))
        self.assertAlmostEqual(theta, math.pi / 2)
        with self.assertRaises(IndexError):
            t.spawn_pose(16)

    def test_segment_ranges(self):
        t = square()
        r = t.segment_ranges(4)
        self.assertEqual(r, ((0, 4), (4, 8), (8, 12), (12, 16)))
        with self.assertRaises(ValueError):
            t.segment_ranges(0)


class TestRegistry(unittest.TestCase):
    def test_cota_loads(self):
        t = load_track("cota")
        self.assertEqual(t.n_waypoints, 150)
        self.assertEqual(t.duplicate_indices, (70,))      # rows 70 and 71 share a position
        self.assertTrue(str(t.map_stem).endswith("maps/vegas"))
        self.assertGreater(t.length, 100.0)

    def test_unknown_track(self):
        with self.assertRaises(ValueError):
            load_track("monaco")

    def test_missing_files(self):
        with self.assertRaises(FileNotFoundError):
            load_track(TrackSpec(name="ghost", map_path="maps/ghost", waypoints_csv="maps/ghost.csv"))
        self.assertNotIn("ghost", TRACKS)


if __name__ == "__main__":
    unittest.main()
