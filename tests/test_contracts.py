"""Fast checks shipped in the image: freeze, guard, tracking and spatial grouping."""

import hashlib, json, os, sys, tempfile, unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["RAIL_DATA"] = tempfile.mkdtemp(prefix="rail-tests-")
sys.path[:0] = [str(ROOT), str(ROOT / "engine/runtime")]
import numpy as np
from extension_guard import guard
from exact_math import NumpyProxy
from obstacle_tracker import Tracker
from app.storage import JOBS, write_json, read_json
from app.grouping import group_tracks


class Contracts(unittest.TestCase):
    def test_frozen_sources(self):
        manifest = json.loads((ROOT / "engine/SOURCE_MANIFEST.json").read_text())
        for name, digest in manifest.items():
            self.assertEqual(
                hashlib.sha256((ROOT / "engine" / name).read_bytes()).hexdigest(), digest, name
            )

    def test_exact_cross(self):
        rng = np.random.default_rng(405)
        proxy = NumpyProxy()
        for _ in range(500):
            a, b = rng.normal(size=(2, 3))
            np.testing.assert_array_equal(proxy.cross(a, b), np.cross(a, b))

    def test_guard_preserves_near_and_rejects_unobserved_tail(self):
        y = np.arange(0, 28.01, 0.1)
        pair = np.stack(
            (
                np.column_stack((y * 0 - 0.792, -y, y * 0)),
                np.column_stack((y * 0 + 0.792, -y, y * 0)),
            ),
            1,
        )
        clipped, _ = guard(pair, np.empty((0, 3)), pair[80])
        np.testing.assert_array_equal(clipped, pair[:81])
        s = np.arange(8, 28.01, 0.1)
        points = np.vstack([np.column_stack((s * 0 + x, -s, s * 0)) for x in [-0.792, 0.792]])
        accepted, _ = guard(pair, points, pair[80])
        np.testing.assert_array_equal(accepted, pair)
        clipped, _ = guard(pair, points[points[:, 0] < 0], pair[80])
        self.assertEqual(len(clipped), 81)

    def test_same_source_is_not_second_confirmation(self):
        component = dict(
            voxels=np.array([[0.0, 0, 0], [0, 0.1, 0]]),
            points=np.array([[0.0, 0, 0], [0, 0.1, 0]]),
            center=np.array([0.0, 0.05, 0]),
            indices=np.array([0, 1]),
        )
        tracker = Tracker()
        tracker.update(10, 1.0, np.eye(4), 0, [component])
        tracker.update(10, 1.0, np.eye(4), 0, [component])
        self.assertIsNone(tracker.tracks[0]["confirmed_frame"])
        tracker.update(11, 1.1, np.eye(4), 0, [component])
        self.assertEqual(tracker.tracks[0]["confirmed_frame"], 11)

    def test_spatial_groups_do_not_chain_or_cross_pose_sessions(self):
        def observation(source, x):
            return dict(
                source_frame=source,
                world_center=[x, 0, 0],
                voxel_count=3,
                count=4,
                nearest_range_m=8.0,
                width_m=0.2,
                height_m=0.4,
                bbox_area_m2=0.08,
            )

        tracks = [
            dict(id=1, observations=[observation(0, 0), observation(1, 9)]),
            dict(id=2, observations=[observation(1, 18), observation(2, 18)]),
            dict(id=3, observations=[observation(3, 0)]),
        ]
        before = json.dumps(tracks, sort_keys=True)
        groups = group_tracks(tracks, {3: 1})
        self.assertEqual(len(groups), 3)
        self.assertEqual(groups[0]["frames"], [0, 1])
        self.assertEqual(groups[1]["frames"], [1, 2])
        self.assertEqual(groups[2]["status"], "SINGLE_FRAME")
        self.assertEqual(json.dumps(tracks, sort_keys=True), before)

    def test_two_pieces_in_same_frame_do_not_confirm_group(self):
        observation = dict(
            source_frame=10,
            world_center=[0, 0, 0],
            voxel_count=3,
            count=4,
            nearest_range_m=8.0,
            width_m=0.2,
            height_m=0.4,
            bbox_area_m2=0.08,
        )
        tracks = [
            dict(id=i, observations=[dict(observation, world_center=[i, 0, 0])]) for i in (3, 4)
        ]
        groups = group_tracks(tracks, {})
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["member_ids"], [3, 4])
        self.assertEqual(groups[0]["status"], "SINGLE_FRAME")


if __name__ == "__main__":
    unittest.main()
