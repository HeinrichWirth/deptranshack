"""ROS ingress framing, live status and confirmation semantics."""

import json
import socket
import struct
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from app.live_protocol import receive_cloud, receive_json, send_json, MAX_CDR_BYTES
from app.results import live_result
from app.results import objects


class LiveTransport(unittest.TestCase):
    def test_unfinished_batch_cannot_confirm_published_object(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "result").mkdir()
            observation = dict(source_frame=1, world_center=[0, 0, 0], nearest_range_m=70,
                               voxel_count=5, count=8, width_m=1, height_m=1, bbox_area_m2=1)
            snapshot = dict(tracks=[dict(id=1, observations=[observation, dict(observation, source_frame=2)])],
                            generations={"1": 0, "2": 0}, source_frame=2)
            (root / "result/live_tracks.json").write_text(json.dumps(snapshot))
            with patch("app.results.job_path", return_value=root), patch("app.results.frames", return_value=[dict(source_frame=1, generation=0)]):
                group = objects("test")["tracks"][0]
                self.assertEqual(group["status"], "SINGLE_FRAME")
                self.assertEqual(group["frames"], [1])
            with patch("app.results.job_path", return_value=root), patch("app.results.frames", return_value=[dict(source_frame=1, generation=0), dict(source_frame=2, generation=0)]):
                group = objects("test", through_source=1)["tracks"][0]
                self.assertEqual(group["status"], "SINGLE_FRAME")

    def test_exact_cdr_payload(self):
        a, b = socket.socketpair()
        with a, b:
            blob = b"\0\1\0\0exact CDR bytes\xff"
            send_json(a, dict(sequence=3, bytes=len(blob)))
            a.sendall(blob)
            meta, actual = receive_cloud(b)
            self.assertEqual(meta["sequence"], 3)
            self.assertEqual(actual, blob)

    def test_oversize_rejected_before_reading_payload(self):
        a, b = socket.socketpair()
        with a, b:
            send_json(a, dict(bytes=MAX_CDR_BYTES + 1))
            with self.assertRaises(ValueError):
                receive_cloud(b)

    def test_disconnect_inside_packet(self):
        a, b = socket.socketpair()
        a.sendall(struct.pack("!I", 20) + b"{}")
        a.close()
        with b, self.assertRaises(EOFError):
            receive_json(b)

    def test_stale_or_unconfirmed_is_not_current_obstacle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "result/batches").mkdir(parents=True)
            (root / "job.json").write_text(json.dumps(dict(status="LIVE_RUNNING")))
            (root / "result/live_tracks.json").write_text(json.dumps(dict(source_frame=3)))
            (root / "result/batches/000003.json").write_text(json.dumps(dict(meta=dict(
                header_time_ns=1000000000, source_frame_id="lidar"))))
            row = dict(source_frame=3, status="INTRUSION", clock=time.perf_counter(),
                       total_ms=220, export_ms=20)
            group = dict(id=1, first_frame=1, first_distance_m=70, width_m=1, height_m=1,
                         bbox_area_m2=1, status="CONFIRMED", observations=[dict(source_frame=2, nearest_range_m=60)])
            with patch("app.results.job_path", return_value=root), patch("app.results.frames", return_value=[row]), patch("app.results.objects", return_value=dict(tracks=[group])):
                self.assertFalse(live_result("test")["obstacle_detected"])
                group["observations"].append(dict(source_frame=3, nearest_range_m=55))
                answer = live_result("test")
                self.assertTrue(answer["obstacle_detected"])
                self.assertEqual(answer["distance_m"], 55)
                self.assertEqual(answer["objects"][0]["first_distance_m"], 70)
                row["clock"] -= 2
                answer = live_result("test")
                self.assertIsNone(answer["obstacle_detected"])
                self.assertFalse(answer["assessment_available"])
                self.assertIsNone(answer["distance_m"])
