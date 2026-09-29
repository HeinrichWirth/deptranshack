"""First valid future pose at least 0.50 m from the fixed source frame.

Only poses/timestamps enter here. No future point clouds, accumulated path length,
interpolation through gaps, or sensor-heading fallback.
"""
import numpy as np
from rail2d_detector import basis


class MotionSelector:
    def __init__(self, meta, minimum_distance=.50):
        self.minimum_distance = float(minimum_distance)
        if not np.isfinite(self.minimum_distance) or self.minimum_distance < .01:
            raise ValueError('minimum_distance must be finite and >= 0.01 m')
        self.poses = np.asarray([r['lidar_pose_in_folder'] for r in meta], dtype=float)
        self.positions = self.poses[:, :3, 3]
        self.times = np.array([int(r['header_time_ns']) for r in meta], dtype=np.int64)
        self.valid = np.array([r.get('pose_status') in ('ok', 'origin') and
            not r.get('pose_uses_future', False) for r in meta]) & np.isfinite(self.poses).all(axis=(1, 2))
        dt = np.diff(self.times)/1e9
        steps = np.linalg.norm(np.diff(self.positions, axis=0), axis=1)
        self.edges = (dt > 0) & (dt <= .3) & (steps <= 5) & (steps/np.maximum(dt, 1e-9) <= 50)
        self.edges &= self.valid[:-1] & self.valid[1:]
        self.ends = np.arange(len(meta))
        for i in range(len(meta)-2, -1, -1):
            if self.edges[i]: self.ends[i] = self.ends[i+1]

    def select(self, i):
        info = dict(anchor_frame=None, wait_frames=None, latency_s=None,
                    displacement_m=None, minimum_displacement_m=self.minimum_distance,
                    reason='', pose_only_lookahead=True)
        if not self.valid[i]:
            info['reason'] = 'unreliable_or_future_derived_pose'
            return None, info
        end = self.ends[i]
        distances = np.linalg.norm(self.positions[i+1:end+1]-self.positions[i], axis=1)
        crossed = np.flatnonzero(distances >= self.minimum_distance)
        if not len(crossed):
            info['reason'] = 'insufficient_motion_before_end' if end == len(self.poses)-1 else 'insufficient_motion_before_gap'
            return None, info
        j = i+1+int(crossed[0])
        direction = self.poses[i, :3, :3].T @ (self.positions[j]-self.positions[i])
        info.update(anchor_frame=j, wait_frames=j-i, latency_s=float((self.times[j]-self.times[i])/1e9),
                    displacement_m=float(distances[crossed[0]]))
        try:
            axes = basis(direction)
        except ValueError as e:
            info['reason'] = str(e)
            return None, info
        info['sensor_axis_angle_deg'] = float(np.degrees(np.arccos(np.clip(np.dot(axes[:, 0], [0, -1, 0]), -1, 1))))
        return axes, info
