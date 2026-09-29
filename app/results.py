"""Read saved causal results. Display grouping never enters the prediction engine."""

import json, struct
import numpy as np
from .storage import job_path, read_json, read_log
from .grouping import group_tracks


def frames(job_id):
    return [
        r
        for r in read_log(job_path(job_id) / "result/geometry.jsonl")
        if r["event"] == "GEOMETRY_COMPLETE"
    ]


def objects(job_id):
    path = job_path(job_id)
    raw = read_json(path / "result/tracks.json", {"tracks": []})["tracks"]
    generations = {r["source_frame"]: r.get("generation", 0) for r in frames(job_id)}
    return dict(
        tracks=group_tracks(raw, generations),
        raw_tracks=len(raw),
        grouping="fixed_world_anchor_10m",
    )


def load_frame(job_id, source, object_id=0):
    path = job_path(job_id) / "result"
    row = next((r for r in frames(job_id) if r["source_frame"] == source), None)
    if row is None:
        raise ValueError("Результат этого кадра ещё не готов")
    with np.load(path / "batches" / f"{source:06d}.npz") as batch:
        xyz = batch["xyz"].copy()
    pair = np.empty((0, 2, 3))
    contact = np.empty((0, 3))
    prior = pair.copy()
    layers = np.zeros(len(xyz), np.float32)
    forecast = path / "rail_refresh" / f"{source:06d}.npz"
    if forecast.exists():
        with np.load(forecast) as data:
            pair = data["pair"]
            contact = data["contact"]
            prior = data["prior_pair"]
            layers[data["support_indices"]] = 1
    if object_id:
        selected = next((t for t in objects(job_id)["tracks"] if t["id"] == object_id), None)
        if selected:
            for observation in selected["observations"]:
                if observation["source_frame"] == source:
                    layers[observation["source_indices"]] = 4
    return row, xyz, layers, pair, contact, prior


def binary_frame(job_id, source, object_id):
    row, xyz, layers, pair, contact, prior = load_frame(job_id, source, object_id)
    curves = dict(
        left=pair[:, 0].tolist(),
        right=pair[:, 1].tolist(),
        center=pair.mean(1).tolist(),
        contact=contact.tolist(),
    )
    arrays = [np.column_stack((xyz, layers))]
    for name, layer in [("left", 21), ("right", 22), ("contact", 20), ("center", 23)]:
        if curves[name]:
            arrays.append(np.column_stack((curves[name], np.full(len(curves[name]), layer))))
    points = np.vstack(arrays).astype("<f4")
    guard = row.get("extension_guard") or {}
    meta = dict(
        source_frame_index=source,
        source_clouds=row["batch_sources"],
        real_points=len(xyz),
        synthetic_points=len(points) - len(xyz),
        point_count=len(points),
        available=len(pair) >= 2,
        status=row["status"],
        reason="",
        up=[0, 0, 1],
        run=job_id,
        curves=curves,
        comparison_curves=dict(left=prior[:, 0].tolist(), right=prior[:, 1].tolist(), contact=[]),
        diagnostic_warning=f"Сохранённый результат кадра {source}. Разрешённый путь до {guard.get('accepted_end_m',8):.1f} м. Дальше — неизвестно. Пурпурным выбранный объект; белым путь до защиты.",
        cloud_fixed=True,
        prediction_recomputed=False,
        held_plane=False,
        recorded_far=row,
        coordinate_frame="current T LiDAR XYZ metres",
        point_fields=["x", "y", "z", "display_layer"],
    )
    header = json.dumps(meta, ensure_ascii=False).encode()
    header += b" " * ((-len(header)) % 4)
    return struct.pack("<I", len(header)) + header + points.tobytes()


def slice_frame(job_id, source, distance, thickness=1.0):
    if not np.isfinite(distance) or distance < 0 or distance > 1000:
        raise ValueError("Расстояние должно быть конечным числом от 0 до 1000 м")
    row, xyz, layers, pair, contact, _ = load_frame(job_id, source)
    if len(pair) < 2:
        return dict(available=False, source=source, status="Рельсы в этом кадре не приняты")
    center = pair.mean(1)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(center, axis=0), axis=1))]
    if distance > arc[-1]:
        return dict(
            available=False,
            source=source,
            status="За пределами принятого прогноза",
            horizon_m=float(arc[-1]),
        )
    k = min(int(np.searchsorted(arc, distance, side="right") - 1), len(pair) - 2)
    k = max(k, 0)
    t = float(np.clip((distance - arc[k]) / max(arc[k + 1] - arc[k], 1e-9), 0, 1))
    heads = pair[k] * (1 - t) + pair[k + 1] * t
    origin = heads.mean(0)
    forward = center[k + 1] - center[k]
    forward /= np.linalg.norm(forward)
    right = heads[1] - heads[0]
    right -= forward * (right @ forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    if up[2] < 0:
        up = -up
    q = (xyz - origin) @ np.column_stack((right, forward, up))
    mask = (abs(q[:, 1]) <= thickness / 2) & (abs(q[:, 0]) <= 5) & (q[:, 2] >= -1) & (q[:, 2] <= 5)
    ids = np.flatnonzero(mask)
    points = q[ids][:, [0, 2]]
    hit = (abs(points[:, 0]) < 1.02) & (points[:, 1] > 0.10) & (points[:, 1] < 2.97)
    colors = np.where(hit, 3, np.where(layers[ids] == 1, 1, 0))
    rails = (heads - origin) @ np.column_stack((right, forward, up))
    cr = (
        (contact - origin) @ np.column_stack((right, forward, up))
        if len(contact)
        else np.empty((0, 3))
    )
    cr = cr[abs(cr[:, 1]) <= thickness / 2]
    # Display is sampled only after detection; counts and nearest range use all eligible points.
    display = (
        np.linspace(0, len(points) - 1, min(len(points), 20000), dtype=int)
        if len(points)
        else np.array([], int)
    )
    return dict(
        available=True,
        source=source,
        distance_m=distance,
        horizon_m=float(arc[-1]),
        points=np.column_stack((points[display], colors[display])).tolist(),
        rails=rails[:, [0, 2]].tolist(),
        contact=cr[:, [0, 2]].tolist(),
        slice_points=len(ids),
        intrusion_points=int(hit.sum()),
        nearest_m=float(np.linalg.norm(xyz[ids[hit]], axis=1).min()) if hit.any() else None,
        algorithm_ms=row["total_ms"] - row["export_ms"],
        status=row["status"],
        display_only=True,
    )
