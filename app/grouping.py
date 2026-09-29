"""Coarse causal object grouping in 10 m world-coordinate neighbourhoods.

The first observation fixes the anchor. Anchors never move and groups are not
linked transitively, so a chain of nearby points cannot swallow an entire run.
This groups detections for reporting; it does not modify rails or envelope hits.
"""

import math

RADIUS_M = 10.0


def group_tracks(tracks, generations, radius=RADIUS_M):
    events = sorted(
        (
            (o["source_frame"], t["id"], i, o)
            for t in tracks
            for i, o in enumerate(t["observations"])
        ),
        key=lambda row: row[:3],
    )
    groups = []
    for source, raw_id, _, observation in events:
        point = observation["world_center"]
        if len(point) != 3 or not all(math.isfinite(x) for x in point):
            raise ValueError("Object observation has no finite world position")
        generation = generations.get(source, 0)
        nearby = [
            (math.dist(point, g["anchor_world"]), g["id"], g)
            for g in groups
            if g["generation"] == generation
        ]
        nearby = [candidate for candidate in nearby if candidate[0] <= radius]
        if nearby:
            group = min(nearby, key=lambda row: row[:2])[2]
        else:
            group = dict(
                id=len(groups) + 1,
                generation=generation,
                anchor_world=list(point),
                members=set(),
                observations=[],
            )
            groups.append(group)
        group["members"].add(raw_id)
        group["observations"].append(observation)
    result = []
    for group in groups:
        observations = group["observations"]
        sources = sorted({o["source_frame"] for o in observations})
        best = max(observations, key=lambda o: (o["voxel_count"], o["count"]))
        result.append(
            dict(
                id=group["id"],
                member_ids=sorted(group["members"]),
                automatic_group=True,
                grouping_radius_m=radius,
                anchor_world=group["anchor_world"],
                generation=group["generation"],
                status="CONFIRMED" if len(sources) >= 2 else "SINGLE_FRAME",
                frames=sources,
                observations=observations,
                first_frame=sources[0],
                last_frame=sources[-1],
                nearest_range_m=min(o["nearest_range_m"] for o in observations),
                first_distance_m=observations[0]["nearest_range_m"],
                width_m=best["width_m"],
                height_m=best["height_m"],
                bbox_area_m2=best["bbox_area_m2"],
                size_frame=best["source_frame"],
            )
        )
    return result
