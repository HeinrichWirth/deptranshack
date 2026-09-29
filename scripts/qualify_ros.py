"""Post-inference audit of a DDS run against the source bag, never inference input."""
import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "engine/baseline/adapter")]
from cdr_cloud import decode
from app.grouping import group_tracks

parser = argparse.ArgumentParser()
parser.add_argument("--job", required=True)
parser.add_argument("--db", required=True)
parser.add_argument("--dds-results", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
root = Path(args.job) / "result"
def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
inputs = [r for r in rows(root / "input.jsonl") if r["event"] == "INPUT"]
geometry = [r for r in rows(root / "geometry.jsonl") if r["event"] == "GEOMETRY_COMPLETE"]
original = {}
with sqlite3.connect(Path(args.db).resolve().as_uri() + "?mode=ro", uri=True) as db:
    topic = db.execute("SELECT id FROM topics WHERE type='sensor_msgs/msg/PointCloud2'").fetchone()[0]
    for i, (blob,) in enumerate(db.execute("SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp,id", (topic,))):
        header, _ = decode(blob)
        original[hashlib.sha256(blob).hexdigest()] = (i, header["header_time_ns"], header["frame_id"])
matched = []
for row in inputs:
    source, stamp, frame_id = original[row["payload_sha256"]]
    assert stamp == row["header_time_ns"] and frame_id == row["source_frame_id"]
    matched.append(source)
assert matched == sorted(set(matched))
causal = 0
for path in (root / "batches").glob("*.json"):
    batch = json.loads(path.read_text())
    source = batch["meta"]["source_frame"]
    assert batch["direction_source_frame"] == source + 1
    assert not batch["uses_future_clouds"]
    assert all(k <= source for k in batch["batch_sources"] + batch["c4_history_sources"])
    causal += 1
raw = json.loads((root / "tracks.json").read_text())["tracks"]
groups = group_tracks(raw, {r["source_frame"]: r["generation"] for r in geometry})
dds = rows(args.dds_results)
def stats(values):
    return dict(median=float(np.median(values)), p95=float(np.percentile(values, 95)), maximum=float(max(values))) if values else None
report = dict(
    job=Path(args.job).name, source_messages=len(original), received=len(inputs),
    cdr_exact_matches=len(matched), missing_source_indices=sorted(set(range(len(original))) - set(matched)),
    geometry_complete=len(geometry), causal_batches_checked=causal,
    confirmed_groups=sum(g["status"] == "CONFIRMED" for g in groups),
    algorithm_ms=stats([r["total_ms"]-r["export_ms"] for r in geometry]),
    stages_ms={key: stats([r[key] for r in geometry]) for key in (
        "prepare_ms", "solve_ms", "rail_refresh_ms", "path_policy_ms", "envelope_build_ms", "query_ms", "cluster_ms", "tracker_ms")},
    callback_to_send_ms=stats([r["transport"]["callback_to_send_ms"] for r in inputs]),
    dds_result_messages=len(dds), dds_positive_messages=sum(r.get("obstacle_detected") is True for r in dds),
    dds_unknown_messages=sum(r.get("assessment_available") is False for r in dds),
    first_confirmed=next((r for r in dds if r.get("obstacle_detected") is True), None),
    complete=json.loads((root / "COMPLETE.json").read_text()),
)
Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k:v for k,v in report.items() if k not in ("complete", "stages_ms", "first_confirmed", "missing_source_indices")}, indent=2))
