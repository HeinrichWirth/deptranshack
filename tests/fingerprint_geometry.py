"""Cross-build qualification on identical causal inputs; data supplied externally.

Run once with the accepted engine and once with the packaged engine. This is an
offline equality test, not a realtime speed measurement. No GT enters inference.
"""

import argparse, hashlib, json, sqlite3, sys, tempfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--engine", type=Path, required=True)
parser.add_argument("--previous", type=Path, required=True)
parser.add_argument("--bag", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
sys.path[:0] = [str(args.engine / "runtime"), str(args.engine / "native")]
from rt_common import *
from exact_math import install
from parallel_history import ParallelRing
from rail_refresh import refresh
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType
from MVP.realtime_final.memory_engine import MemoryEngine
from MVP.realtime_final.native_api import load
from MVP.final_pipeline.frame_data import FrameData
from cdr_cloud import decode


def fingerprint(value):
    h = hashlib.sha256()

    def add(v):
        if isinstance(v, np.ndarray):
            h.update(str(v.dtype).encode())
            h.update(str(v.shape).encode())
            h.update(np.ascontiguousarray(v).tobytes())
        elif isinstance(v, dict):
            for k in sorted(v):
                h.update(k.encode())
                add(v[k])
        else:
            h.update(json.dumps(v, sort_keys=True).encode())

    add(value)
    return h.hexdigest()


p = args.previous
logs = [json.loads(x) for x in (p / "registration.jsonl").read_text().splitlines()]
poses = {
    r["source_frame"]: r["pose"]["pose"]
    for r in logs
    if r["event"] == "REGISTRATION_COMPLETE" and r["pose"]["pose"] is not None
}
inputs = {
    r["source_frame"]: r
    for r in map(json.loads, (p / "input.jsonl").read_text().splitlines())
    if r["event"] == "INPUT"
}
available = sorted(int(f.stem) for f in (p / "batches").glob("*.json") if int(f.stem) + 1 in poses)
con = sqlite3.connect(args.bag.resolve().as_uri() + "?mode=ro", uri=True)
session = Path(tempfile.mkdtemp())
dump(session / "input.frames.json", dict(frames=[]))
pool = ThreadPoolExecutor(max_workers=4)
cache = {}
answers = []
for src in available:
    m = json.loads((p / "batches" / f"{src:06d}.json").read_text())
    records = [
        dict(
            file="UNAVAILABLE",
            header_time_ns=0,
            lidar_pose_in_folder=np.full((4, 4), np.nan).tolist(),
            pose_status="unavailable",
            pose_uses_future=False,
        )
        for _ in range(src + 2)
    ]
    for h in m["history"]:
        records[h["meta"]["source_frame"]] = record(h["meta"], h["pose"])
    records[src + 1] = record(inputs[src + 1], poses[src + 1])
    raws = []
    for h in m["history"]:
        k = h["meta"]["source_frame"]
        assert src - 11 <= k <= src
        if k not in cache:
            blob = con.execute(
                "SELECT data FROM messages WHERE id=?", (h["meta"]["message_id"],)
            ).fetchone()[0]
            _, points = decode(blob)
            cache[k] = np.column_stack([points[n] for n in ["x", "y", "z", "intensity"]])
        v = cache[k]
        ids = np.flatnonzero(finite(v)).astype(np.uint32)
        K = np.asarray(h["pose"])
        w = v[ids, :3].astype(float) @ K[:3, :3].T + K[:3, 3]
        raws.append(
            FrameData(
                w,
                np.full(len(ids), -1, np.int16),
                ids,
                MappingProxyType(dict(intensity_raw=v[ids, 3])),
                f"RAW_{k}",
                k,
                len(w) * 16,
            )
        )
    cache = {k: v for k, v in cache.items() if k >= src - 11}
    ring = ParallelRing(records)
    ring.arrive_many(raws, pool)
    e = MemoryEngine(8)
    e.start_run(session)
    e.records = records
    e.cache = ring.snapshot(src)
    e.native_frames = set(e.cache.frames)
    e.marcher = load().Marcher(8, 12, 0)
    e.marcher.options(True)
    e.marcher.attach(ring.store, sorted(e.cache.frames))
    install(e)
    result = e.process_frame(src)
    near = refresh(ring.entries[src][1])
    prediction = result["prediction"]
    c4 = result["c4"]
    checked = dict(
        prediction=(
            None
            if prediction is None
            else {k: v for k, v in prediction.items() if isinstance(v, np.ndarray)}
        ),
        provenance=result["provenance"],
        near_status=near["status"],
        near={k: v for k, v in near.items() if isinstance(v, np.ndarray)},
        c4_reason=c4.get("reason"),
        c4={
            k: c4[k]
            for k in ("curve", "local_uvw", "point_state", "point_anchor", "template_residual")
            if k in c4
        },
    )
    if len(result["cloud"]["keys"]):
        assert (result["cloud"]["keys"] >> np.uint64(32)).max() <= src
    answers.append(dict(source=src, sha256=fingerprint(checked)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(dict(checked=len(answers), available=len(available), rows=answers), indent=2)
    )
    if len(answers) % 25 == 0:
        print(args.engine.name, len(answers), "/", len(available), flush=True)
pool.shutdown()
con.close()
print("DONE", args.out, len(answers), flush=True)
