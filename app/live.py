"""Live CDR ingress into the unchanged causal registration/geometry workers."""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import signal
import socket
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "engine"
sys.path[:0] = [str(ENGINE / "native"), str(ENGINE / "runtime")]
from rt_common import CloudRing, Latest, dump, log, log_open, np, sha
sys.path.insert(0, str(ENGINE / "runtime"))
from cdr_cloud import decode
from registration import recovery_worker
from heavy import registration_worker
from .live_worker import geometry_worker
from .live_protocol import receive_json, receive_cloud, send_json
from .storage import JOBS, read_json, write_json


def guarded(function, args, errors):
    try:
        function(*args)
    except BaseException:
        detail = traceback.format_exc()
        errors.put(detail)
        print(detail, flush=True)
        raise


def run_session(conn, shutdown):
    hello = receive_json(conn)
    if hello.get("protocol") != 1 or not isinstance(hello.get("topic"), str):
        raise ValueError("Unsupported ROS bridge protocol")
    job_id = uuid.uuid4().hex
    path = JOBS / job_id
    path.mkdir()
    out = path / "result"
    out.mkdir()
    state = dict(id=job_id, name="ROS 2 " + hello["topic"], status="LIVE_RUNNING",
                 created=time.time(), started=time.time(), transport="ROS 2 DDS",
                 topic=dict(id=0, name=hello["topic"]), topics=[])
    write_json(path / "job.json", state)
    cfg = read_json(ENGINE / "config.json")
    manifest = read_json(ENGINE / "SOURCE_MANIFEST.json")
    for name, expected in manifest.items():
        if sha(ENGINE / name) != expected:
            raise RuntimeError("Frozen source changed: " + name)
    source_hashes = {str(p.relative_to(ENGINE)): sha(p)
                     for folder in ("runtime", "native") for p in (ENGINE / folder).rglob("*")
                     if p.is_file() and p.suffix != ".pyc"}
    dump(out / "CODE_BEFORE.json", source_hashes)
    os.environ.update(COPY_REGISTRATION_ACCEL="parallel", COPY_VOXEL="packed", COPY_KD_WORKERS="1")
    ctx = mp.get_context("spawn")
    stop, errors = ctx.Event(), ctx.Queue()
    ring = CloudRing(ctx, cfg["ring_slots"], cfg["max_points"])
    boxes = {name: Latest(ctx, 32 * 1024 * 1024 if name.startswith("recovery") else 256 * 1024)
             for name in ("near", "registration", "pose", "geometry", "model_near", "model_far",
                          "recovery_in", "recovery_out", "retry", "near_reference")}
    work = [
        ("recovery", recovery_worker, (ring, boxes["recovery_in"], boxes["recovery_out"], stop)),
        ("registration", registration_worker, (ring, boxes["registration"], boxes["pose"], boxes["geometry"],
            boxes["recovery_in"], boxes["recovery_out"], boxes["retry"], stop)),
        ("geometry", geometry_worker, (ring, boxes["geometry"], [boxes["model_near"], boxes["model_far"]], boxes["retry"], stop)),
    ]
    processes = []
    started = time.perf_counter()
    received = 0
    digest = hashlib.sha256()
    last_stamp = first_stamp = last_frame = last_frame_id = None
    generation = 0
    last_slot = previous_target = -1
    logger = log_open(out, "input")
    failure = None
    try:
        for name, function, args in work:
            ready = ctx.Event()
            tail = (ready, str(out), False) if name == "recovery" else (
                (ready, str(out), cfg, False) if name == "registration" else (ready, str(out), cfg))
            process = ctx.Process(target=guarded, args=(function, args + tail, errors), name=name)
            process.start()
            processes.append((process, ready))
        for process, ready in processes:
            while not ready.wait(.05):
                if shutdown[0] or not process.is_alive() or time.perf_counter() - started > 60:
                    raise RuntimeError("Worker startup failed: " + process.name)
        send_json(conn, dict(ready=True, job=job_id))
        print("ROS_READY", job_id, hello["topic"], flush=True)
        begin = time.perf_counter()
        while not shutdown[0]:
            # A quiet lidar is not an empty tunnel. No clouds are invented on a timeout.
            import select
            if not select.select([conn], [], [], .1)[0]:
                if not errors.empty():
                    raise RuntimeError(errors.get())
                continue
            try:
                transport, blob = receive_cloud(conn)
            except EOFError:
                break
            entered = time.perf_counter()
            source = transport["sequence"]
            if not isinstance(source, int) or source < 0 or (last_frame is not None and source <= last_frame):
                raise ValueError("Non-increasing transport sequence")
            transit = time.monotonic() - float(transport["callback_monotonic"])
            if not math.isfinite(transit) or transit < -.01:
                raise ValueError("ROS bridge and engine must share the host monotonic clock")
            if transit > .1:
                log(logger, event="ROS_TRANSPORT_EXPIRED", source_frame=source, transport_ms=transit*1000)
                send_json(conn, dict(accepted=source, dropped=True))
                continue
            header, points = decode(blob)
            if not {"x", "y", "z", "intensity"} <= set(points.dtype.names):
                raise ValueError("PointCloud2 needs x, y, z, intensity")
            stamp = header["header_time_ns"]
            if stamp <= 0:
                raise ValueError("PointCloud2 header stamp must be nonzero")
            gap = last_frame is not None and (source != last_frame + 1 or
                not 0 < stamp - last_stamp < 150000000 or header["frame_id"] != last_frame_id)
            if gap:
                generation += 1
                previous_target = source - 1
                log(logger, event="ROS_INPUT_GAP", source_frame=source, previous_frame=last_frame,
                    stamp_delta_ns=stamp-last_stamp, generation=generation)
            if first_stamp is None or gap and stamp <= last_stamp:
                first_stamp, last_slot = stamp, -1
            if last_frame is None:
                previous_target = source - 1
            slot = (stamp - first_stamp) * cfg["full_hz"] // 1000000000
            target = slot > last_slot
            meta = dict(source_frame=source, message_id=source + 1, bag_time_ns=stamp,
                header_time_ns=stamp, deadline=entered-max(0, transit), generation=generation, points=len(points),
                source_frame_id=header["frame_id"], geometry_target=target,
                solve_slot=slot, batch_start=previous_target + 1)
            if target:
                last_slot, previous_target = slot, source
            decoded = time.perf_counter()
            ring.put(meta, np.column_stack([points[k] for k in ("x", "y", "z", "intensity")]))
            boxes["registration"].put(meta)
            published = time.perf_counter()
            digest.update(blob)
            received += 1
            log(logger, event="INPUT", **meta, decode_ms=(decoded-entered)*1000,
                dispatch_ms=(published-decoded)*1000, transport_ms=transit*1000, transport=transport,
                payload_sha256=hashlib.sha256(blob).hexdigest())
            last_frame, last_stamp, last_frame_id = source, stamp, header["frame_id"]
            send_json(conn, dict(accepted=source))
            if not errors.empty():
                raise RuntimeError(errors.get())
        time.sleep(.85)
    except Exception as error:
        failure = str(error)
    finally:
        stop.set()
        for process, _ in processes:
            process.join(4)
            if process.is_alive():
                process.terminate()
                process.join()
                failure = failure or "Worker watchdog terminated " + process.name
            elif process.exitcode:
                failure = failure or "Worker failed: " + process.name
        logger.close()
        if any(sha(ENGINE / name) != value for name, value in source_hashes.items()):
            failure = "Frozen runtime changed"
        dump(out / "COMPLETE.json", dict(pass_=failure is None, received=received, config=cfg,
            transport="ROS 2 DDS / bounded acknowledged CDR", payload_sha256=digest.hexdigest(),
            pending_replacements={name: box.replaced.value for name, box in boxes.items()},
            runtime_unchanged=True, no_saved_poses=True, no_annotation=True, error=failure))
        state.update(status="FAILED" if failure else "COMPLETE", finished=time.time(), error=failure)
        write_json(path / "job.json", state)
    print("ROS_FINISHED", job_id, received, failure or "OK", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8871)
    args = parser.parse_args()
    shutdown = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: shutdown.__setitem__(0, True))
    for path in JOBS.glob("*/job.json"):
        state = read_json(path)
        if state.get("status") == "LIVE_RUNNING":
            state.update(status="INTERRUPTED", error="Live receiver restarted")
            write_json(path, state)
    with socket.create_server((args.bind, args.port)) as server:
        server.settimeout(1)
        print("ROS_LISTENING", args.port, flush=True)
        while not shutdown[0]:
            try:
                conn, _ = server.accept()
            except socket.timeout:
                continue
            with conn:
                conn.settimeout(10)
                try:
                    run_session(conn, shutdown)
                except Exception as error:
                    print("ROS_SESSION_ERROR", repr(error), flush=True)


if __name__ == "__main__":
    main()
