# ROS2 bag replay with causal stitching

Separate adapter for frozen `geometry_pipeline_v1.0.0`; no changes to `MVP/realtime_final` or original registration scripts. Input: `cloud_with_fake_obj_0.db3`, one `/lidar_points` PointCloud2 topic, 1510 messages. No ROS installation is needed to decode this SQLite/CDR recording.

## Sequence

1. Replay producer releases each message no earlier than its original bag timestamp. No cloud is preloaded before its scheduled arrival; only message IDs/timestamps are indexed for scheduling.
2. Decode CDR1 using its field offsets/endianness. The decoder is verified byte-for-byte against the existing `tools/cloud.py` / rosbags reader on 11 real messages.
3. Compute pose from scratch using current cloud and causal ICP state. Numerical functions come unchanged from `tools/register.py` and `tools/register_occlusion.py`; the existing sequential acceptance/recovery rules are retained. No saved map/pose file is mounted into inference.
4. Build immutable map-space FrameData from all finite nonzero source points. Preserve original point ordinal and float32 intensity; ring is missing (`-1`), per-point timestamp is not invented. Registration-only voxel reduction never thins detector input.
5. Append only the arrived pose to the scheduler's session metadata. Production factory keeps 2 workers × 4 native candidate threads. On arrival T+1, solve T sees cloud history at most T−11…T and pose metadata at most T+1. Assertions record every dispatched lease.
6. ICP refusal produces no accepted pose. Retire geometry session and do not expose the last curve across unknown motion. Begin a new session after recovery. Past missing poses remain missing; the old offline interpolation is deliberately absent. Last T cannot be dispatched without T+1.

The original registration time-gap and unresolved-occlusion limits remain explicit failures. This replay is not a new odometry design or tuning experiment.

## Timing

Registration producer, frozen geometry workers and diagnostic exporter execute concurrently. Raw frames are not dropped; if processing is slower than the source clock, backlog is reported relative to each original acquisition deadline. The adapter can therefore demonstrate that a configuration **fails** 10 Hz. One-frame data latency is not a promise of one-frame end-to-end latency.

SQLite read, CDR decoding, ICP, FrameData construction, arrival service, full solves and publication latency are separated. Windows Docker bind-mount file I/O is not representative of a live ROS subscription. Positive remaining horizon is measured relative to the latest *processed* frame; with backlog it is not a guarantee for the train's actual wall-clock position.

## Run

Use the existing `deptrans-realtime:production` image and mount only:

- Source bag directory to `/bag`, read-only.
- Existing `tools` directory to `/registration_source`, read-only (only the two registration source files are read).
- This adapter directory to `/adapter`, read-only.
- Frozen release directory to `/freeze`, read-only, for hash validation.
- A new result directory to `/out`, writable.

```sh
docker run --rm --network none --entrypoint python \
  --mount type=bind,source=ABS_BAG_DIRECTORY,target=/bag,readonly \
  --mount type=bind,source=ABS_TOOLS_DIRECTORY,target=/registration_source,readonly \
  --mount type=bind,source=ABS_ADAPTER_DIRECTORY,target=/adapter,readonly \
  --mount type=bind,source=ABS_FROZEN_CORE_DIRECTORY,target=/freeze,readonly \
  --mount type=bind,source=ABS_NEW_RESULT_DIRECTORY,target=/out \
  deptrans-realtime:production -B -u /adapter/replay.py \
  --db /bag/cloud_with_fake_obj_0.db3 --output /out/full \
  --registration-source /registration_source --freeze /freeze --count 1510
```

Adapter records are append-only in memory while workers receive bounded copied prefixes. There is no new geometry/threshold configuration. Transport integration is outside the immutable core.

## Output / postprocessing

`replay.py` writes every completed solve, observed publications, per-arrival and per-pose JSONL, and causality/timing summaries. It validates all 60 project runtime hashes before and after. `export_report.py` runs **after** inference, reads the source bag again, and creates full stitched LAS, separate coloured model LAS, sample sensor-frame overlays, charts and HTML/Markdown report. The full map excludes unavailable poses, zero placeholders and nonfinite points; original DB remains intact.

`verify_outputs.py` checks complete 1510-frame coverage, causality, solve-export hashes, exact source IDs/intensity and map-coordinate agreement to LAS quantization, plus frozen source hashes. Neither report nor saved poses are an inference dependency. These scripts do not implement ROS publishers/subscribers or TF transport.
