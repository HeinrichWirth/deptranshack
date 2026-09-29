# Final realtime Linux pipeline

Implementation: `MVP/realtime_final/`. Results: `results_realtime_final/REPORT_REALTIME_FINAL.html`.
Frozen STEP1 / STEP2 / C4_SMOOTH / STEP6 and RAW seed adapter are imported unchanged.
No source LAS or annotation is changed. Classification dimensions are never exposed to inference.

Production selection: **2 independent solve workers × 4 candidate threads**, portable strict GCC,
12-frame immutable raw history, exact map-space AABB before current-coordinate filtering,
selective historical materialization. Each C4 solve starts from scratch. No persistent track.
`results_realtime_final/REALTIME_SELECTION_LOCK.json` records the selection before final held-out inference.

## Live API

```python
from pathlib import Path
from MVP.realtime_final.production import create_scheduler
from MVP.final_pipeline.frame_data import FrameData

scheduler = create_scheduler(Path('/data/recording'))
# Provide frames in order. Metadata must use the documented pose/coordinate convention.
# Only the next pose is used for bootstrap; its cloud is never included in that solve.
for frame_data in sensor_or_replay_producer:
    snapshot, raw_service_ms = scheduler.arrival(frame_data)
    # snapshot is None if no valid geometry; otherwise includes remaining_horizon.
    # Call poll between arrivals as well to publish completions without waiting for next scan.
    scheduler.poll()
scheduler.close()
```

`FrameData` carries raw map-coordinate XYZ, ring, point_index and immutable measurement arrays.
This API does not infer registration or poses; it consumes the same registered inputs as the frozen pipeline.
`Scheduler` owns engine instances; never call an engine concurrently. Shared frame arrays and C++ grids are
read-only. Snapshot leases extend object lifetimes through ring eviction; they do not copy a 12-frame cloud.
Only one newest pending request is retained. Publication requires increasing source frame and current gap generation.
Failed geometry does not erase a still-valid previous prediction. Time/pose gaps do erase it.

## Docker

Image: `deptrans-realtime:production`.

```powershell
docker run --rm --mount 'type=bind,source=C:\path\ANNOTATED,target=/data,readonly' --mount 'type=bind,source=C:\path\derived,target=/out' deptrans-realtime:production /data/recording --output /out --start 100 --count 10
```

Outputs include prediction NPZ, point provenance and summary JSON; source files are untouched.
The production wrapper defaults to 4 candidate threads, 2 solve workers.
Offline export processes selected frames sequentially. `--mode live-replay --count 200` exercises actual 10 Hz
arrivals and the bounded concurrent scheduler, saving replay metrics. It is a test producer, not a ROS node.

## Reproduce qualification

PowerShell `run_linux.ps1 -Arguments @('-m','MVP.realtime_final.bench','development')` runs the exact development comparison.
Other entry points: `linux_baseline`, `evaluate_linux`, `capture`, `trace`, `compare_trace`, `input_diagnosis`,
`branch_diagnosis`, `bench`, `point_sensitivity`, `live`, `qualification`, `metrics`, `report`, `test_contracts`.
Do not rerun `qualification lock` or final inference into the existing result root; they deliberately refuse overwrite.
Use a separate `REALTIME_OUTPUT` for a new experiment.

Native regeneration order (new stage only): `generate_native`, `optimize_native`, `counts_native`.
Then build `native/Dockerfile` as `deptrans-realtime:exact-gcc`, extract its Linux extension to `native/`,
and build this directory's Dockerfile as `deptrans-realtime:production`.
Compiler flags: C++17, -O3, no fast-math, no FMA contraction, no associative-math. Default pybind11 LTO is retained.
PGO, density thinning and CPU-specific instructions were not selected.

Trace runs are diagnostic and have profiling overhead; do not use their wall time as clean solve latency.
Point counts in ROI/candidate counters sum repeated queries, not globally unique raw rows.
Live replay RSS includes a producer-only preloaded future fixture. Inference sees only the causal snapshot.
See the report for geometry publish rate, distinct from 10 Hz input service rate, and finite-duration test limits.
