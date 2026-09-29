# C4 performance sprint 2

Isolated equivalent implementation and diagnostics. Frozen dependencies remain
in their original folders; no source LAS or former stage is modified.

Production recommendation: `native_batch_spatial` (internal benchmark alias
`batch`), one worker. It keeps the frozen SciPy solver with native exact residual
and finite-difference Jacobian. Batching still invokes SciPy/Python per seed;
this is not a complete C++ solver. Experimental analytic/native solvers are
excluded. See `results_performance_final2/REPORT_C4_OPTIMIZATION_FINAL.html`.

## Runtime

Run from repository root with the documented Python runtime and existing
dependencies; BLAS/OMP/MKL thread limits are set by the imported pipeline.

```powershell
$py = 'C:\Users\heinrich.wirth\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py -B -m MVP.performance_final2.cli --run 'PATH_TO_ONE_RECORDING' --output 'NEW_EMPTY_OUTPUT' --backend native_batch_spatial --workers 1 --paced
```

`--paced` replays original timestamps with a genuine background GeometryWorker.
Without it, ingress runs at raw read/transform speed; that mode is not a 10 Hz
simulation. The consumer reads every raw cloud and transforms published geometry
to current sensor coordinates. This is an integration/profiling consumer, not an
obstacle detector. The CLI saves per-frame service/state metadata and worker
events in `RUN_COMPLETE.json`. It does not rewrite LAS. Call
`GeometryWorker.on_frame(index)` to receive actual prediction arrays in code.

One active solve + at most one newest pending. Pending replacement is explicit.
0.5 m trigger uses the last successful geometry source pose. Gap generations
invalidate old publications. Pose T+1 may arrive before solving T; no cloud >T
is used. There is no new age cutoff. Users of geometry must inspect source age,
distance and remaining horizon; state existence is not a quality guarantee.

The native extension must import and pass its startup arithmetic self-check;
failure is explicit. `python`, `native_spatial`, `fd`, `batch`, `grid`, `soa`
remain available as explicit comparison backends. A backend named `python`
selects the unchanged original numerical implementation.

## Reproduction

Do not run latency benchmarks simultaneously. Outputs below overwrite only the
new sprint's artifacts; use a copy of the stage for a new independent audit.

```powershell
& MVP/performance_final2/build_native.ps1
& $py -B -m MVP.performance_final2.baseline
& $py -B -m MVP.performance_final2.micro
& $py -B -m MVP.performance_final2.extra_micro
& $py -B -m MVP.performance_final2.suite
& $py -B -m MVP.performance_final2.paired_benchmark
& $py -B -m MVP.performance_final2.final_equivalence
& $py -B -m MVP.performance_final2.quality_sanity
& $py -B -m MVP.performance_final2.solver_audit
& $py -B -m MVP.performance_final2.audit
& $py -B -m MVP.performance_final2.spatial_micro
& $py -B -m MVP.performance_final2.gil_benchmark
& $py -B -m MVP.performance_final2.thread_paired
& $py -B -m MVP.performance_final2.offline_throughput
& $py -B -m MVP.performance_final2.async_simulation --backend native_spatial
& $py -B -m MVP.performance_final2.async_simulation --backend batch
& $py -B -m MVP.performance_final2.async_simulation --backend batch --live
& $py -B -m unittest discover -s MVP/performance_final2/tests -v
& $py -B -m MVP.performance_final2.report
& $py -B -m MVP.performance_final2.plots
& $py -B -m MVP.performance_final2.finalize
```

`micro` measures candidate thresholds; final runtime reads the pinned
`runtime_config.json` (forward 256, reverse 1024), independent of report folders.
Changing thresholds is a new experiment, not reproducing the frozen final
configuration. `benchmark_cohort.json` contains both fixed cohorts.
`quality_sanity` requires the completed 200-start inference barrier, then reads
approximate class1 reference strictly for evaluation.

## Linux Docker

```powershell
& $py -B -m MVP.performance_final2.prepare_docker
docker build --progress plain -t deptrans-final-runtime:performance-v2 results_performance_final2/docker_context
```

The builder uses Python 3.12.14, GCC 12, CMake and pybind11 3.0.1; runtime extends
the previously frozen `deptrans-final-runtime:performance-v1`. Runtime contains
no compiler. Actual image IDs and code/dependency hashes are in the final manifest.
Do not copy the LAS dataset into the image. Test with these read-only mounts:

```powershell
docker run --rm --network none --mount 'type=bind,source=ABS_DATASET_ROOT,target=/data,readonly' --mount 'type=bind,source=ABS_RESULTS_PERFORMANCE_FINAL,target=/app/results_performance_final,readonly' --mount 'type=bind,source=ABS_RESULTS_PERFORMANCE_FINAL2/docker_portable,target=/app/results_performance_final2' --entrypoint python deptrans-final-runtime:performance-v2 -B -m MVP.performance_final2.docker_test
```

`prepare_docker` seeds the output directory with the cohort, selected backend and
thresholds. `--wait-for-timing` waits for an `ALLOW_TIMING` file after correctness
tests, so isolated timing can begin after other workloads exit. The diagnostic
image is built with `--build-arg HOST_OPTIMIZED=ON` and tested with
`--diagnostic-host`; it uses `-march=native` and must not be distributed as the
portable default. Platform equivalence compares each platform's original and
optimized pipeline; Windows/Linux hashes are not asserted equal across BLAS builds.

## Limitations

The 32-start paired cohort selects the best variant and is not an independent
performance holdout. The 200-start cohort checks correctness, not latency under
the parallel validation workload. Chronological ablations drifted; the report
keeps them and counterbalanced results separately. Async full runs use actual
measured solve wall times and source timestamps, but an independent consumer
resource model; a separate paced live replay measures actual thread contention.
No claim of obstacle-detection accuracy or new async geometric accuracy is made.
