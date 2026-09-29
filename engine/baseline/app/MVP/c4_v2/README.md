# C4 V2 — isolated final experiment

Production default remains **reference**, the frozen `native_batch_spatial`.
Persistent V2 failed development quality gates. Do not treat the existence of
the implementation, Docker image or selected configuration as production approval.
`FINAL_C4_V2.freeze` is deliberately absent.

Code: `MVP/c4_v2/`. Results: `results_c4_v2/`.
Open `results_c4_v2/REPORT_C4_V2_FINAL.html` and `gallery.html`.

## Build and run

From repository root, using the existing Python 3.12 environment:

```powershell
& MVP/c4_v2/build_native.ps1
$py = 'C:\Users\heinrich.wirth\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py -B -m MVP.c4_v2.cli --run 'PATH_TO_ONE_REGISTERED_RECORDING' --output 'NEW_EMPTY_RESULT_FOLDER' --c4-backend reference --paced
& $py -B -m MVP.c4_v2.cli --run 'PATH_TO_ONE_REGISTERED_RECORDING' --output 'ANOTHER_EMPTY_FOLDER' --c4-backend v2 --config results_c4_v2/selected_config.json --paced
```

`C4_BACKEND=reference|v2` provides the CLI default. Native import failures are
explicit errors. Source LAS and frozen source code are read-only. The CLI reads
raw frames, submits causal geometry work to one background worker, and keeps one
replaceable pending request. It does not rewrite LAS or implement obstacle search.

One-frame pose latency is inherited from the approved bootstrap: pose T+1 is
allowed, cloud T+1 is forbidden. Old map observations preserve source keys; they
are not reclassified as current evidence. History for extension is at most the
contiguous T−11…T sequence. See `CONTRACT.md`.

## Reproduction scripts

- `solver_study.py`: native LM caps and GRID_LM captured-fit comparison; persistent thread-pool scaling.
- `development_probe.py`: full native marching without persistence on the old 200 RAW starts.
- `screen_persistence.py`: causal parameter screen and separate old-SciPy persistence control.
- `run_sequences.py`: full chronological development / held-out benchmark; first frame starts without a map.
- `fill_reference.py`: independent RAW reference inference on the same held-out starts.
- `evaluate.py`: GT only after saved inference markers; common transverse planes.
- `clean_timing.py`: serial clean full / persistent timing and cProfile.
- `stress.py`: isolated 1/3/5/10 cm tail perturbation experiment.
- `async_audit.py`: full-run latest-value event replay with measured solve durations.
- `integrity.py`: check frozen hashes against prior freeze files.
- `report.py`, `gallery.py`, `finalize.py`: derived report, figures and manifest.

All experiment outputs stay in `results_c4_v2`. To rerun a changed experiment,
use a new `--name`; do not mix a new algorithm with existing prediction files.
The benchmark config was locked before held-out inference in `SELECTION_LOCK.json`.
The old 200-start cohort belongs to development, not held-out testing.

## Docker

```powershell
& $py -B -m MVP.c4_v2.prepare_docker
docker build -t deptrans-c4-v2:experimental results_c4_v2/docker_context
docker run --rm --entrypoint python deptrans-c4-v2:experimental -B -m unittest MVP.c4_v2.tests.test_contracts -v
```

Mount the dataset read-only and a separate writable output directory. The runtime
is based on `deptrans-final-runtime:performance-v2`; no dataset is embedded.
The C++ compiler exists only in the build stage. Linux smoke tests are distinct
from the Windows full quality/timing study; the report does not claim a full
Linux accuracy benchmark. No default switch to V2 is made after a failed gate.

## Known limits / why no production freeze

The unchanged RAW seed adapter is run at bootstrap, not on every valid update.
Its real observations remain in their original 0–8 m coordinate system. Frozen
C4_SMOOTH / STEP6 then operate in that system as the persistent map grows, and
published values are transformed to the current lidar. This preserves the model
contract but exposes accumulation of offset / map errors. Reuse is fast; repairs
still incur frozen downstream costs. A longer persistent horizon alone is not a
quality improvement. See the failures and matching-plane metrics before use.
