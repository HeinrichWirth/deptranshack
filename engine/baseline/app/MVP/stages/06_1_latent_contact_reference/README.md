# STEP 6.1 — Latent contact-rail reference curve

Research only. Frozen STEP 6 and C4 remain read-only. Results are written to
`../../../results_latent_contact_reference/` from the repository root path resolver.

The complete Russian request is in `REQUEST_RU.txt`. `baseline_freeze.json` pins
the existing code, configuration, template, C4 input index and expected results.
The mandatory reproduction passed for all 1,530 starts before new model work.

Scripts (run with the repository-compatible Python runtime and `-B`):

1. `src/prepare_reproduction.py` — pin original dependencies and expectations.
2. `src/reproduce_baseline.py --workers 6` — recompute old geometry, seed, rails,
   covariance and metrics; stops on any mismatch.
3. `src/prepare_protocol.py` — development split, 40 stratified starts, candidates.
4. `tests/test_latent.py` — causal input and geometric regression checks.
5. `src/run_phase.py phase_a --workers 6` — candidate screening; four mandated
   old benchmark failures are diagnostic only and excluded from selection.
6. `src/select_models.py phase_a` — development-only successive halving.
7. `src/run_phase.py phase_b --workers 6` — all 108 development starts.
8. `src/select_models.py phase_b` — leave-one-run-out ranking and top three.
9. `src/stress_tests.py` — partial visibility, source dropout, registration,
   corrupted/duplicate anchors, and observation gaps.
10. `src/freeze_finalists.py` — range-conditional CR covariance floors and final
    research freeze. Never call after benchmark inference has completed.
11. `src/run_phase.py phase_d --workers 6` — calibrated finalists on development.
12. `src/run_phase.py phase_e --workers 6` — the one reused benchmark evaluation.
13. `src/analyze_final.py` — common-plane metrics, matched oracle gap, failure corpus.
14. `src/profile_anchor_diagnostic.py`, `src/remaining_roll.py`,
    `src/curvature_error_eval.py`, `src/uncertainty_loro_full.py`,
    `src/geometry_audit.py` — isolated post-inference explanations.
15. `src/advanced_anchor_stress.py` — raw-point T1/EM response to synthetic anchor
    corruption; inputs perturbed in memory only.
16. `src/extra_diagnostics.py` — exact source sensor timestamps, class2 diagnostic
    visibility after prediction, and cold profile runtime.
17. `src/write_parquet.py` — requested station table plus round-trip validation.
18. `src/export_las.py`, `src/build_gallery.py`, `src/build_report.py` — delivery.
19. `src/verify_delivery.py` — hashes, inference barriers, C0 identity, original LAS
    fields, local links/JS, required files, and final manifest.

`src/finish_development.py` runs development gates in sequence and stops before
benchmark. Matplotlib is read from the existing `results_rail2d/_packages` runtime.
Parquet serialization reads the installed Arrow package under
`MVP/stages/06_pipeline_performance/vendor`; that directory is not modified.

Result: verdict C. The two severe interpolation failures improve substantially;
the matched 75–100 m oracle gap does not close. See the full report for model
approximations, one unavailable endpoint plane, endpoint station-assignment
limitations, covariance limitations, regressions and retained failures.

Every phase saves all predictions and physical checks before opening future
references. `latent_inference.py` reads only the whitelisted numeric C4 geometry
and the already-provided initial head sections. It never reads classification
arrays. Future class1/class2 proxies belong exclusively to evaluation scripts.

Physical priors: horizontal curvature 0.01/m from
`tools/trajectory_predictor.py::KMAX`, also explicitly requested in this study.
`GMAX=0.045` is an existing engineering heuristic, not enforced as a normative
raw-sensor-Z grade limit. The vertical smoothness scale is derived from frozen
development C4 curves and recorded in the protocol. Saved registration world-up
is an estimate, not measured gravity.

Important distinctions:

- A partial-profile repeatability error is relative to its full observed profile,
  not physical ground truth.
- Future class2 and class1 are imperfect empirical proxies.
- A physically smooth curve may still be wrong. All methods retain physical flags
  and failures, including solver failures and unavailable reference segments.
- PCHIP/Hermite use a declared station and plan/vertical frame; their component
  shape preservation is coordinate dependent.
- The empirical canonical profile is not a full factory drawing.
- The old benchmark is reused research data. No fresh blind recording is consumed.
