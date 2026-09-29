# HACKATHON_FINAL_CANDIDATE

Frozen C4_SMOOTH + STEP6 B4_SPLINE_50__O1. No STEP6.3 correction is active.

`src/final_geometry.py` is a numeric integration API, not a new LAS detector.
Inputs: the frozen C4 causal geometry/anchors, its source-frame indices, and the
provided STEP1 0–8 m rail-head seed. No labels, future clouds or dataset identities.
The existing STEP5.2 C4 source remains upstream and unchanged.

Call `predict_geometry(seed, geometry, side, source_frames, current_frame)`.
The returned prediction includes contact reference, near/far rails, states and
covariances. Caller supplies NumPy/SciPy; this repository has them in `.runtime`.
The source files here are the required mathematical functions extracted from
frozen stages, without their experiment runners, evaluation or GT readers.

Uncertainty is an estimate. Seed cross-section errors and long unsupported
regions remain limitations. This is an integration candidate, not a guarantee
that all railway geometry has been recovered correctly.
