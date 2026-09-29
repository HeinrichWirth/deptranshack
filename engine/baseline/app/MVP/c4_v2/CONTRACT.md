# C4 V2 isolated experiment

Reference files are read-only. The code under `MVP/c4_v2` is not production
until a separate `FINAL_C4_V2.freeze` exists and the held-out gates pass.

The native core ports fixed C4: 8 m windows, 4 m advance, 12 causal frames,
three beam hypotheses, two tentative lookahead steps, P0 current-frame PCA,
unchanged template, exact NN, Cauchy residual and bounds. C4's frozen
`partial=subset` sets strong candidates to false; tentative promotion remains
necessary. The only candidate-fit change is projected native LM or GRID_LM.
The pool lasts for the engine lifetime. Marching and residual evaluations hold
no Python GIL; results are serialized after the native update completes.

The map holds original observation keys and accepted geometry, not synthetic
raw points. Only current-frame points initialize a repaired tail; C4 extension
uses the permitted contiguous 12-frame history. Stored old points remain map
evidence for smoothing and are never assigned age=0 for marching.

STEP6 seed observations remain the original RAW bootstrap observations in their
original 0–8 m coordinate system. No new current seed is fabricated. Frozen
C4_SMOOTH and STEP6 are invoked there, and results are transformed for display.
This permits accumulating error as distance from the original seed increases:
the experiment MUST measure it and cannot claim equivalence from timing alone.
Published sampled curves before the repair boundary retain their exact prior
values. A seam disagreement over 8 cm or 5 degrees causes full recovery.

Development uses five requested recordings. Independent benchmark uses the
other six; the old 200-start set overlaps development and is not a held-out
benchmark. The gates and parameter search are recorded in `plan.json` before
selection. GT reading is exclusively post-inference.

Timing from concurrent development jobs is for screening, not clean production
latency claims. Final timing must be measured in an otherwise idle process.
