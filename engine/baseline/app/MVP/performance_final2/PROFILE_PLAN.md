# Audit interpretation

H1 initial identity-cache defect: cKDTree can copy a noncontiguous candidate
array. Keeping only the tree did not keep the key array alive, allowing Python
id reuse. The corrected cache retains `(original_points, tree)`. The failed
prototype artifacts are preserved; they are not an accepted NN approximation.

Native full residual and FD preserve the solver and Cauchy objective. The
analytic Jacobian fails captured-fit equivalence and is diagnostic only.
Batching uses a mechanical AST transformation of the frozen orchestration:
candidate_records suspends only at tt.fit; all beam parents prepare their
independent inputs, their seeds are evaluated in predetermined slots, and the
original scoring, duplicate removal, sorting and beam selection resume intact.
No candidates/history/search regions are pruned. Python callbacks into the
unchanged SciPy solver remain; one outer native batch call does not mean zero
Python/native transitions inside the solve. Counts will include those callbacks.

Async simulations replay original source timestamps using actual measured solve
wall times. They model an independent consumer resource; raw read/transform
service costs are measured, not zero. An additional wall-clock-paced background
thread test measures real consumer contention and arrival lateness. Neither
implements obstacle detection. Expired horizon remains explicit, with no new
unapproved cutoff or extrapolation.
