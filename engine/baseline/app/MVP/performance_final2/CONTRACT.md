# Fixed before measurements

Frozen pipeline files, the first performance stage and its results are immutable.
Float64 tolerance: absolute 1e-10, never enlarged after inspection. Discrete
support IDs, order, statuses, selected candidates, horizons and acceptance must
be identical. Prefer complete bitwise equality. All inference is RAW / GT-free;
only permitted past clouds and the approved pose direction adapter are used.

H4 preserves the actual residual: nearest-template distance / 0.015 for a
two-parameter translation. Cauchy loss stays inside SciPy. Reverse coverage
belongs to post-fit scoring, not to the residual. Adding it would change the
algorithm and is forbidden. H5 reproduces installed SciPy 1.18.1 two-point FD,
relative step 1e-3, zero-step fallback, bound sign/size adjustment, and actual
floating-point dx. Returned residual/Jacobian arrays own their memory.

Nearest-distance ties are harmless only for distance-only paths. Template NN
indices used by template_regions retain the original cKDTree implementation.
Small-N thresholds are selected by timings only. Native finite differences and
analytic derivatives are independently checked; a divergent variant is excluded.

Performance cohort: previous 32 starts. Final cohort: 40 uniform frame ordinals
from each of the previous five complete runs, selected before any new results.
No class labels or quality scores select these ordinals.

Async: one active solve plus one replaceable newest pending request. Published
geometry is immutable in map space. Requests and publications retain exact
source frame and gap generation. A result from before a pose gap cannot publish
into the new segment. Consumer service cost and geometry latency are distinct.
