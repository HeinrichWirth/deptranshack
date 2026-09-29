"""Independent causal linear continuation of a completed three-rail geometry.

No points, registration, future frames or GT are inputs. The existing vertices
remain bit-identical. One common unit direction preserves terminal rail offsets.
"""
import numpy as np


def extend_curves(curves, length=5., tail=3., spacing=.1):
    if not 0 < length <= 20 or tail <= 0 or spacing <= 0:
        raise ValueError('Expected extension (0,20] m and positive tail/spacing')
    arrays = {k: np.asarray(curves[k], dtype=np.float64) for k in ('left','right','contact')}
    left, right = arrays['left'], arrays['right']
    n = len(left)
    if n < 3 or any(a.shape != (n,3) or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Three matched finite rail curves are required')
    center = (left + right) * .5
    steps = np.linalg.norm(np.diff(center, axis=0), axis=1)
    if np.any(steps <= 1e-8) or np.any(steps > 5):
        raise ValueError('Repeated points or discontinuity in source pair')
    s = np.r_[0., np.cumsum(steps)]
    first = max(0, int(np.searchsorted(s, s[-1] - tail)) - 1)
    x = s[first:] - s[-1]
    if len(x) < 3 or -x[0] < .5:
        raise ValueError('Insufficient terminal geometry')
    x -= x.mean()
    slope = np.sum(x[:,None] * (center[first:] - center[-1]), axis=0) / np.dot(x,x)
    norm = np.linalg.norm(slope)
    if norm <= 1e-8 or np.dot(slope, center[-1] - center[first]) <= 0:
        raise ValueError('Degenerate terminal direction')
    direction = slope / norm
    extra = np.linspace(length / int(np.ceil(length / spacing)), length, int(np.ceil(length / spacing)))
    result = {k: np.vstack((a, a[-1] + extra[:,None] * direction)) for k,a in arrays.items()}
    original_center = np.asarray(curves.get('center', center), dtype=np.float64)
    if original_center.shape != (n,3) or not np.isfinite(original_center).all():
        raise ValueError('Invalid center curve')
    result['center'] = np.vstack((original_center,original_center[-1] + extra[:,None]*direction))
    return result, dict(method='common_linear_tangent_last_3m', source_count=n,
                        source_length_m=float(s[-1]), added_length_m=float(length),
                        length_m=float(s[-1]+length), tail_span_m=float(s[-1]-s[first]),
                        direction=direction.tolist(), uses_future=False, supported=False)
