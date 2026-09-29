"""STEP1_TO_STEP6_SEED_ADAPTER v1, authorized integration change.

No new detector: use frozen STEP1's accepted surface membership. Per-metre
head estimator and seed statistics reproduce the old estimator's mathematics,
but the evidence is RAW predicted surface support, never annotation labels.
Only current T points are used. Pair/thresholds come from frozen STEP1.
"""
import numpy as np
from track_geometry import sample_curve, offsets


def build_seed(xyz, source_rows, pair, model, initial, side):
    xyz = np.asarray(xyz)
    source_rows = np.asarray(source_rows, dtype=np.int64)
    vw = xyz[source_rows] @ initial
    pair = np.asarray(pair)
    which = np.argmin(np.linalg.norm(vw[:, None, 1:] - pair[None], axis=2), axis=1)
    observations = []
    used = []
    for k in range(8):
        heads, diagnostics, members = [], [], []
        for j in (0, 1):
            mask = (vw[:, 0] >= k) & (vw[:, 0] < k + 1) & (which == j)
            selected = vw[mask]
            unique = np.unique(np.round(selected, 5), axis=0)
            if len(unique) < 5:
                break
            upper = unique[unique[:, 2] >= np.quantile(unique[:, 2], .7)]
            if len(upper) < 3:
                break
            center = np.median(upper, axis=0)
            heads.append(center @ initial.T)
            diagnostics.append(dict(points=len(unique), surface_points=len(upper),
                spread=np.median(abs(upper-center), axis=0)*1.4826))
            # Every selected point in this section participates in robust quantile estimation.
            members.extend(source_rows[mask].tolist())
        if len(heads) != 2:
            continue
        u = float(np.mean(np.asarray(heads) @ initial[:, 0]))
        station = float(np.interp(u, model['C'] @ initial[:, 0], model['s']))
        center = sample_curve(model, station)
        if not np.isfinite(center).all():
            continue
        B = model['B'][int(np.argmin(abs(model['s'] - station)))]
        near = 1 if side > 0 else 0
        heads = np.array([heads[near], heads[1-near]])
        heads -= np.outer((heads-center) @ B[:, 0], B[:, 0])
        observations.append(dict(s=station, x=offsets(center, B, heads, side),
                                 heads=heads, cr=center, support=diagnostics))
        used.extend(members)
    if len(observations) < 2:
        return dict(status='SEED_TOO_SPARSE', sections=len(observations),
                    detector_pair=pair), np.unique(used).astype(np.int64)
    xx = np.array([o['x'] for o in observations]); ss = np.array([o['s'] for o in observations])
    xx[:, 0] = np.unwrap(xx[:, 0]); med = np.median(xx, axis=0)
    mad = 1.4826 * np.median(abs(xx-med), axis=0)
    covariance = np.cov(xx.T)/max(1, len(xx)/2) + np.diag(np.array([np.deg2rad(.1),.004,.004,.006,.004])**2)
    slope = np.polyfit(ss-ss.mean(), xx, 1)[0]
    return dict(status='AVAILABLE', x0=med, covariance=covariance, mad=mad, slope=slope,
                sections=len(xx), observations=observations, reference_s=float(np.median(ss)),
                seed_length=8, detector_pair=pair), np.unique(used).astype(np.int64)
