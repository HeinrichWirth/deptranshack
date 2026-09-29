"""Prediction-independent validity check for derived rail-head reference pairs."""
import math

MIN_CENTER_DISTANCE_M = 1.48
MAX_CENTER_DISTANCE_M = 1.62


def validate_gt(pair, minimum=MIN_CENTER_DISTANCE_M, maximum=MAX_CENTER_DISTANCE_M):
    """Inclusive 2D centre distance; this does not repair or certify head centres."""
    if not 0 < minimum < maximum:
        raise ValueError('invalid centre-distance interval')
    result = dict(gt_geometry_valid=False, gt_geometry_reason='reference_unavailable',
                  gt_separation_2d_m=None, gt_separation_lateral_m=None)
    if pair is None:
        return result
    if len(pair) != 2 or any(len(p) != 2 for p in pair):
        raise ValueError('expected two (v,w) centres')
    if not all(math.isfinite(x) for p in pair for x in p):
        result['gt_geometry_reason'] = 'reference_nonfinite'
        return result
    dv, dw = pair[1][0] - pair[0][0], pair[1][1] - pair[0][1]
    distance = math.hypot(dv, dw)
    result.update(gt_separation_2d_m=distance, gt_separation_lateral_m=abs(dv))
    result['gt_geometry_reason'] = ('reference_too_narrow' if distance < minimum else
                                    'reference_too_wide' if distance > maximum else 'passed')
    result['gt_geometry_valid'] = result['gt_geometry_reason'] == 'passed'
    return result


def annotate_record(row):
    pair = [[row[f'gt_{side}_{axis}'] for axis in ('v', 'w')] for side in ('left', 'right')] if row['gt_valid'] else None
    result = dict(row, **validate_gt(pair))
    result['evaluation_status'] = ('scored' if row['status'] == 'ok' else 'refused_with_usable_reference') if result['gt_geometry_valid'] else (
        'unscored_rejected_reference' if row['gt_valid'] else 'unscored_missing_reference')
    result['checked_pair_error'] = row.get('pair_error') if result['gt_geometry_valid'] else None
    return result


def evaluation_view(row):
    """Temporary metric view. Saved raw coordinates and errors stay unchanged."""
    view = dict(row, gt_valid=row['gt_geometry_valid'])
    if not row['gt_geometry_valid']:
        for field in list(view):
            if field.endswith('_error') or field.startswith('both_within_'):
                view[field] = None
    return view
