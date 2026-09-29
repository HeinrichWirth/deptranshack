"""Integration of the independently tested current-near / old-shape adapter."""
from rail_refresh import refresh as previous_refresh
from reanchor import reanchor


def refresh(points, prior=None, *, pose_available=True):
    result = previous_refresh(points, prior)
    if result['status'] == 'ACCEPTED' and prior is not None:
        tail = reanchor(prior, result['near_pair'], pose_available=pose_available)
        result['reanchor'] = {k: v for k, v in tail.items() if k not in ('pair', 'rotation', 'old_anchor', 'new_anchor')}
        result['timing_ms']['reanchor'] = tail['ms']
        result['timing_ms']['total'] += tail['ms']
        if tail['status'] == 'REANCHORED':
            result['corrected_pair'] = tail['pair']
            result['prior_status'] = 'REANCHORED_TO_CURRENT_RAILS'
            result['original_count'] = tail['near_count']
        else:
            result['corrected_pair']=None
            result['prior_status']=tail['status']
    return result
