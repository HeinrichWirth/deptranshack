"""Contact-first integration policy. Frozen detectors are imported unchanged.

Availability is checked on the CURRENT raw 0..8 m cloud with frozen STEP2.
A visible/current accepted CR never triggers whole-tail rail reanchoring.
"""
import time
import numpy as np
from contact_rail_step2 import ContactRailDetector
from path_state import attach_near
from reanchor import reanchor


def probe(detector, refreshed):
    start = time.perf_counter()
    if 'head_pair' not in refreshed:
        return dict(status='UNKNOWN_NO_RUNNING_PAIR', available=False, support_count=0,
                    ms=(time.perf_counter()-start)*1000)
    found = detector.detect({'uvw': refreshed['contact_uvw']}, refreshed['head_pair'])
    return dict(status=found['status'], available=found['status'] in ('LEFT','RIGHT'),
                support_count=len(found['support_indices']), anchor_v=found['anchor_v'],
                anchor_w=found['anchor_w'], confidence=found['confidence'],
                reason=found['diagnostics']['reason'], ms=(time.perf_counter()-start)*1000)


def select(base_pair, fallback_pair, near, contact, *, base_fresh, base_side, pose_available=True):
    """Return geometry and explicit provenance; never use hit counts to choose it."""
    start = time.perf_counter()
    if contact['available']:
        if not base_fresh or base_side != contact['status']:
            return dict(pair=near.copy(), scope='NEAR_ONLY_WAITING_FOR_CONTACT_MODEL',
                        reanchor=None, fallback=False, original_count=len(near),
                        ms=(time.perf_counter()-start)*1000)
        # Current observations own the near prefix. A bounded join ends at 12 m;
        # the CR-derived far shape stays bit-for-bit unchanged beyond that join.
        pair, joined = attach_near(base_pair, near)
        scope = 'CONTACT_PRIMARY' if len(pair)>len(near) else 'NEAR_ONLY_CONTACT_JOIN_REJECTED'
        return dict(pair=pair, scope=scope, reanchor=None, fallback=False,
                    join_status=joined, original_count=len(near),
                    ms=(time.perf_counter()-start)*1000)
    if fallback_pair is not None and pose_available:
        tail = reanchor(fallback_pair, near, pose_available=pose_available)
        if tail['status']=='REANCHORED':
            return dict(pair=tail['pair'], scope='RAIL_FALLBACK_CONTACT_UNAVAILABLE',
                        reanchor={k:v for k,v in tail.items() if k not in ('pair','rotation','old_anchor','new_anchor')},
                        fallback=True, original_count=len(near),
                        ms=(time.perf_counter()-start)*1000)
    return dict(pair=near.copy(), scope='NEAR_ONLY_NO_USABLE_PRIOR', reanchor=None,
                fallback=False, original_count=len(near), ms=(time.perf_counter()-start)*1000)
