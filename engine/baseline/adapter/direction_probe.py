"""Evaluate the immediate next pose without advancing the retained 5 Hz chain."""
import copy
import time


def fork(registration):
    """Independent mutable state and independently bound timing closures."""
    cloned=type(registration)(registration.source_root)
    cloned.__dict__.update({k:copy.deepcopy(v) for k,v in registration.__dict__.items() if k not in ('fn','source_hashes')})
    return cloned


def evaluate_fork(registration,xyz,stamp):
    begin=time.perf_counter()
    pose=registration.step(xyz,registration.index+1,stamp,update_reference=False)
    finished=time.perf_counter()
    return pose,dict(registration.last_timing_ms),(finished-begin)*1000,finished


def probe(registration, xyz, stamp):
    begin=time.perf_counter()
    # Functions close over this instance; keep them bound to it. Numerical state
    # is isolated because _step appends support/history and may replace references.
    original=registration.__dict__.copy()
    state={k:copy.deepcopy(v) for k,v in original.items() if k not in ('fn','source_hashes')}
    registration.__dict__.update(state)
    try:
        # The final reference refresh cannot affect the returned pose and would
        # immediately be discarded with this probe. Keep ICP and gates exact.
        pose=registration.step(xyz,original['index']+1,stamp,update_reference=False)
        detail=dict(registration.last_timing_ms)
    finally:
        registration.__dict__.clear()
        registration.__dict__.update(original)
    return pose,detail,(time.perf_counter()-begin)*1000
