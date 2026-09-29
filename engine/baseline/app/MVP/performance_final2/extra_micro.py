"""Isolated H7/H13 diagnostics, never selected as production solver."""
from . import ROOT,OUT
from .native_api import startup
from .micro import clock
import long_common as lc
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import time

def main():
    native=startup();rows=[];buffers=[]
    for path in sorted((ROOT/'results_performance_final/profiles').glob('fit_*.npz')):
        with np.load(path) as z:p=z['points'];t=z['template'];a=z['center'];gate=float(z['gate'])
        points=np.unique(p[np.all((p-a>=t.min(0)-.015-gate)&(p-a<=t.max(0)+.015+gate),axis=1)],axis=0)
        if len(points)<6:continue
        _,ix=np.unique(np.floor(points/.002).astype(np.int32),axis=0,return_index=True);fitp=points[ix]
        obj=native.Problem(fitp,t,a-gate,a+gate);obj.fun(a);j=obj.jac(a);before=clock(lambda:obj.jac(a));obj.reuse_workspace=True
        after=clock(lambda:obj.jac(a));np.testing.assert_array_equal(j,obj.jac(a))
        buffers.append(dict(case=path.stem,temporary_us=before*1e6,reused_us=after*1e6,bitwise=True,bytes=obj.bytes(),allocations_saved_per_jac=1))
        if len(rows)>=150:continue
        for seed_id,seed in enumerate([a,a+[gate*.7,0],a+[-gate*.7,0],a+[0,gate*.7],a+[0,-gate*.7]]):
            seed=np.clip(seed,a-gate+1e-9,a+gate-1e-9)
            tic=time.perf_counter();ref=least_squares(obj.fun,seed,jac=obj.jac,bounds=(a-gate,a+gate),loss='cauchy',f_scale=1.,diff_step=.001,max_nfev=24);refms=(time.perf_counter()-tic)*1000
            tic=time.perf_counter();out=native.experimental_fit(obj,seed);ms=(time.perf_counter()-tic)*1000
            delta=float(np.max(abs(np.array(out['x'])-ref.x)))
            rows.append(dict(case=path.stem,seed=seed_id,experimental_ms=ms,scipy_fd_ms=refms,iterations=out['iterations'],nfev=out['nfev'],
                scipy_nfev=ref.nfev,cost=out['cost'],scipy_cost=ref.cost,cost_delta=abs(out['cost']-ref.cost),max_parameter_delta=delta,
                passed=delta<=1e-10 and abs(out['cost']-ref.cost)<=1e-10,production=False))
    lc.csv_write(OUT/'experimental_solver.csv',rows);lc.csv_write(OUT/'allocation_workspace.csv',buffers)
    lc.save(OUT/'experimental_solver.json',dict(cases=len(rows),passed=sum(r['passed'] for r in rows),rejected=True,
        reason='Projected damped Gauss Newton changes TRF step/termination semantics and fails captured-fit equivalence; no full-C4 deployment',
        Ceres='Not installed; different solver semantics offer no equivalence guarantee. No new dependency added.'))
    print('EXTRA_MICRO',len(rows),sum(r['passed'] for r in rows),len(buffers),flush=True)

if __name__=='__main__':main()
