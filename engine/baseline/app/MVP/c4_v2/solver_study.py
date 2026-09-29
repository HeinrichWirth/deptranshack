"""Captured development fits: solver choice, not a repeat NN micro-audit."""
from . import ROOT,OUT
from .native_api import startup
import long_common as lc
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import time


def main():
    native=startup(); rows=[];scaling=[];fixtures=[]
    for file in sorted((ROOT/'results_performance_final/profiles').glob('fit_*.npz')):
        with np.load(file) as z:c={k:z[k] for k in z.files}
        t=c['template'];center=c['center'];gate=float(c['gate']);fixed=bool(c['fixed'])
        lo=t.min(0)-.015;hi=t.max(0)+.015;p=c['points'];p=np.unique(p[np.all((p-center>=lo-gate)&(p-center<=hi+gate),1)],axis=0)
        if len(p)<3:continue
        _,ix=np.unique(np.floor(p/.002).astype(np.int32),axis=0,return_index=True);fit=p[ix];tree=cKDTree(t)
        starts=[center] if fixed else [center,center+[gate*.7,0],center+[-gate*.7,0],center+[0,gate*.7],center+[0,-gate*.7]]
        base=[];tic=time.perf_counter()
        for a in starts:
            o=least_squares(lambda x:tree.query(fit-x)[0]/.015,np.clip(a,center-gate+1e-9,center+gate-1e-9),bounds=(center-gate,center+gate),loss='cauchy',diff_step=.001,max_nfev=24) if not fixed else None
            anchor=center if fixed else o.x;d=tree.query(p-anchor)[0];inside=np.all((p-anchor>=lo)&(p-anchor<=hi),1);n=int(np.sum(inside&(d<=.02)))
            err=float(np.mean(np.sort(d[inside])[:max(1,int(.7*inside.sum()))])) if n else .5
            coverage=np.mean(cKDTree(p).query(t[::2]+anchor)[0]<=.025) if n else 0.
            score=np.exp(-err/.015)+.25*coverage+.1*(1-np.exp(-n/8)) if n else -1.
            if not any(np.linalg.norm(anchor-b['anchor'])<.006 for b in base):base.append(dict(anchor=anchor,score=score,support=n,cost=0 if fixed else o.cost))
        base.sort(key=lambda r:-r['score']);ref_ms=(time.perf_counter()-tic)*1000
        fixtures.append((file.stem,c,base,ref_ms))
    for grid,cap in [(0,k) for k in (4,6,8,12)]+[(g,8) for g in (5,7,9)]:
        solver=native.Solver(1,cap,grid)
        for name,c,base,ref_ms in fixtures:
            tic=time.perf_counter();out=solver.fit_batch([(c['points'],c['center'],float(c['gate']),bool(c['fixed']))],c['template'],3)[0][0];dt=(time.perf_counter()-tic)*1000
            b=out[0];a=base[0];rows.append(dict(case=name,solver='GRID_LM' if grid else 'NATIVE_LM',grid=grid,iterations=cap,native_ms=dt,reference_ms=ref_ms,anchor_delta_m=float(np.linalg.norm(np.asarray(b['anchor'])-a['anchor'])),score_delta=b['score']-a['score'],cost_delta=b['cost']-a['cost'],same_selected_6mm=np.linalg.norm(np.asarray(b['anchor'])-a['anchor'])<.006,same_support=b['support_unique']==a['support']))
        subset=rows[-len(fixtures):];print('SOLVER',grid,cap,'same',sum(r['same_selected_6mm'] for r in subset),'/',len(subset),'median_ms',np.median([r['native_ms'] for r in subset]),flush=True)
    # Full independent candidate batches, no Python work in worker threads.
    for threads in (1,2,4,8,16):
        solver=native.Solver(threads,8,0);wall=time.perf_counter();cpu=time.process_time();equal=True
        for name,c,_,_ in fixtures:
            pack=(c['points'],c['center'],float(c['gate']),bool(c['fixed']))
            value=solver.fit_batch([pack]*3,c['template'],3)
            if threads==1:c['thread_reference']=value
            else:equal &= value==c['thread_reference']
        dt=time.perf_counter()-wall;active=(time.process_time()-cpu)/dt
        scaling.append(dict(threads=threads,seconds=dt,average_active_cores=active,parallel_efficiency=active/threads,deterministic=equal,**solver.stats()))
    lc.csv_write(OUT/'native_solver.csv',rows);lc.csv_write(OUT/'thread_scaling.csv',scaling)
    lc.save(OUT/'solver_study.json',dict(fixtures=len(fixtures),scaling=scaling,selection='No final selection: downstream development quality still required.'))

if __name__=='__main__':main()
