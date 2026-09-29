from . import ROOT,OUT
from .native_api import startup
import long_common as lc
import numpy as np
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
from scipy.optimize._numdiff import approx_derivative
import time,sys,platform

def clock(fn,repeats=20):
    fn();samples=[]
    for j in range(3):
        t=time.perf_counter()
        for _ in range(repeats):fn()
        samples.append((time.perf_counter()-t)/repeats)
    return float(np.median(samples))

def main():
    native=startup();nn=[];residual=[];jac=[];eq=[];layout=[];fits=[]
    captures=[];cfg=lc.load(ROOT/'results_performance_final/profiles/capture_config.json')['cfg']
    for path in sorted((ROOT/'results_performance_final/profiles').glob('fit_*.npz')):
        with np.load(path) as z:c={k:z[k] for k in z.files}
        c['id']=path.stem;captures.append(c)
    for c in captures:
        p=c['points'];t=c['template'];a=c['center'];gate=float(c['gate']);lo=t.min(0)-.015;hi=t.max(0)+.015
        points=np.unique(p[np.all((p-a>=lo-gate)&(p-a<=hi+gate),axis=1)],axis=0)
        if not len(points):continue
        _,ix=np.unique(np.floor(points/.002).astype(np.int32),axis=0,return_index=True);fitp=points[ix]
        for direction,reference,queries in [('forward',t,fitp-a),('reverse',points,t[::2]+a)]:
            tree=cKDTree(reference);old=tree.query(queries)[0];new=native.exact_nn_bruteforce(queries,reference)
            assert np.array_equal(old,new)
            kd=clock(lambda:cKDTree(reference).query(queries)[0]);reuse=clock(lambda:tree.query(queries)[0]);brute=clock(lambda:native.exact_nn_bruteforce(queries,reference))
            nn.append(dict(case=c['id'],direction=direction,N=len(reference),queries=len(queries),kdtree_build_query_ns=kd*1e9,
                kdtree_query_ns=reuse*1e9,native_ns=brute*1e9,ns_per_query=brute*1e9/len(queries),bitwise=True))
        obj=native.Problem(fitp,t,a-gate,a+gate,False);objsoa=native.Problem(fitp,t,a-gate,a+gate,True);tree=cKDTree(t)
        fun=lambda x:tree.query(fitp-x)[0]/.015
        f=fun(a);nf=obj.fun(a);J=approx_derivative(fun,a,method='2-point',rel_step=.001,bounds=(a-gate,a+gate));nj=obj.jac(a)
        df=float(np.max(abs(f-nf)));dj=float(np.max(abs(J-nj)))
        eq.append(dict(case=c['id'],residual_max=df,jacobian_max=dj,residual_bitwise=np.array_equal(f,nf),jacobian_bitwise=np.array_equal(J,nj),passed=df<=1e-10 and dj<=1e-10))
        pytime=clock(lambda:fun(a));nt=clock(lambda:obj.fun(a));jt=clock(lambda:obj.jac(a));pj=clock(lambda:approx_derivative(fun,a,method='2-point',rel_step=.001,bounds=(a-gate,a+gate)),5)
        residual.append(dict(case=c['id'],points=len(fitp),python_us=pytime*1e6,native_us=nt*1e6,speedup=pytime/nt,workspace_bytes=obj.bytes()))
        jac.append(dict(case=c['id'],python_us=pj*1e6,native_us=jt*1e6,speedup=pj/jt,max_difference=dj,bitwise=np.array_equal(J,nj)))
        layout.append(dict(case=c['id'],aos_us=nt*1e6,soa_us=clock(lambda:objsoa.fun(a))*1e6,
            gil_held_us=clock(lambda:obj.fun(a,False))*1e6,gil_released_us=nt*1e6,bitwise=np.array_equal(nf,objsoa.fun(a))))
        if len(fits)<600 and len(fitp)>=cfg['min_support']:
            for seed_id,seed in enumerate([a,a+[gate*.7,0],a+[-gate*.7,0],a+[0,gate*.7],a+[0,-gate*.7]]):
                reference=None
                for mode in ('original','residual','fd','analytic'):
                    problem=native.Problem(fitp,t,a-gate,a+gate,False)
                    kw={} if mode in ('original','residual') else dict(jac=problem.jac if mode=='fd' else lambda x:problem.jac(x,True))
                    tic=time.perf_counter();o=least_squares(fun if mode=='original' else problem.fun,np.clip(seed,a-gate+1e-9,a+gate-1e-9),
                        bounds=(a-gate,a+gate),loss='cauchy',f_scale=1.,diff_step=1e-3,max_nfev=24,**kw);elapsed=time.perf_counter()-tic
                    if reference is None:reference=o
                    dx=float(np.max(abs(o.x-reference.x)));dc=abs(o.cost-reference.cost)
                    same=o.status==reference.status and o.nfev==reference.nfev and o.njev==reference.njev and np.array_equal(o.active_mask,reference.active_mask)
                    fits.append(dict(case=c['id'],seed=seed_id,backend=mode,ms=elapsed*1000,parameters=len(o.x),nfev=o.nfev,njev=o.njev,
                        iterations=None,cost=o.cost,cost_delta=dc,max_parameter_delta=dx,active_mask=o.active_mask.tolist(),status=o.status,
                        discrete_solver_same=same,passed=same and dx<=1e-10 and dc<=1e-10,bitwise=np.array_equal(o.x,reference.x),
                        native_fun_calls=problem.fun_calls,native_jac_calls=problem.jac_calls))
    sweep=[];selected={}
    for direction in ('forward','reverse'):
        rr=[r for r in nn if r['direction']==direction];base=sum(r['kdtree_query_ns' if direction=='forward' else 'kdtree_build_query_ns'] for r in rr)
        choices=[]
        for threshold in (0,32,64,128,256,512,1024):
            cost=sum(r['native_ns'] if r['N']<threshold else r['kdtree_query_ns' if direction=='forward' else 'kdtree_build_query_ns'] for r in rr)
            sweep.append(dict(direction=direction,threshold=threshold,total_ns=cost,speedup=base/cost));choices.append((cost,threshold))
        selected[direction]=min(choices)[1]
    lc.csv_write(OUT/'nn_benchmark.csv',nn);lc.csv_write(OUT/'nn_threshold_sweep.csv',sweep)
    lc.csv_write(OUT/'residual_benchmark.csv',residual);lc.csv_write(OUT/'jacobian_benchmark.csv',jac)
    lc.csv_write(OUT/'native_equivalence.csv',eq);lc.csv_write(OUT/'layout_gil.csv',layout);lc.csv_write(OUT/'fit_backends.csv',fits)
    assert all(r['passed'] for r in eq)
    lc.save(OUT/'micro_summary.json',dict(thresholds=selected,cases=len(eq),max_residual=max(r['residual_max'] for r in eq),
        max_jacobian=max(r['jacobian_max'] for r in eq),fd_fits_passed=all(r['passed'] for r in fits if r['backend']=='fd'),
        analytic_fits_passed=all(r['passed'] for r in fits if r['backend']=='analytic'),python=sys.version,platform=platform.platform()))
    print(lc.load(OUT/'micro_summary.json'),flush=True)

if __name__=='__main__':main()
