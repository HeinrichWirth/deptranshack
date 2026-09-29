from . import ROOT,OUT
from .native_backend import load_native
from .hotspots import make_tracker
import long_common as lc
from scipy.spatial import cKDTree
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor


def main():
    native=load_native();cfg=lc.load(OUT/'profiles/capture_config.json')['cfg'];eq=[];timings=[];fits=[]
    captures=[]
    for path in sorted((OUT/'profiles').glob('fit_*.npz')):
        with np.load(path) as z:c={k:z[k] for k in z.files}
        captures.append(c)
        tree=cKDTree(c['template']);p=c['points'];a=c['center']
        old=tree.query(p-a)[0];new=native.distances(p,a,c['template'])
        difference=float(np.max(abs(old-new))) if len(p) else 0.
        eq.append(dict(case=path.stem,points=len(p),template=len(c['template']),max_abs_difference=difference,
            tolerance=1e-10,passed=difference<=1e-10,bitwise=np.array_equal(old,new)))
        for name,fn in [('python',lambda:tree.query(p-a)[0]),('native',lambda:native.distances(p,a,c['template']))]:
            samples=[]
            for repeat in range(3):
                t=time.perf_counter()
                for _ in range(50):fn()
                samples.append((time.perf_counter()-t)/50)
            timings.append(dict(case=path.stem,backend=name,median_us=float(np.median(samples))*1e6,p95_us=float(np.percentile(samples,95))*1e6))
        if len(captures)<=40:
            results=[]
            for backend in ('python','optimized','native'):
                tt=make_tracker(backend,None)(c['template'],1,cfg);t=time.perf_counter()
                value=tt.fit(c['points'],c['center'],float(c['gate']),bool(c['fixed']));dt=time.perf_counter()-t
                results.append(value)
                fits.append(dict(case=path.stem,backend=backend,ms=dt*1000,candidates=len(value[0])))
            for backend,result in zip(('optimized','native'),results[1:]):
                original=results[0]
                exact=len(original[0])==len(result[0]) and original[1]==result[1]
                delta=0.
                if exact:
                    for a,b in zip(original[0],result[0]):
                        exact &= a['support_unique']==b['support_unique'] and a['success']==b['success']
                        delta=max(delta,float(np.max(abs(a['anchor']-b['anchor']))),abs(a['score']-b['score']))
                fits[-2 if backend=='optimized' else -1].update(decisions_same=exact,max_anchor_score_difference=delta,passed=exact and delta<=1e-10)
    lc.csv_write(OUT/'native_equivalence.csv',eq);lc.csv_write(OUT/'native_benchmark.csv',timings);lc.csv_write(OUT/'native_fit_benchmark.csv',fits)
    # Same complete frozen fit inputs; preserve map order when comparing concurrency.
    data=captures[:40]
    def one(c):
        tt=make_tracker('python',None)(c['template'],1,cfg)
        return tt.fit(c['points'],c['center'],float(c['gate']),bool(c['fixed']))
    thread_rows=[]
    expected=None
    for workers in (1,2,4):
        t=time.perf_counter()
        if workers==1:result=list(map(one,data))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:result=list(pool.map(one,data))
        elapsed=time.perf_counter()-t
        from MVP.final_pipeline.verify_raw import normalized
        actual=normalized(result)
        if expected is None:expected=actual
        assert actual==expected
        thread_rows.append(dict(workers=workers,seconds=elapsed,fits=len(data),bitwise_identical=True))
    lc.csv_write(OUT/'threading_benchmark.csv',thread_rows)
    lc.save(OUT/'native_test.json',dict(cases=len(eq),max_abs_difference=max(r['max_abs_difference'] for r in eq),
        kernel_passed=all(r['passed'] for r in eq),fit_passed=all(r.get('passed',True) for r in fits),native_ABI=1))
    assert all(r['passed'] for r in eq) and all(r.get('passed',True) for r in fits)
    print(lc.load(OUT/'native_test.json'),flush=True)


if __name__=='__main__':main()
