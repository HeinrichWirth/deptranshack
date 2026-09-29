from . import ROOT,OUT
from .engine import Engine
from MVP.performance_final.ablation import fingerprint
from MVP.final_pipeline.verify_raw import normalized
import long_common as lc
import numpy as np
import time

def main():
    best=lc.load(OUT/'selection.json')['backend'];cohort=lc.load(OUT/'benchmark_cohort.json')['final'];rows=[];run=None;baseline=None;optimized=None
    for case in cohort:
        if case['run']!=run:
            if optimized:optimized.close()
            run=case['run'];baseline=Engine('python');optimized=Engine(best)
            for e in (baseline,optimized):e.start_run(lc.dataset()/run)
        a=baseline.process_frame(case['frame']);b=optimized.process_frame(case['frame'])
        same=fingerprint(a)==fingerprint(b);deltas=[];xyz=[]
        for part in ('prediction','c4_smooth','seed','c4'):
            if a[part] is None:continue
            for k,v in a[part].items():
                if isinstance(v,np.ndarray) and np.issubdtype(v.dtype,np.floating):
                    w=b[part][k]
                    if v.shape==w.shape:
                        valid=np.isfinite(v)&np.isfinite(w);delta=abs(v[valid]-w[valid]);deltas.extend(delta.tolist())
                        if k in ('C','pair','center','state'):xyz.extend(delta.tolist())
        values=np.asarray(xyz or deltas or [0.]);row=dict(**case,backend=best,bitwise=same,discrete_exact=same,
            max_abs=float(values.max()),p99_abs=float(np.percentile(values,99)),median_abs=float(np.median(values)),
            baseline_ms=a['summary']['timing']['T_TOTAL']*1000,best_ms=b['summary']['timing']['T_TOTAL']*1000,
            available=a['prediction'] is not None)
        rows.append(row)
        folder=OUT/'equivalence'/run/f'{case["frame"]:06d}';folder.mkdir(parents=True,exist_ok=True)
        if a['prediction'] is not None:
            np.savez_compressed(folder/'original.npz',**a['prediction']);np.savez_compressed(folder/'best.npz',**b['prediction'])
        lc.save(folder/'PREDICTION_COMPLETE.json',dict(time_ns=time.time_ns(),original_hash=fingerprint(a),best_hash=fingerprint(b),labels_used=False,
            files={p.name:lc.sha(p) for p in folder.glob('*.npz')}))
        print('FINAL_EQ',len(rows),len(cohort),run,case['frame'],same,flush=True)
        if not same:
            lc.save(folder/'original.json',normalized(a['c4']));lc.save(folder/'divergent.json',normalized(b['c4']))
            lc.csv_write(OUT/'full_equivalence.csv',rows);raise RuntimeError('Final equivalence mismatch')
    optimized.close();lc.csv_write(OUT/'full_equivalence.csv',rows)
    lc.save(OUT/'FINAL_EQ_COMPLETE.json',dict(cases=len(rows),bitwise=True,discrete_differences=0,max_difference=max(r['max_abs'] for r in rows),time_ns=time.time_ns(),backend=best))

if __name__=='__main__':main()
