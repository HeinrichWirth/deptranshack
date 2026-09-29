"""OFFLINE ONLY: GT-admitted mild overlap violations, never imported by inference."""
from long_common import *
from fusion import CausalSource,assemble
from contact_rail_step2 import ContactRailDetector
from scipy.spatial import cKDTree
from evaluation import evaluate_prediction
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,types

def oracle_engine():
    original=ROOT/'MVP/stages/05_contact_rail_marching/src/tracker.py'
    code=original.read_text(encoding='utf-8')
    a="if rec['overlap_residual_p90']>cfg['overlap_tolerance']:reason='OVERLAP_INCONSISTENT';break"
    b="if np.quantile(tt.distances(overlap,a),.90)>cfg['overlap_tolerance']:reason='OVERLAP_INCONSISTENT';break"
    assert code.count(a)==1 and code.count(b)==1
    # First old fit can be mildly inconsistent, but new support still has to pass
    # the second gate's GT test before any point is accepted.
    code=code.replace(a,"if rec['overlap_residual_p90']>.06:reason='OVERLAP_INCONSISTENT';break")
    code=code.replace(b,"if (np.quantile(tt.distances(overlap,a),.90)>cfg['overlap_tolerance'] or rec['overlap_residual_p90']>cfg['overlap_tolerance']) and not oracle_admit(xyz[new_ids], uvw[new_ids,1:], a, tt, float(np.quantile(tt.distances(overlap,a),.90))):reason='OVERLAP_INCONSISTENT';break")
    module=types.ModuleType('offline_overlap_oracle');exec(compile(code,str(original)+'::OFFLINE_ORACLE','exec'),module.__dict__)
    return module,sha(original)

def one(task):
    run,i,group=task;folder=OUT/group/'B8'/key(run,i);marker=check_marker(folder);target=OUT/'audit/oracle_threshold'/key(run,i)
    if (target/'result.json').exists():return run,i,'cached'
    seed=seed_from_cache(run,i)
    if seed['status']!='AVAILABLE':return run,i,'unavailable'
    ref,info=reference_readonly(run,i)
    if not len(ref['future']):return run,i,'no GT'
    tree=cKDTree(ref['future']);module,h=oracle_engine();admissions=[]
    def admit(xyz,yz,a,tt,old_residual):
        d=tt.distances(yz,a);inside=np.all((yz-a>=tt.lo)&(yz-a<=tt.hi),axis=1);p=xyz[inside&(d<=.02)]
        match=tree.query(p)[0] if len(p) else np.empty(0)
        ok=old_residual<=.06 and len(np.unique(p,axis=0))>=3 and np.mean(match<=.05)>=.95
        admissions.append(dict(overlap_p90=old_residual,points=len(p),matched_fraction=float(np.mean(match<=.05)) if len(p) else None,admitted=bool(ok)))
        return ok
    module.oracle_admit=admit
    c,_=assemble(CausalSource(run,i),dict(frames=8),seed);p=module.predict(c['xyz'],seed,ContactRailDetector().template,CFG);p.update(run=run,start_frame=i)
    row,steps,_,_=evaluate_prediction(p,c['xyz'],ref,info)
    save(target/'result.json',dict(summary=row,steps=steps,admissions=admissions,evaluation_only=True,baseline_code_sha=h,
      prediction_completed_before_oracle_ns=marker['time_ns'],oracle_completed_ns=time.time_ns(),bound='Only 3..6cm overlap vetoes bypassed when actual proposed NEW observations match future GT <=5cm (95%, at least 3 positions). Other gates unchanged; F8 whole-cloud geometry.'))
    return run,i,row.get('continuous_reach_5cm'),sum(x['admitted'] for x in admissions)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('group');ap.add_argument('--workers',type=int,default=6);a=ap.parse_args()
    rr=load(OUT/'audit/cohort.json')['heldout' if a.group=='phase_e_benchmark' else 'development'];t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        ff=[pool.submit(one,(r['run'],r['frame'],a.group)) for r in rr]
        for j,f in enumerate(as_completed(ff),1):
            v=f.result()
            if j%30==0 or j==len(ff):print('ORACLE',j,len(ff),v,'elapsed',round(time.perf_counter()-t,1),flush=True)
    rows=[dict(load(p)['summary'],admitted_gates=sum(v['admitted'] for v in load(p)['admissions'])) for p in (OUT/'audit/oracle_threshold').glob('*/result.json')];csv_write(OUT/'fusion/oracle_threshold.csv',rows)

if __name__=='__main__':main()
