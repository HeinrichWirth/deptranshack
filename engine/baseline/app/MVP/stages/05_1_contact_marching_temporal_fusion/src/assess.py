"""Offline-only evaluation after a hash-verified completion marker.

Uses exactly STEP5 frozen reference arrays and evaluator; never modifies STEP5.
"""
from fusion_common import *
from fusion import CausalSource,assemble
from evaluation import evaluate_prediction
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse

def reference_readonly(run,i):
    path=BASE_OUT/'future_gt'/(key(run,i)+'_reference_v2.npz')
    with np.load(path) as z:ref={k:z[k] for k in z.files}
    return ref,load(path.with_suffix('.json'))

def check_marker(folder):
    marker=load(folder/'PREDICTION_COMPLETE.json')
    assert sha(folder/'prediction.json')==marker['prediction_sha256'];assert sha(folder/'points.npz')==marker['points_sha256']
    return marker

def one(path):
    folder=Path(path)
    if (folder/'evaluation.json').exists():return folder.parent.name,'cached'
    marker=check_marker(folder);p=load_prediction(folder);run=p['run'];i=p['start_frame'];variant=p['variant']
    if p['seed']['status']!='AVAILABLE':
        row=dict(run=run,start_frame=i,variant=variant,seed_available=False,evaluation_eligible=False,stop_reason=p['reason'],terminal_evaluation_reason='START_UNAVAILABLE',march_steps=0)
        e=dict(summary=row,steps=[],bins=[],reference={},evaluated_after_prediction_ns=time.time_ns());save(folder/'evaluation.json',e);return variant,'unavailable'
    if not (BASE_OUT/'future_gt'/(key(run,i)+'_reference_v2.npz')).exists():
        row=dict(run=run,start_frame=i,variant=variant,seed_available=True,evaluation_eligible=False,stop_reason=p['reason'],terminal_evaluation_reason='NO_BASELINE_REFERENCE_FOR_DIAGNOSTIC_SEED',march_steps=sum(s['status']=='ACCEPTED' for s in p['steps']))
        save(folder/'evaluation.json',dict(summary=row,steps=[],bins=[],reference={},evaluated_after_prediction_ns=time.time_ns()));return variant,'no baseline reference'
    ref,info=reference_readonly(run,i)
    c,_=assemble(CausalSource(run,i),p['fusion']['config'],seed_from_cache(run,i))
    row,steps,bins,point=evaluate_prediction(p,c['xyz'],ref,info)
    row['variant']=variant
    for r in steps+bins:r['variant']=variant
    bad=[s for s in steps if s['status']=='ACCEPTED' and (s.get('point_p95') or 0)>.20]
    row['raw_wrong_structure_trigger']=bool(bad)
    if bad and all(s.get('evaluated_fraction',0)<.80 for s in bad):
        row['wrong_structure_suspected']=False;row['wrong_structure_review']='INSUFFICIENT_GT_COVERAGE_AT_REFERENCE_END'
    np.savez_compressed(folder/'point_evaluation.npz',**point)
    e=dict(summary=row,steps=steps,bins=bins,reference=info,evaluated_after_prediction_ns=time.time_ns(),prediction_time_ns=marker['time_ns'],evaluator_sha256=sha(PREVIOUS/'src/evaluation.py'))
    save(folder/'evaluation.json',e)
    if variant=='F0' and folder.parent.parent.name=='heldout':
        prior=load(BASE_OUT/'heldout'/key(run,i)/'evaluation.json')['summary']
        fields=[k for k in prior if k.startswith(('continuous_reach','point_matched','anchor_error','max_accepted','point_error','farthest_correct'))]
        for k in fields:assert row.get(k)==prior.get(k),(run,i,k,row.get(k),prior.get(k))
        save(folder/'BASELINE_METRICS_IDENTICAL.json',dict(fields=fields,exact=True))
    return variant,run,i,row.get('continuous_reach_5cm')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('group',choices=['development','heldout']);ap.add_argument('--variants',nargs='+');ap.add_argument('--workers',type=int,default=6);a=ap.parse_args()
    names=a.variants or [p.name for p in (OUT/a.group).iterdir() if p.is_dir()]
    paths=[str(p.parent) for name in names for p in (OUT/a.group/name).glob('*/PREDICTION_COMPLETE.json')];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(one,p) for p in paths]
        for j,f in enumerate(as_completed(futures)):
            v=f.result()
            if j%25==0 or j==len(futures)-1:print('EVALUATE',j+1,len(futures),v,'elapsed',round(time.perf_counter()-start,1),flush=True)
if __name__=='__main__':main()
