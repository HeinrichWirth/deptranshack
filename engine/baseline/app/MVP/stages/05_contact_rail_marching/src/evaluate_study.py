from common import *
from evaluation import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse

def one(task):
    run,i,group=task;folder=require_prediction(group,run,i)
    if (folder/'evaluation.json').exists() and load(folder/'evaluation.json').get('version')==2:return run,i,'cached'
    pred=load_prediction(folder)
    if pred['seed']['status']!='AVAILABLE':
        # No propagation exists to evaluate. Do not materialize hundreds of future
        # scans for an unavailable start; retain it in the complete census.
        row,sr,bins,point=evaluate_prediction(pred,np.empty((0,3)),{},dict(future_frames=0))
        row.update(evaluation_eligible=False,terminal_evaluation_reason='START_UNAVAILABLE')
        save(folder/'evaluation.json',dict(version=2,summary=row,steps=sr,bins=bins,reference=dict(reason='START_UNAVAILABLE',future_GT_not_read=True)))
        np.savez_compressed(folder/'point_evaluation.npz',**point)
        return run,i,'START_UNAVAILABLE'
    xyz=np.load(OUT/'cache'/key(run,i)/'xyz.npy');ref,info=reference(run,i,group)
    row,sr,bins,point=evaluate_prediction(pred,xyz,ref,info)
    save(folder/'evaluation.json',dict(version=2,summary=row,steps=sr,bins=bins,reference=info));np.savez_compressed(folder/'point_evaluation.npz',**point)
    return run,i,row.get('continuous_reach_5cm'),row.get('point_error_p95')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('group');ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    rr=[(p.parent.name.split('__')[0],int(p.parent.name.split('__')[1]),args.group) for p in (OUT/args.group).glob('*/PREDICTION_COMPLETE.json')]
    runs=sorted(set(r[0] for r in rr));t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(prepare_run,r,args.group,[i for run,i,g in rr if run==r]) for r in runs]
        for f in as_completed(fs):print('REFERENCE_SOURCE',f.result(),round(time.perf_counter()-t,1),flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(one,r) for r in rr]
        for j,f in enumerate(as_completed(fs)):
            result=f.result()
            if j%20==0 or j==len(fs)-1:print('EVAL',j+1,len(fs),result,round(time.perf_counter()-t,1),flush=True)
if __name__=='__main__':main()
