"""Prediction phase; never imports the evaluation or oracle modules."""
from common import *
from bootstrap import production_seed,pose_record,initial_frame
from tracker import predict,DEFAULT
from contact_rail_step2 import ContactRailDetector
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse

ARRAY_KEYS=('indices','point_step','local_uvw','template_residual','confidence','point_anchor','s_from_seed','curve')

def seed_geometry(run,i):
    cache=OUT/'cache'/key(run,i); cache.mkdir(parents=True,exist_ok=True)
    mm=frames(run);row=mm[i]; p=cache/'seed.json'
    if p.exists():
        xyz=np.load(cache/'xyz.npy');seed=load(p)
        for k in ('basis','anchor','indices'):
            if seed.get(k) is not None:seed[k]=np.asarray(seed[k],dtype=np.int64 if k=='indices' else float)
        return xyz,seed
    t=time.perf_counter();xyz=read_geometry(run,row);io_ms=1000*(time.perf_counter()-t)
    contact=ContactRailDetector();cfg=load(ROOT/'MVP/stages/01_rail2d/config/detector_v2.json')
    seed=production_seed(xyz,pose_record(row),pose_record(mm[i+1]) if i+1<len(mm) else None,cfg,contact)
    seed['io_ms']=io_ms
    np.save(cache/'xyz.npy',xyz);save(p,seed)
    return xyz,seed

def save_prediction(folder,result,run,i):
    folder.mkdir(parents=True,exist_ok=True)
    arrays={k:result[k] for k in ARRAY_KEYS if k in result}
    np.savez_compressed(folder/'points.npz',**arrays)
    record={k:v for k,v in result.items() if k not in ARRAY_KEYS};record.update(run=run,start_frame=i)
    save(folder/'prediction.json',record)
    save(folder/'PREDICTION_COMPLETE.json',dict(prediction_sha256=sha(folder/'prediction.json'),points_sha256=sha(folder/'points.npz'),
         time_ns=time.time_ns(),inference_input='current frame XYZ + sanitized pose T and T+1 + frozen bootstrap; no classification'))

def one(task):
    run,i,cfg,group=task;folder=OUT/group/key(run,i)
    if (folder/'PREDICTION_COMPLETE.json').exists():return run,i,'cached'
    xyz,seed=seed_geometry(run,i);contact=ContactRailDetector();result=predict(xyz,seed,contact.template,cfg)
    save_prediction(folder,result,run,i)
    return run,i,result['reason'],round(result['total_ms'],1),len(result['indices'])

def cohort(dev_n=12):
    rows=[]
    for split,runs in SPLIT.items():
        for run in runs:
            mm=frames(run)
            if split=='validation':ids=range(len(mm))
            else:
                eligible=[i for i in range(len(mm)-1) if initial_frame(pose_record(mm[i]),pose_record(mm[i+1]))[0] is not None]
                ids=[eligible[j] for j in np.unique(np.linspace(0,len(eligible)-1,min(dev_n,len(eligible))).astype(int))]
            rows += [dict(run=run,frame=int(i),split=split) for i in ids]
    return rows

def batch(tasks,workers=4):
    t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        ff=[pool.submit(one,x) for x in tasks]
        for j,f in enumerate(as_completed(ff)):
            r=f.result()
            if j%20==0 or j==len(ff)-1:print('PREDICT',j+1,'/',len(ff),r,'elapsed',round(time.perf_counter()-t,1),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['pilot','development','heldout']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args();initialize()
    if args.phase=='pilot':rr=[dict(run=r,frame=frames(r).__len__()//3) for r in SPLIT['development'][-3:]];cfg=DEFAULT
    elif args.phase=='development':rr=[r for r in cohort() if r['split']=='development'];save(OUT/'audit/cohort.json',cohort());cfg=DEFAULT
    else:
        assert (OUT/'freeze.json').exists(),'Freeze before held-out inference'
        cfg=load(OUT/'config.json');rr=[r for r in load(OUT/'audit/cohort.json') if r['split']=='validation']
        fr=load(OUT/'freeze.json')
        for p,h in fr['inference_sha256'].items():assert sha(STAGE/p)==h,p
    batch([(r['run'],r['frame'],cfg,args.phase) for r in rr],args.workers)
if __name__=='__main__':main()
