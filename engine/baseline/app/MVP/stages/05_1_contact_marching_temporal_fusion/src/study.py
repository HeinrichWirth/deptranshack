"""Inference phase: no evaluation imports and no labels or future-cloud reader."""
from fusion_common import *
from fusion import CausalSource,assemble
from tracker import predict
from bootstrap import production_seed,pose_record
from contact_rail_step2 import ContactRailDetector
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse

def configurations():
    c={'F0':dict(frames=1),'F2':dict(frames=2),'F3':dict(frames=3),'F4':dict(frames=4)}
    c.update({f'H{d:g}':dict(distance=d) for d in (.5,1.,2.,4.)})
    c.update({f'F4_V{int(v*1000)}':dict(frames=4,representation='VOXEL_UNIQUE',voxel=v) for v in (.005,.01,.02)})
    c['F4_BAL']=dict(frames=4,representation='FRAME_BALANCED')
    c.update({f'F4_TAU{d:g}':dict(frames=4,representation='FRAME_BALANCED',tau=d) for d in (.5,1.,2.)})
    return c

def without_clocks(x):
    if isinstance(x,dict):return {k:without_clocks(v) for k,v in x.items() if not k.endswith('_ms') and k not in ('io_ms',)}
    if isinstance(x,(list,tuple)):return [without_clocks(v) for v in x]
    return clean(x)

def predict_cloud(c,seed,config):
    cfg=dict(CFG,method=config.get('method','M1'));template=ContactRailDetector().template
    if config.get('representation')=='FRAME_BALANCED':
        from weighted_tracker import predict_weighted
        return predict_weighted(c['xyz'],seed,template,cfg,c['source_frame'],c['age_distance'],config.get('tau'))
    return predict(c['xyz'],seed,template,cfg)

def one(task):
    run,i,group,name,config=task;folder=OUT/group/name/key(run,i)
    if (folder/'PREDICTION_COMPLETE.json').exists():return name,run,i,'cached'
    seed=seed_from_cache(run,i);source=CausalSource(run,i);start=time.perf_counter()
    if seed['status']!='AVAILABLE' and not(config.get('fused_bootstrap') and seed.get('basis') is not None):
        c,seed=assemble(source,dict(frames=1),seed)
        c['meta']['skipped_history_because']='CURRENT_ONLY_BOOTSTRAP_UNAVAILABLE'
    else:c,seed=assemble(source,config,seed)
    if config.get('fused_bootstrap'):
        seed=production_seed(c['xyz'],pose_record(source.past[-1]),source.direction_next,load(ROOT/'MVP/stages/01_rail2d/config/detector_v2.json'),ContactRailDetector())
    p=predict_cloud(c,seed,config)
    c['meta']['total_start_ms']=(time.perf_counter()-start)*1000+seed.get('seed_ms',0)
    save_prediction(folder,p,run,i,name,c)
    if name=='F0' and group=='heldout':
        prior=load_prediction(BASE_OUT/'heldout'/key(run,i));tests={}
        for k in ARRAY_KEYS:
            if k in prior:np.testing.assert_array_equal(p[k],prior[k]);tests[k]=True
        assert without_clocks(p['steps'])==without_clocks(prior['steps']),'Baseline step mismatch'
        assert p['reason']==prior['reason'];save(folder/'BASELINE_IDENTICAL.json',dict(arrays=tests,steps_identical_except_timers=True,reason_identical=True))
    return name,run,i,p['reason'],len(p['indices']),round(c['meta']['total_start_ms'],1)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('group',choices=['development','heldout']);ap.add_argument('--variants',nargs='+');ap.add_argument('--workers',type=int,default=6);ap.add_argument('--limit',type=int);a=ap.parse_args();initialize()
    configs=load(OUT/'configs.json') if (OUT/'configs.json').exists() else configurations()
    if not (OUT/'configs.json').exists():save(OUT/'configs.json',configs)
    if a.group=='heldout':
        freeze=load(OUT/'research_lock.json');assert freeze['config_sha256']==sha(OUT/'configs.json')
        for path,h in freeze['inference_sha256'].items():assert sha(STAGE/path)==h,path
    rows=load(OUT/'audit/cohort.json')[a.group]
    if a.limit:rows=rows[:a.limit]
    chosen=a.variants or (freeze['heldout_variants'] if a.group=='heldout' else list(configs))
    tasks=[(r['run'],r['frame'],a.group,name,configs[name]) for name in chosen for r in rows]
    t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(one,task) for task in tasks]
        for j,f in enumerate(as_completed(futures)):
            v=f.result()
            if j%25==0 or j==len(futures)-1:print('PREDICT',j+1,len(futures),v,'elapsed',round(time.perf_counter()-t,1),flush=True)
if __name__=='__main__':main()
