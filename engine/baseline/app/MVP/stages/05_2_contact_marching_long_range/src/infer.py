"""Run causal inference and persist predictions before evaluation can begin."""
from long_common import *
from roi_source import RoiSource,point_keys
from fusion import CausalSource,assemble
from study import predict_cloud,without_clocks
from long_tracker import infer
from contact_rail_step2 import ContactRailDetector
from configurations import configurations
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse

def baseline(run,i,name,cfg,seed):
    source=CausalSource(run,i);c,_=assemble(source,dict(frames=cfg['frames'] if seed['status']=='AVAILABLE' else 1),seed)
    p=predict_cloud(c,seed,dict(frames=cfg['frames']))
    oldname='F0' if name=='B0' else 'F'+str(cfg['frames'])
    group='development' if run in SPLIT['development'] else 'heldout'
    old=OLD/group/oldname/key(run,i)
    if (old/'PREDICTION_COMPLETE.json').exists():
        prior=read_prediction(old)
        for f in ARRAYS:
            if f in prior and f in p:np.testing.assert_array_equal(p[f],prior[f])
        assert without_clocks(p['steps'])==without_clocks(prior['steps']),(name,run,i,'step mismatch')
        assert p['reason']==prior['reason']
        p['baseline_identical']=True;p['baseline_source']=str(old)
    else:p['baseline_identical']=None;p['baseline_source']='NO_PREVIOUS_RESULT_FOR_THIS_DEVELOPMENT_ABLATION'
    p['point_state']=np.full(len(p['indices']),2,dtype=np.uint8)
    p['hypotheses']=[]
    c['keys']=point_keys(c)
    return p,c

def one(task):
    run,i,group,names=task;configs=load(OUT/'configs.json');seed=seed_from_cache(run,i)
    source=None;template=ContactRailDetector().template;messages=[]
    for name in names:
        cfg=configs[name];folder=OUT/group/name/key(run,i)
        if (folder/'PREDICTION_COMPLETE.json').exists():messages.append((name,'cached'));continue
        begin=time.perf_counter()
        if cfg.get('baseline'):p,c=baseline(run,i,name,cfg,seed)
        else:
            if source is None:source=RoiSource(run,i)
            p,c=infer(source,seed,template,cfg)
        p['start_processing_ms']=(time.perf_counter()-begin)*1000
        p['seed_cached']=True
        save_prediction(folder,p,c,run,i,name)
        messages.append((name,p['reason'],len(p['indices']),round(p['total_ms'])))
    return run,i,messages

def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['a','b','d','e','smoke']);ap.add_argument('--variants',nargs='+');ap.add_argument('--workers',type=int,default=8);ap.add_argument('--limit',type=int);a=ap.parse_args()
    groups={'a':'phase_a_screen','b':'phase_b_full_dev','d':'phase_d_combinations','e':'phase_e_benchmark','smoke':'audit/smoke'};group=groups[a.phase]
    configs=load(OUT/'configs.json')
    if a.phase=='e':
        lock=load(OUT/'research_freeze.json');assert lock['code_sha256']==code_hashes();assert lock['config_sha256']==sha(OUT/'configs.json')
        names=a.variants or ['B0','B2','B4','B8']+lock['top3'];rows=load(OUT/'audit/cohort.json')['heldout']
    elif a.phase in ('a','smoke'):names=a.variants or list(configurations());rows=load(OUT/'audit/screen.json')
    else:names=a.variants or load(OUT/('phase_b_selection.json' if a.phase=='b' else 'phase_d_selection.json'))['variants'];rows=load(OUT/'audit/cohort.json')['development']
    if a.limit:rows=rows[:a.limit]
    tasks=[(r['run'],r['frame'],group,names) for r in rows];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(one,t) for t in tasks]
        for n,f in enumerate(as_completed(futures),1):
            run,i,msg=f.result();print('INFER',group,n,len(tasks),run,i,msg,'elapsed',round(time.perf_counter()-start,1),flush=True)

if __name__=='__main__':main()
