"""Production-like rail predictions must finish on disk BEFORE opening future GT."""
from rr_common import *
from seed_input import save_input,read_input
from rail_models import predict,configurations,INPUT_KEYS
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse

INFERENCE_FILES=('track_geometry.py','seed_input.py','rail_models.py','run_inference.py','rr_common.py')
def code_hashes():return {n:sha(STAGE/'src'/n) for n in INFERENCE_FILES}
def check_freeze():
    lock=load(OUT/'research_freeze.json');assert lock['code_sha256']==code_hashes()
    for path,field in ((OUT/'configs.json','config_sha256'),(OUT/'models/global_prior.json','global_prior_sha256'),(STAGE/'dependency_lock.json','dependency_lock_sha256')):assert sha(path)==lock[field],str(path)
    for n,h in lock['evaluation_sha256'].items():assert sha(STAGE/'src'/n)==h,n
    return lock
def one(task):
    r,group,names,seed_length=task;run=r['run'];i=r['frame'];cfgs=load(OUT/'configs.json');folder=save_input(run,i,seed_length);meta,a=read_input(folder);messages=[]
    for name in names:
        dest=OUT/group/name/key(run,i)
        prior_sha=sha(OUT/'models/global_prior.json')
        if (dest/'PREDICTION_COMPLETE.json').exists():
            marker=load(dest/'PREDICTION_COMPLETE.json');record=load(dest/'prediction.json')
            if marker.get('code_sha256')==code_hashes() and record['config']==cfgs[name] and record['input_sha']==sha(folder/'COMPLETE.json') and marker.get('prior_sha256')==prior_sha:
                for n,h in marker['files'].items():assert sha(dest/n)==h
                messages.append((name,'cached'));continue
        t=time.perf_counter();out={}
        if meta['status']=='AVAILABLE':
            priors=load(OUT/'models/global_prior.json') if (OUT/'models/global_prior.json').exists() else None
            prior=priors['leave_run_out'].get(run,priors['default']) if priors else None
            out=predict(meta['seed'],{k:a[k] for k in INPUT_KEYS},meta['profiles'],meta['q'],cfgs[name],prior)
        dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'prediction.npz',**out)
        rec=dict(run=run,i=i,method=name,status=meta['status'],cr_input='CR1_C4',seed_length=seed_length,input_folder=folder.relative_to(OUT).as_posix(),input_sha=sha(folder/'COMPLETE.json'),config=cfgs[name],prediction_ms=(time.perf_counter()-t)*1000,input_prepare_ms=meta.get('ms'),pure_prediction=True)
        save(dest/'prediction.json',rec);save(dest/'PREDICTION_COMPLETE.json',dict(time_ns=time.time_ns(),files={n:sha(dest/n) for n in ('prediction.json','prediction.npz')},code_sha256=code_hashes(),prior_sha256=prior_sha,future_class1_used=False,future_class2_used=False,post_seed_raw_rail_points_used=False))
        messages.append((name,meta['status'],len(out.get('s',[]))))
    return key(run,i),messages

def main():
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['a','b','e','smoke']);ap.add_argument('--methods',nargs='+');ap.add_argument('--workers',type=int,default=6);ap.add_argument('--limit',type=int);a=ap.parse_args()
    check_dependencies();protocol=load(OUT/'protocol.json')
    if not (OUT/'configs.json').exists():save(OUT/'configs.json',configurations())
    if a.phase=='e':
        lock=check_freeze();names=lock['methods'];rows=protocol['cohort']['heldout']
        assert a.methods is None and a.limit is None,'Final benchmark must use the complete frozen cohort and method list.'
    elif a.phase=='b':names=load(OUT/'phase_b_selection.json')['methods'];rows=protocol['cohort']['development']
    else:names=list(configurations());rows=protocol['screen']
    if a.methods:names=a.methods
    if a.limit:rows=rows[:a.limit]
    group={'a':'phase_a','b':'phase_b','e':'phase_e','smoke':'audit/smoke'}[a.phase]
    tasks=[(r,group,names,8) for r in rows];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        ff=[pool.submit(one,t) for t in tasks]
        for j,f in enumerate(as_completed(ff),1):
            result=f.result();print('RAIL INFERENCE',group,j,len(ff),result[0],result[1][0],round(time.perf_counter()-start,1),flush=True)
    save(OUT/group/'INFERENCE_COMPLETE.json',dict(time_ns=time.time_ns(),methods=names,starts=rows,code_sha256=code_hashes(),config_sha256=sha(OUT/'configs.json')))

if __name__=='__main__':main()
