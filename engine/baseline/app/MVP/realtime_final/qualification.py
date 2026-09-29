"""Final locked selection, one fresh 240-start inference and post-inference evaluation."""
from . import ROOT,OUT
from .memory_engine import MemoryEngine,SharedRing
from .bench import run
import long_common as lc
import numpy as np
import argparse,time,shutil,os,platform,hashlib

def save_result(folder,result,row):
    folder.mkdir(parents=True,exist_ok=True);r=result;c=r['c4'];steps=[s for s in c.get('steps',[]) if 'anchor_3d' in s and 'basis' in s]
    row.update(backend='full_native',seed=r['seed'],c4_seed={k:v for k,v in c['seed'].items() if k not in ('indices','diagnostics')})
    if r['prediction'] is not None:np.savez_compressed(folder/'full_native.npz',**r['prediction'])
    np.savez_compressed(folder/'full_native_c4.npz',curve=c.get('curve',np.empty((0,3))),anchors=np.array([s['anchor_3d'] for s in steps]),basis=np.array([s['basis'] for s in steps]),ranks=np.array([s.get('selected_candidate_rank',-1) for s in steps]),states=np.array([s['status'] for s in steps]),ranges=np.array([s.get('range_far',0) for s in steps]))
    lc.save(folder/'full_native.json',row)
def lock(workers,threads):
    assert all(c['c4_exact'] and c['final_exact'] for c in lc.load(OUT/'development.json')['equality'])
    assert len(lc.load(OUT/'development.json')['equality'])==200
    path=OUT/'REALTIME_SELECTION_LOCK.json'
    if path.exists():raise RuntimeError('Selection already locked')
    files={p.relative_to(ROOT).as_posix():lc.sha(p) for p in (ROOT/'MVP/realtime_final').rglob('*') if p.is_file() and p.suffix in ('.py','.cpp','.hpp','.so') and 'build' not in p.parts and 'clang' not in p.parts}
    lc.save(path,dict(time_ns=time.time_ns(),workers=workers,threads=threads,geometry='fresh_from_scratch',tight_exact_roi=True,selective_materialization=True,shared_history=12,pose_latency=1,code_sha256=files,selection_data='development only; 200 exact paired starts plus paced worker matrix',approximate_nn=False,thinning=False))
def inference():
    config=lc.load(OUT/'REALTIME_SELECTION_LOCK.json');original=lc.load(OUT/'ORIGINAL_SELECTION_LOCK.json');dest=OUT/'linux_final';dest.mkdir(exist_ok=True)
    if (dest/'INFERENCE_COMPLETE.json').exists():raise RuntimeError('Final inference already completed; do not rerun held-out')
    baseline=lc.load(OUT/'linux_baseline/INFERENCE_COMPLETE.json');rows=[r for r in baseline['rows'] if r['backend']=='reference'];equal=[]
    for name in original['benchmark']:
        directory=lc.dataset()/name;e=MemoryEngine(config['threads']);e.start_run(directory);ring=SharedRing(e.records)
        for case in (c for c in original['cohort'] if c['run']==name):
            index=case['frame'];ring.prepare_from_files(directory,index);e.prepare(ring,index);r,row=run(e,index);folder=dest/'benchmark'/name/f'{index:06d}';save_result(folder,r,row);rows.append(row)
            old=OUT/'linux_baseline/benchmark'/name/f'{index:06d}'
            for p in old.glob('reference*'):shutil.copy2(p,folder/p.name)
            oldp=old/'full_native.npz';newp=folder/'full_native.npz'
            exact=oldp.exists()==newp.exists()
            if exact and newp.exists():
                with np.load(oldp) as a,np.load(newp) as b:exact=all(np.array_equal(a[k],b[k],equal_nan=True) for k in ('C','B','pair','s','q'))
            equal.append(dict(run=name,frame=index,final_exact_vs_linux_baseline=exact))
            lc.save(folder/'PREDICTION_COMPLETE.json',dict(time_ns=time.time_ns(),uses_annotation_labels=False,uses_future_clouds=False,persistent_c4=False,files={p.name:lc.sha(p) for p in folder.iterdir() if p.suffix in ('.json','.npz') and p.name!='PREDICTION_COMPLETE.json'}))
            print('FINAL_240',name,index,round(row['wall_ms'],1),exact,flush=True)
    lc.save(dest/'INFERENCE_COMPLETE.json',dict(time_ns=time.time_ns(),rows=rows,reference_source='unchanged Linux baseline, exact same parent image/reference binary and frozen code',platform=platform.platform()))
    lc.save(dest/'FULL_NATIVE_SELECTION_LOCK.json',original);lc.csv_write(OUT/'final_vs_linux_baseline.csv',equal)
def evaluate():
    from MVP.full_native_final import evaluate as ev
    dest=OUT/'linux_final';assert (dest/'INFERENCE_COMPLETE.json').exists();(dest/'contact_gt').mkdir(exist_ok=True)
    for p in (OUT/'linux_baseline/contact_gt').glob('*.npz'):shutil.copy2(p,dest/'contact_gt'/p.name)
    ev.OUT=dest;ev.main();shutil.copy2(dest/'range_quality.csv',OUT/'linux_quality_240.csv');lc.save(OUT/'FINAL_LINUX_QUALITY.json',lc.load(dest/'QUALITY.json'))
def performance(mode):
    original=lc.load(OUT/'ORIGINAL_SELECTION_LOCK.json');config=lc.load(OUT/'REALTIME_SELECTION_LOCK.json');rows=[]
    for name in original['benchmark']:
        directory=lc.dataset()/name;e=MemoryEngine(config['threads']);e.start_run(directory);ring=SharedRing(e.records)
        for case in (c for c in original['performance_cohort'] if c['run']==name):
            t=time.perf_counter();n=len(ring.arrivals);io=ring.prepare_from_files(directory,case['frame']);prep=time.perf_counter()-t;e.prepare(ring,case['frame']);r,row=run(e,case['frame'])
            row.update(mode=mode,offline_full_ms=prep*1000+row['wall_ms'],las_read_ms=io*1000,arrival_prepare_ms=prep*1000,frames_built=len(ring.arrivals)-n,index_ms=sum(x['index_ms'] for x in ring.arrivals[n:]),source_metadata_ms=sum(x['metadata_ms'] for x in ring.arrivals[n:]));rows.append(row)
            print('PERF48',mode,name,case['frame'],round(row['offline_full_ms'],1),round(row['wall_ms'],1),flush=True)
    lc.save(OUT/('performance_'+mode+'.json'),dict(rows=rows,platform=platform.platform(),time_ns=time.time_ns()))
def stage_volume():
    original=lc.load(OUT/'ORIGINAL_SELECTION_LOCK.json');dest=__import__('pathlib').Path('/benchdata')
    from fusion import CausalSource
    from MVP.final_pipeline.pipeline import Pipeline
    for name in original['benchmark']:
        src=lc.dataset()/name;target=dest/name;target.mkdir(exist_ok=True);e=Pipeline();e.start_run(src)
        for p in src.glob('*.frames.json'):shutil.copy2(p,target/p.name)
        frames=set()
        for c in (c for c in original['performance_cohort'] if c['run']==name):frames.update(CausalSource(name,c['frame'],e.records).history(dict(frames=12))[0])
        for i in sorted(frames):shutil.copy2(src/e.records[i]['file'],target/e.records[i]['file'])
        print('EXT4_STAGED',name,len(frames),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['lock','inference','evaluate','performance','stage_volume']);p.add_argument('--workers',type=int,default=4);p.add_argument('--threads',type=int,default=1);p.add_argument('--mode',default='bind');a=p.parse_args()
    if a.action=='lock':lock(a.workers,a.threads)
    elif a.action=='inference':inference()
    elif a.action=='evaluate':evaluate()
    elif a.action=='performance':performance(a.mode)
    else:stage_volume()
if __name__=='__main__':main()
