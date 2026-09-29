"""Fresh causal registration to the recovery event; fixed-work microbenchmarks.

Not a live replay. Saved poses are read only after inference for comparison.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['COPY_VOXEL']='packed'
import sys,json,time,sqlite3,hashlib,shutil
from pathlib import Path
import numpy as np
sys.path.insert(0,'/work/adapter')
from causal_registration import CausalRegistration,original_functions
from registration_acceleration import install,install_parallel_recovery
from cdr_cloud import decode
from concurrent.futures import ThreadPoolExecutor

out=Path('/work/results')/os.environ.get('HOTSPOT_RUN','REGISTRATION_HOTSPOTS');out.mkdir(exist_ok=False)
src=Path('/source/cloud_with_fake_obj_0.db3');db=Path('/tmp/input.db3')
begin=time.perf_counter();shutil.copyfile(src,db)
h=hashlib.sha256()
with db.open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
assert h.hexdigest()=='73ee5bfc2087bb160267110563b298a9629ac9e3df1459d857ec72ce1ef89d60'
print('INPUT_READY',round(time.perf_counter()-begin,2),flush=True)
reg=CausalRegistration('/work/registration')
cache=install(reg.fn,'/work/registration',bounded=False,profile=True,cache_size=0)
cases=[];trial_events=[];poses=[];rows=[]
old_recovery=reg.fn['recover_endpoint']
def recovery(source,predicted,references):
    cases.append((source.copy(),predicted.copy(),references[:]))
    return old_recovery(source,predicted,references)
reg.fn['recover_endpoint']=recovery
old_trial=reg.fn['recovery_icp']
def trial(*args,**kwargs):
    begin=time.perf_counter();start=len(cache.events)
    result=old_trial(*args,**kwargs)
    trial_events.append(dict(ms=(time.perf_counter()-begin)*1000,metric=result[1],nn_ms=sum(e['ms'] for e in cache.events[start:] if e['kind']=='query'),target_points=len(args[1]),source_points=min(len(args[0]),12000)))
    return result
reg.fn['recovery_icp']=trial
con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True)
schedule=con.execute('SELECT id FROM messages ORDER BY timestamp,id LIMIT 234').fetchall()[::2]
ordinary=[]
for index,(message_id,) in enumerate(schedule):
    meta,points=decode(con.execute('SELECT data FROM messages WHERE id=?',(message_id,)).fetchone()[0]);xyz=np.column_stack([points[k] for k in ('x','y','z')])
    start=len(cache.events);begin=time.perf_counter();pose=reg.step(xyz,index,meta['header_time_ns']);poses.append(pose)
    rows.append(dict(source_frame=index*2,ms=(time.perf_counter()-begin)*1000,detail=reg.last_timing_ms.copy(),queries=cache.events[start:]))
    if index%25==0:print('PROFILE',index*2,round(rows[-1]['ms'],2),flush=True)
assert len(cases)==1
reference=[json.loads(x) for x in Path('/work/results/INPUT5HZ_SOLVE2_POSE1_ASYNC/payload/poses.jsonl').read_text().splitlines()]
for i,p in enumerate(poses):assert p['pose']==reference[i]['pose']
source,predicted,references=cases[0]
bench=[];expected=None
for bounded,reuse,parallel in ((False,False,False),(False,True,False),(True,True,False),(True,True,True)):
    for repeat in range(3):
        scope,_=original_functions('/work/registration');c=install(scope,'/work/registration',bounded=bounded,profile=True,cache_size=16 if reuse else 0)
        if parallel:install_parallel_recovery(scope)
        begin=time.perf_counter();result=scope['recover_endpoint'](source,predicted,references);ms=(time.perf_counter()-begin)*1000
        if expected is None:expected=result
        np.testing.assert_array_equal(result[0],expected[0]);assert result[1]==expected[1] and result[-1]==expected[-1]
        bench.append(dict(bounded=bounded,reuse=reuse,parallel=parallel,repeat=repeat,ms=ms,query_ms=sum(e['ms'] for e in c.events if e['kind']=='query'),build_ms=sum(e['ms'] for e in c.events if e['kind']=='build')))
        print('BENCH',bench[-1],flush=True)
result=dict(input_sha256=h.hexdigest(),profiled_poses_exact=len(poses),recovery_trials=trial_events,frames=rows,recovery_bench=bench,mode='fixed causal work; no live timing claim')
(out/'PROFILE.json').write_text(json.dumps(result,indent=2)+'\n')
print('PROFILE_COMPLETE',flush=True)
if os.environ.get('HOTSPOT_REPLAY')=='1':
    import subprocess
    env=dict(os.environ,COPY_REGISTRATION_ACCEL='parallel',COPY_ASYNC_DIRECTION='1',COPY_IMMEDIATE_DIRECTION='1',COPY_SOLVE_STRIDE='2',COPY_SOLVE_OFFSET='1',COPY_INPUT_STRIDE='2')
    subprocess.run([sys.executable,'-B','-u','/work/run_trial.py','INPUT5HZ_SOLVE2_REGOPT','--count','1510','--workers','2','--threads','2','--kd-workers','1','--voxel','packed'],env=env,check=True)
