"""Local Linux source AND diagnostics; export archive after measured replay."""
import argparse,os,subprocess,time,json,tarfile,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--count',type=int,default=400);p.add_argument('--workers',type=int,default=1);p.add_argument('--threads',type=int,default=4);p.add_argument('--kd-workers',type=int,default=1);p.add_argument('--voxel',default='packed');p.add_argument('--original',action='store_true');a=p.parse_args()
assert a.name.replace('_','').isalnum()
stride=int(os.environ.get('COPY_SOLVE_STRIDE','1'));offset=int(os.environ.get('COPY_SOLVE_OFFSET','0'))
assert stride>=1 and 0<=offset<stride
dest=Path('/work/results')/a.name;local=Path('/tmp/trials')/a.name
assert not dest.exists() and not local.exists();dest.mkdir(parents=True)
env=dict(os.environ,COPY_RUNTIME_ROOT='/app' if a.original else '/candidate',COPY_VOXEL=a.voxel,COPY_WORKERS=str(a.workers),COPY_THREADS=str(a.threads),COPY_KD_WORKERS=str(a.kd_workers))
entry='replay_direction.py' if os.environ.get('COPY_IMMEDIATE_DIRECTION')=='1' else 'replay.py'
if os.environ.get('COPY_ASYNC_DIRECTION')=='1':entry='replay_direction_async.py'
command=['python','-B','-u','/work/adapter/'+entry,'--db','/tmp/input.db3','--output',str(local),'--registration-source','/work/registration','--freeze','/freeze','--count',str(a.count)]
with (dest/'progress.log').open('w') as log:
    process=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    for line in process.stdout:log.write(line);log.flush();print(line.rstrip(),flush=True)
    rc=process.wait()
assert rc==0,rc
subprocess.run(['python','-B','/work/summarize.py',str(local)],check=True)
for name in ('SUMMARY.json','STAGES_MS.json','COPY_IO_TIMINGS.json'):
    if (local/name).exists():shutil.copyfile(local/name,dest/name)
t=time.perf_counter()
with tarfile.open(dest/'results.tar.gz','w:gz',compresslevel=1) as archive:archive.add(local,arcname='payload')
(dest/'EXPORT.json').write_text(json.dumps(dict(archive='results.tar.gz',post_run_export_ms=(time.perf_counter()-t)*1000,local_path=str(local),las_count=len(list(local.rglob('*.las'))),command=command,parameters=vars(a),registration_acceleration=env.get('COPY_REGISTRATION_ACCEL','off')),indent=2)+'\n')
print('TRIAL_EXPORTED',a.name,flush=True)
