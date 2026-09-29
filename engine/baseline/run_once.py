"""One self-contained run; Docker --rm deletes the copied DB on exit."""
import argparse,hashlib,json,shutil,subprocess,time,sys,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--count',type=int,default=1510);p.add_argument('--solve-stride',type=int,default=1);p.add_argument('--solve-offset',type=int,default=0);p.add_argument('--input-stride',type=int,choices=(1,2),default=1);p.add_argument('--immediate-direction',action='store_true');p.add_argument('--async-direction',action='store_true');a=p.parse_args()
assert a.solve_stride>=1 and 0<=a.solve_offset<a.solve_stride
os.environ['COPY_SOLVE_STRIDE']=str(a.solve_stride);os.environ['COPY_SOLVE_OFFSET']=str(a.solve_offset)
os.environ['COPY_INPUT_STRIDE']=str(a.input_stride)
os.environ['COPY_IMMEDIATE_DIRECTION']='1' if a.immediate_direction else '0'
os.environ['COPY_ASYNC_DIRECTION']='1' if a.async_direction else '0'
cfg=json.loads(Path('/work/config.json').read_text());assert cfg['export_las'] is False
os.environ['COPY_REGISTRATION_ACCEL']=os.environ.get('COPY_REGISTRATION_ACCEL',cfg.get('registration_acceleration','off') if a.input_stride==2 else 'off')
source=Path('/source/cloud_with_fake_obj_0.db3');target=Path('/tmp/input.db3');begin=time.perf_counter();shutil.copyfile(source,target);copied=time.perf_counter()
h=hashlib.sha256()
with target.open('rb') as f:
    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
assert h.hexdigest()==cfg['input_sha256']
metadata=dict(bytes=target.stat().st_size,sha256=h.hexdigest(),copy_ms=(copied-begin)*1000,checksum_ms=(time.perf_counter()-copied)*1000,copy_excluded_from_replay=True,warm_cache=True)
print('COPY_MAIN_INPUT_VERIFIED',json.dumps(metadata),flush=True)
try:
    subprocess.run([sys.executable,'-B','-u','/work/run_trial.py',a.name,'--count',str(a.count),'--workers',str(cfg['workers']),'--threads',str(cfg['candidate_threads']),'--kd-workers',str(cfg['icp_query_workers']),'--voxel',cfg['voxel']],check=True)
finally:
    destination=Path('/work/results')/a.name;destination.mkdir(parents=True,exist_ok=True);(destination/'INPUT_COPY.json').write_text(json.dumps(metadata,indent=2)+'\n')
