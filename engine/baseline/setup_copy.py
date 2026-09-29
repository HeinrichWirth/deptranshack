"""One-time setup/instrumentation of COPY_MAIN only."""
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
manifest=json.loads((HERE/'ORIGINAL_RELEASE_MANIFEST.json').read_text())
checks={p:hashlib.sha256((HERE/'app'/p).read_bytes()).hexdigest()==v['sha256'] for p,v in manifest['runtime_files'].items()}
assert all(checks.values()),checks
(HERE/'COPY_ORIGIN.json').write_text(json.dumps(dict(source_image='deptrans-realtime:production',runtime_files_exact=checks,source_version=manifest['version']),indent=2)+'\n')
p=HERE/'adapter/replay.py';s=p.read_text()
s=s.replace("sys.path.insert(0,'/app')","sys.path.insert(0,os.environ.get('COPY_RUNTIME_ROOT','/candidate'))")
s=s.replace("from MVP.realtime_final.production import create_scheduler","from MVP.realtime_final.live import Scheduler\ndef create_scheduler(directory):return Scheduler(directory,workers=int(os.environ.get('COPY_WORKERS','1')),threads=int(os.environ.get('COPY_THREADS','2')))")
s=s.replace("pose=reg.step(xyz,index,meta['header_time_ns']);registered=time.perf_counter()","hashed=time.perf_counter();pose=reg.step(xyz,index,meta['header_time_ns']);registered=time.perf_counter()")
s=s.replace("registration_ms=(registered-decoded)*1000,", "payload_hash_ms=(hashed-decoded)*1000,registration_ms=(registered-hashed)*1000,registration_detail_ms=dict(reg.last_timing_ms),")
s=s.replace("source_stat=args.db.stat();source_digest=hashlib.sha256()", "source_stat=args.db.stat();source_digest=hashlib.sha256()\n    candidate_files={str(p):sha(p) for p in Path(os.environ.get('COPY_RUNTIME_ROOT','/candidate')).rglob('*') if p.is_file()};save(out/'CANDIDATE_SOURCE.json',candidate_files)")
s=s.replace("    summary['realtime_10hz_deadline_pass']", "    assert all(sha(Path(p))==digest for p,digest in candidate_files.items())\n    summary['copy_main']=dict(workers=int(os.environ.get('COPY_WORKERS','1')),threads=int(os.environ.get('COPY_THREADS','2')),runtime_root=os.environ.get('COPY_RUNTIME_ROOT'),voxel=os.environ.get('COPY_VOXEL','original'),las_export=False)\n    summary['realtime_10hz_deadline_pass']")
p.write_text(s)
p=HERE/'adapter/causal_registration.py';s=p.read_text();s=s.replace('import ast,hashlib','import ast,hashlib,time,os')
s=s.replace('        self.pose=np.eye(4);', '''        self.last_timing_ms={}
        if os.environ.get('COPY_VOXEL')=='native':
            from copy_voxel import voxel_sample
            self.fn['voxel_sample']=voxel_sample
        for name in ('voxel_sample','normals','icp','recover_endpoint'):
            original=self.fn[name]
            def measured(*args,_name=name,_fn=original,**kwargs):
                begin=time.perf_counter()
                try:return _fn(*args,**kwargs)
                finally:self.last_timing_ms[_name]=self.last_timing_ms.get(_name,0.)+(time.perf_counter()-begin)*1000
            self.fn[name]=measured
        self.pose=np.eye(4);''')
s=s.replace('    def step(self,xyz,index,stamp):','''    def step(self,xyz,index,stamp):
        self.last_timing_ms={};begin=time.perf_counter()
        result=self._step(xyz,index,stamp)
        self.last_timing_ms['total']=(time.perf_counter()-begin)*1000
        self.last_timing_ms['other']=self.last_timing_ms['total']-sum(self.last_timing_ms.get(k,0.) for k in ('voxel_sample','normals','recover_endpoint' if 'recover_endpoint' in self.last_timing_ms else 'icp'))
        return result
    def _step(self,xyz,index,stamp):''')
p.write_text(s)
print('COPY_CREATED',len(checks),'exact frozen runtime files')
