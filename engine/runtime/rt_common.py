"""Bounded shared transport; no point data is sent through a FIFO."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,pickle,hashlib
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(HERE/'baseline/adapter'),str(HERE/'baseline/postprocess'),str(HERE/'baseline/app')]

class Latest:
    """One replaceable pending message, sequence and replacement accounting."""
    def __init__(self,ctx,capacity=256*1024):
        self.buffer=ctx.RawArray('B',capacity);self.lock=ctx.Lock();self.length=ctx.Value('i',0,lock=False)
        self.version=ctx.Value('q',0,lock=False);self.ack=ctx.Value('q',0,lock=False);self.replaced=ctx.Value('q',0,lock=False)
    def put(self,value):
        data=pickle.dumps(value,protocol=5)
        if len(data)>len(self.buffer):raise ValueError('Mailbox payload exceeds fixed capacity')
        with self.lock:
            if self.version.value>self.ack.value:self.replaced.value+=1
            np.frombuffer(self.buffer,dtype=np.uint8,count=len(data))[:]=np.frombuffer(data,dtype=np.uint8)
            self.length.value=len(data);self.version.value+=1
    def take(self):
        with self.lock:
            if self.ack.value==self.version.value:return None
            data=bytes(memoryview(self.buffer).cast('B')[:self.length.value]);self.ack.value=self.version.value
        return pickle.loads(data)

class CloudRing:
    def __init__(self,ctx,slots,maximum):
        self.slots=slots;self.maximum=maximum
        self.data=ctx.RawArray('f',slots*maximum*4)
        self.tags=ctx.Array('q',[-1]*slots,lock=False);self.counts=ctx.Array('i',[0]*slots,lock=False)
        self.meta=[Latest(ctx,4096) for _ in range(slots)];self.locks=[ctx.Lock() for _ in range(slots)]
    def array(self):return np.frombuffer(self.data,dtype=np.float32).reshape(self.slots,self.maximum,4)
    def put(self,meta,points):
        n=len(points);i=meta['source_frame'];slot=i%self.slots
        if n>self.maximum:raise ValueError('Cloud exceeds declared capacity')
        # Locks protect a short memory copy, never an ICP or geometry computation.
        with self.locks[slot]:
            self.data_view=self.array() if not hasattr(self,'data_view') else self.data_view
            self.data_view[slot,:n]=points;self.counts[slot]=n;self.tags[slot]=i
            self.meta[slot].put(meta)
    def read(self,i):
        slot=i%self.slots
        with self.locks[slot]:
            if self.tags[slot]!=i:return None
            box=self.meta[slot]
            with box.lock:meta=pickle.loads(bytes(memoryview(box.buffer).cast('B')[:box.length.value]))
            data=self.array()[slot,:self.counts[slot]].copy()
        return meta,data

def log_open(out,name):return (Path(out)/(name+'.jsonl')).open('w',buffering=1)
def log(file,**row):file.write(json.dumps(row,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item(),allow_nan=False)+'\n')
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item())+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def age(meta):return (time.perf_counter()-meta['deadline'])*1000
def finite(points):return np.isfinite(points[:,:3]).all(axis=1)&np.any(points[:,:3]!=0,axis=1)
def record(meta,pose):
    return dict(file='LIVE_SHARED_MEMORY',header_time_ns=meta['header_time_ns'],lidar_pose_in_folder=pose,
                pose_status='ok',pose_uses_future=False,source_frame_index=meta['source_frame'])
