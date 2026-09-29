from . import ROOT,OUT
from MVP.c4_v2.native_engine import NativeEngine
import long_common as lc
import numpy as np
import time

CASES=[('doubleT_platform',0),('doubleT_platform',44),('doubleT_platform',88),('new_data_part_01a1',717),('new_data_part_01a1',358),('new_data_part_01c2',256),('new_data_part_01c2',640),('roundT_doubleT',212),('roundT_doubleT',6),('roundT_squareT_pressureGate_squareT',100)]

class Capture:
    def __init__(self,base,path):self.base=base;self.path=path;self.frames={}
    def __getattr__(self,k):return getattr(self.base,k)
    def add_frame(self,k,world):self.frames[k]=world;return self.base.add_frame(k,world)
    def extend_track(self,origin,basis,seed_ids,template,pose,inverse,index,history,confidence):
        data=dict(origin=origin,basis=basis,seed_ids=seed_ids,template=template,pose=pose,inverse=inverse,index=index,history=history,confidence=confidence)
        data.update({f'world_{k}':self.frames[k] for k in history});self.path.parent.mkdir(parents=True,exist_ok=True)
        np.savez(self.path,**data)
        return self.base.extend_track(origin,basis,seed_ids,template,pose,inverse,index,history,confidence)

def main():
    rows=[]
    for run,i in CASES:
        path=OUT/'captures'/f'{run}__{i:06d}.npz';e=NativeEngine(1,12,0);e.marcher=Capture(e.marcher,path);e.start_run(lc.dataset()/run);r=e.process_frame(i);e.close()
        assert path.exists();rows.append(dict(run=run,frame=i,file=path.name,sha256=lc.sha(path),bytes=path.stat().st_size,uses_labels=False,uses_future_cloud=False))
        print('CAPTURE',run,i,flush=True)
    lc.save(OUT/'captures/manifest.json',dict(rows=rows,time_ns=time.time_ns(),same_bits_on_all_platforms=True))

if __name__=='__main__':main()
