"""Controlled +100k irrelevant RAW rows; diagnostic only, no detector parameter tuning."""
from . import ROOT,OUT
from .memory_engine import MemoryEngine,SharedRing
from .bench import run,same
from MVP.final_pipeline.frame_data import FrameData
from MVP.performance_final.direction import direction
from fusion import CausalSource
from types import MappingProxyType
import numpy as np,time
import long_common as lc

def main():
    cases=lc.load(ROOT/'results_performance_final2/benchmark_cohort.json')['clean'][::4];rows=[]
    for case in cases:
        directory=lc.dataset()/case['run'];setup=MemoryEngine(8);setup.start_run(directory);i=case['frame'];records=setup.records
        history=CausalSource(case['run'],i,records).history(dict(frames=12))[0]
        raw={k:FrameData.read(directory/records[k]['file'],k) for k in history};base=raw[i];B,_,_=direction(records,i);P=np.asarray(records[i]['lidar_pose_in_folder'])
        xyz=np.repeat((-10000*B[:,0])[None],100000,axis=0);world=xyz@np.linalg.inv(P[:3,:3])+P[:3,3]
        extra=lambda a:np.concatenate([a,np.zeros(100000,dtype=a.dtype)])
        point=np.concatenate([base.point_index,np.arange(int(base.point_index.max())+1,int(base.point_index.max())+100001,dtype=base.point_index.dtype)])
        more=FrameData(np.concatenate([base.world,world]),extra(base.ring),point,MappingProxyType({k:extra(v) for k,v in base.measurements.items()}),base.source_file,i,base.source_bytes)
        for a in (more.world,more.ring,more.point_index,*more.measurements.values()):a.setflags(write=False)
        for repeat in range(3):
            results={};metrics={}
            for variant in (('base','plus100k') if repeat%2==0 else ('plus100k','base')):
                e=MemoryEngine(8);e.start_run(directory);ring=SharedRing(records)
                for k in sorted(history):ring.arrive(more if k==i and variant=='plus100k' else raw[k])
                prep=ring.arrivals[-1];e.prepare(ring,i);r,row=run(e,i);results[variant]=r;metrics[variant]=row
                rows.append(dict(run=case['run'],frame=i,repeat=repeat,variant=variant,raw_points=row['source_points'],wall_ms=row['wall_ms'],c4_ms=row['c4_ms'],arrival_ms=prep['arrival_ms'],index_ms=prep['index_ms'],metadata_ms=prep['metadata_ms']))
            check=same(results['base'],results['plus100k']);assert check['c4_exact'] and check['final_exact']
        print('COUNT_SENSITIVITY',case,flush=True)
    lc.csv_write(OUT/'controlled_plus100k.csv',rows);lc.save(OUT/'controlled_plus100k.json',rows)
if __name__=='__main__':main()
