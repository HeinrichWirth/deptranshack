"""Container-only ABI, captured kernel and real current-cloud integration check."""
from .engine import Engine
from .native_backend import load_native
from .ablation import fingerprint
from MVP.final_pipeline.spatial_scheduler import SpatialScheduler
import long_common as lc
import numpy as np
from scipy.spatial import cKDTree
from pathlib import Path
import shutil,platform,sys,json


def main():
    assert all(shutil.which(x) is None for x in ('gcc','g++','cmake','make'))
    native=load_native();count=0;maximum=0.
    for path in Path('/fixtures').glob('fit_*.npz'):
        with np.load(path) as z:
            p=z['points'];a=z['center'];t=z['template'];x=cKDTree(t).query(p-a)[0];y=native.distances(p,a,t)
            maximum=max(maximum,float(np.max(abs(x-y))) if len(x) else 0.);count+=1
    assert maximum<=1e-10 and count>0
    hashes=[];timings=[]
    for backend in ('spatial','native_spatial'):
        e=Engine(backend);e.start_run('/data');result=e.process_frame(0)
        assert result['prediction'] is not None and result['summary']['source_frames']==[0]
        hashes.append(fingerprint(result));timings.append(result['summary']['timing'])
    assert hashes[0]==hashes[1],'Linux Python/native C4 result differs'
    scheduler=SpatialScheduler(Engine('spatial'));scheduler.start_run('/data')
    current=scheduler.process_frame(0)
    assert current['summary']['fresh'] and fingerprint(current['fresh_result'])==hashes[0]
    snapshot=scheduler.geometry_for_pose(scheduler.engine.records[1],1)
    assert snapshot is not None and snapshot['source_frame']==0 and snapshot['geometry_age_frames']==1
    lc.save('/out/docker_smoke.json',dict(platform=platform.platform(),python=sys.version,kernels=count,
        max_kernel_difference=maximum,full_RAW_frame_python_native_bitwise=True,future_cloud_absent=True,
        no_build_tools_in_runtime=True,scheduler_snapshot_passed=True,output_hash=hashes[0],timings=timings,
        compiler=Path('/app/compiler.txt').read_text(),cmake=Path('/app/cmake.txt').read_text()))
    print('DOCKER_SMOKE_OK',flush=True)


if __name__=='__main__':main()
