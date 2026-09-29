from . import OUT
from .native_api import startup
from .micro import clock
from MVP.final_pipeline.frame_data import FrameData
import long_common as lc
import numpy as np
from scipy.spatial import cKDTree
import time

def main():
    native=startup();rows=[]
    for case in lc.load(OUT/'benchmark_cohort.json')['clean'][::8]:
        records=lc.frames(case['run']);raw=FrameData.read(lc.dataset()/case['run']/records[case['frame']]['file'],case['frame']);points=raw.world
        tic=time.perf_counter();tree=cKDTree(points);treebuild=time.perf_counter()-tic
        tic=time.perf_counter();grid=native.Grid(points,2.);gridbuild=time.perf_counter()-tic
        # Real source coordinates, uniformly selected points; no class labels.
        for ordinal in np.linspace(0,len(points)-1,12,dtype=int):
            center=points[ordinal];radius=float(np.hypot(2.,.9))
            def kd():return np.asarray(sorted(tree.query_ball_point(center,radius)),dtype=np.int64)
            def hashed():
                ids=grid.query(center,radius);inside=cKDTree(points[ids]).query_ball_point(center,radius)
                return ids[np.asarray(sorted(inside),dtype=int)]
            def brute():
                d=points-center
                # Conservative brute box then original exact Euclidean refinement.
                ids=np.flatnonzero(np.all(abs(d)<=radius+128*np.finfo(float).eps*(1+abs(center).max()),axis=1))
                inside=cKDTree(points[ids]).query_ball_point(center,radius)
                return ids[np.asarray(sorted(inside),dtype=int)]
            expected=kd();np.testing.assert_array_equal(hashed(),expected);np.testing.assert_array_equal(brute(),expected)
            rows.append(dict(**case,point=int(ordinal),source_points=len(points),roi_points=len(expected),radius=radius,
                kdtree_us=clock(kd,3)*1e6,grid_us=clock(hashed,3)*1e6,brute_us=clock(brute,3)*1e6,
                tree_build_ms=treebuild*1000,grid_build_ms=gridbuild*1000,grid_bytes=grid.bytes(),exact=True))
    lc.csv_write(OUT/'spatial_benchmark.csv',rows)

if __name__=='__main__':main()
