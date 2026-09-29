from MVP.performance_final.spatial_cache import CachedSpatialRoi
from .native_api import startup
import numpy as np
from scipy.spatial import cKDTree

class GridRoi(CachedSpatialRoi):
    def sphere(self,k,C,B,W):
        cache=self.source.cache;p=self.read(k)
        if not hasattr(cache,'grids'):cache.grids={}
        for old in list(cache.grids):
            if old<self.i-11:del cache.grids[old]
        if k not in cache.grids:cache.grids[k]=startup().Grid(cache.get(k,self.i).world,2.)
        P=np.asarray(self.source.past[self.i]['lidar_pose_in_folder']);inv=np.linalg.inv(P[:3,:3])
        center=C+B[:,0]*(W/2);radius=np.hypot(W/2,.9);world=center@inv+P[:3,3];bound=radius*np.linalg.norm(inv,2)
        guard=128*np.finfo(float).eps*(1+np.linalg.norm(world)+np.linalg.norm(P[:3,3])+bound)
        broad=cache.grids[k].query(world,bound+guard)
        exact=cKDTree(p['xyz'][broad]).query_ball_point(center,radius);ids=broad[np.asarray(sorted(exact),dtype=np.int64)]
        self.processed+=len(ids)
        return p,ids,(p['xyz'][ids]-C)@B
