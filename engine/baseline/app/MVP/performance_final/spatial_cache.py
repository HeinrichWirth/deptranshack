"""Exact refinement after a conservative map-space KDTree broad phase.

The final radius membership is tested by the same SciPy KDTree on exactly the
same current-frame float64 XYZ. The map tree only supplies a superset; neither
search radius nor final ROI gates are changed. Trees expire with raw history.
"""
import numpy as np
from scipy.spatial import cKDTree
from MVP.final_pipeline.pipeline import RawRoi


class CachedSpatialRoi(RawRoi):
    def sphere(self,k,C,B,W):
        cache=self.source.cache
        if cache is None:return super().sphere(k,C,B,W)
        p=self.read(k)
        if not hasattr(cache,'world_trees'):cache.world_trees={}
        for old in list(cache.world_trees):
            if old<self.i-11:del cache.world_trees[old]
        raw=cache.get(k,self.i)
        if k not in cache.world_trees:cache.world_trees[k]=cKDTree(raw.world)
        P=np.asarray(self.source.past[self.i]['lidar_pose_in_folder']);inv=np.linalg.inv(P[:3,:3])
        center=C+B[:,0]*(W/2);radius=np.hypot(W/2,.9)
        world_center=center@inv+P[:3,3]
        bound=radius*np.linalg.norm(inv,2)
        # Broad-phase numerical guard only. Exact original radius is applied below.
        guard=128*np.finfo(float).eps*(1+np.linalg.norm(world_center)+np.linalg.norm(P[:3,3])+bound)
        broad=np.asarray(sorted(cache.world_trees[k].query_ball_point(world_center,bound+guard)),dtype=np.int64)
        exact=cKDTree(p['xyz'][broad]).query_ball_point(center,radius)
        ids=broad[np.asarray(sorted(exact),dtype=np.int64)]
        self.processed+=len(ids)
        return p,ids,(p['xyz'][ids]-C)@B
