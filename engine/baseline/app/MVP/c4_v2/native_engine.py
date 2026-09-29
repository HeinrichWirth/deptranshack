"""One native marching call; Python performs I/O and frozen downstream stages."""
from .native_api import startup
from MVP.performance_final.engine import Engine as Base,clone
from types import MethodType
import numpy as np
import time


def infer_native(owner,source,seed,template,cfg):
    source.reset_audit();begin=time.perf_counter()
    if seed['status']!='AVAILABLE':
        return dict(status=seed['status'],reason=seed.get('reason',''),seed=seed,config=cfg,steps=[],hypotheses=[],indices=np.empty(0,dtype=np.int64),point_step=np.empty(0,dtype=np.uint16),point_state=np.empty(0,dtype=np.uint8),total_ms=0.),source.materialize()
    assert cfg['partial']=='subset' and cfg['prior']=='P0' and cfg['geometry']=='T'
    history=source.history(cfg,12)
    for k in sorted(history):
        if k not in owner.native_frames:
            raw=source.source.cache.get(k,source.i)
            owner.marcher.add_frame(k,raw.world);owner.native_frames.add(k)
    owner.native_frames={k for k in owner.native_frames if k>=source.i-11}
    P=np.asarray(source.source.past[source.i]['lidar_pose_in_folder'])
    out=owner.marcher.extend_track(np.asarray(seed['anchor']),np.asarray(seed['basis']),np.asarray(seed['indices'],dtype=np.int64),
        np.asarray(template)*[seed['side'],1],P,np.linalg.inv(P[:3,:3]),source.i,history,float(seed['confidence']))
    blocks=out['blocks'];keys=np.concatenate([np.asarray(b['keys'],dtype=np.uint64) for b in blocks])
    for k in sorted(set((keys>>np.uint64(32)).tolist())):
        if k!=source.i:source.used[k]=set((keys[(keys>>np.uint64(32))==k]&np.uint64(0xffffffff)).tolist())
    cloud=source.materialize();order=np.argsort(cloud['keys']);ix=order[np.searchsorted(cloud['keys'][order],keys)]
    assert np.array_equal(cloud['keys'][ix],keys) and len(np.unique(keys))==len(keys)
    arr=lambda f:np.concatenate([np.asarray(b[f]) for b in blocks])
    rep=lambda f,dtype=float:np.concatenate([np.full(len(b['keys']),b[f],dtype=dtype) for b in blocks])
    steps=[]
    for b in blocks[1:]:
        mask=np.isin(cloud['keys'],np.asarray(b['keys'],dtype=np.uint64));ranges=np.linalg.norm(cloud['xyz'][mask],axis=1)
        steps.append(dict(step_index=b['step'],status='ACCEPTED' if b['state']==2 else 'TENTATIVE',state='CONFIRMED' if b['state']==2 else 'TENTATIVE',
            original_state='TENTATIVE',anchor_3d=np.asarray(b['anchor_3d']),basis=np.asarray(b['basis_columns']).T,plane_origin=np.asarray(b['plane_origin']),
            window=8,history=12,anchor_v=b['anchor'][0],anchor_w=b['anchor'][1],new_support_n=len(b['keys']),selected_candidate_rank=b['rank'],
            confidence=b['confidence'],continuity_score=b['continuity_score'],template_score=b['template_score'],range_far=float(ranges.max()),range_near=float(ranges.min())))
    state=rep('state',np.uint8);ranges=np.linalg.norm(cloud['xyz'][ix],axis=1)
    pred=dict(status='STOPPED',reason=out['reason'],seed=seed,config=cfg,steps=steps,hypotheses=[],indices=ix,
        point_step=rep('step',np.uint16),point_state=state,local_uvw=arr('uvw'),template_residual=arr('residual'),
        confidence=rep('confidence'),continuity_score=rep('continuity_score'),template_score=rep('template_score'),track_prior_score=rep('track_prior_score'),
        point_anchor=np.concatenate([np.repeat(np.asarray(b['anchor'])[None],len(b['keys']),0) for b in blocks]),
        s_from_seed=np.concatenate([4*b['step']+np.asarray(b['uvw'])[:,0] for b in blocks]),curve=np.asarray([b['anchor_3d'] for b in blocks if b['state']==2]),
        max_search_hypothesis_range=out['search_range'],max_selected_search_range=out['search_range'],max_tentative_range=float(ranges[state==1].max()) if np.any(state==1) else 0.,
        max_confirmed_observed_range=float(ranges[state==2].max()),max_unobserved_length=0,path_score=out['path_score'],total_ms=(time.perf_counter()-begin)*1000)
    return pred,cloud


class NativeEngine(Base):
    def __init__(self,threads=1,iterations=12,grid=0):
        super().__init__('spatial');self.marcher=startup().Marcher(threads,iterations,grid);self.native_frames=set()
        self.process_frame=MethodType(clone(self.process_frame.__func__,infer=lambda *args:infer_native(self,*args)),self)
    def close(self): pass
