from . import ROOT,OUT
from .engine import Engine
from MVP.performance_final.engine import clone
from MVP.performance_final.hotspots import Profile
from MVP.performance_final.profile_c4 import COHORT
from MVP.performance_final.spatial_cache import CachedSpatialRoi
from MVP.final_pipeline.pipeline import RawRoi
import long_common as lc
import tracker,long_tracker,roi_source
from scipy.spatial import cKDTree
import numpy as np
import inspect,hashlib,time,cProfile,pstats,json
from types import MethodType
from collections import OrderedDict

def main():
    best=lc.load(OUT/'selection.json')['backend'];rows=[];builds=[];profiles=[]
    (OUT/'profiles').mkdir(exist_ok=True)
    for backend in ('python','native_spatial',best):
        for run,index in COHORT:
            e=Engine(backend);e.start_run(lc.dataset()/run);tree_counts=[]
            def tree(points,*args,**kwargs):
                t=time.perf_counter();a=np.asarray(points);digest=hashlib.sha256(a.tobytes()).hexdigest();hash_dt=time.perf_counter()-t
                frame=inspect.currentframe().f_back;context={}
                for _ in range(8):
                    if frame is None:break
                    for name in ('k','step','parent_id','fit_id'):
                        if name in frame.f_locals and isinstance(frame.f_locals[name],int):context[name]=frame.f_locals[name]
                    obj=frame.f_locals.get('self')
                    if hasattr(obj,'next_fit'):context['fit_id']=obj.next_fit
                    if frame.f_code.co_name in ('candidate_records','sphere','reverse','prepare','fit'):context.setdefault('role',frame.f_code.co_name)
                    frame=frame.f_back
                t=time.perf_counter();obj=cKDTree(a,*args,**kwargs);dt=time.perf_counter()-t
                tree_counts.append(dict(run=run,frame=index,backend=backend,call_id=len(tree_counts),object_id=id(points),pointer=a.__array_interface__['data'][0],
                    shape=list(a.shape),content_hash=digest,bytes=a.nbytes,hash_seconds=hash_dt,build_seconds=dt,**context))
                return obj
            # Only instance-local cloned globals or this sprint's own module.
            if backend=='python':
                class TT(tracker.TemplateTracker):
                    __init__=clone(tracker.TemplateTracker.__init__,cKDTree=tree)
                    fit=clone(tracker.TemplateTracker.fit,cKDTree=tree)
                class Roi(RawRoi):sphere=clone(roi_source.RoiSource.sphere,cKDTree=tree)
                e.process_frame=MethodType(clone(e.process_frame.__func__,infer=clone(long_tracker.infer,TemplateTracker=TT),RawRoi=Roi),e)
            elif backend=='native_spatial':
                from MVP.performance_final import hotspots
                base=hotspots.make_tracker('native',None)
                class TT(base):
                    def __init__(self,*a,**kw):
                        clone(tracker.TemplateTracker.__init__,cKDTree=tree)(self,*a,**kw)
                        self.memo=OrderedDict();self.hits=0
                    fit=clone(base.fit,optimized_fit=clone(hotspots.optimized_fit,cKDTree=tree))
                class Roi(CachedSpatialRoi):sphere=clone(CachedSpatialRoi.sphere,cKDTree=tree)
                e.process_frame=MethodType(clone(e.process_frame.__func__,infer=clone(long_tracker.infer,TemplateTracker=TT),RawRoi=Roi),e)
            else:
                from . import tracker_backend
                original_tree=tracker_backend.cKDTree;tracker_backend.cKDTree=tree
                base_roi=e.invoke.__func__.__globals__['RawRoi']
                class Roi(base_roi):sphere=clone(base_roi.sphere,cKDTree=tree)
                e.invoke=MethodType(clone(e.invoke.__func__,RawRoi=Roi),e)
            cp=cProfile.Profile();cp.enable();r=e.process_frame(index);cp.disable()
            if backend not in ('python','native_spatial'):tracker_backend.cKDTree=original_tree
            file=OUT/'profiles'/f'{backend}__{run}__{index}.prof';cp.dump_stats(str(file))
            with file.with_suffix('.txt').open('w',encoding='utf-8') as stream:pstats.Stats(cp,stream=stream).sort_stats('cumulative').print_stats(55)
            for (path,line,fn),(cc,nc,tt,ct,callers) in pstats.Stats(cp).stats.items():
                rows.append(dict(backend=backend,run=run,frame=index,function=fn,file=path,line=line,calls=nc,self_seconds=tt,cumulative_seconds=ct))
            profiles.append(dict(backend=backend,run=run,frame=index,timing=r['summary']['timing'],stats=dict(e.trackers[-1].stats) if e.trackers else {}))
            builds.extend(tree_counts);e.close()
    lc.csv_write(OUT/'tree_reuse.csv',builds);lc.csv_write(OUT/'profile_functions.csv',rows);lc.save(OUT/'profiles/summary.json',profiles)
    lc.save(OUT/'profile_audit.json',dict(profiles=len(profiles),builds=len(builds),note='cProfile inclusive times overlap; instrumented runs are not latency benchmark'))

if __name__=='__main__':main()
