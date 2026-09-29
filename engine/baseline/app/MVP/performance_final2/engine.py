from MVP.performance_final.engine import Engine as Previous,clone
from MVP.performance_final.spatial_cache import CachedSpatialRoi
from .tracker_backend import factory
import long_tracker
from types import MethodType

class Engine(Previous):
    def __init__(self,backend='native_batch_spatial',workers=1):
        if backend=='native_batch_spatial':backend='batch'
        if backend=='grid_native':
            super().__init__('native_spatial');self.trackers=[];self.backend=backend
            from .grid import GridRoi
            self.process_frame=MethodType(clone(self.process_frame.__func__,RawRoi=GridRoi),self)
            return
        if backend in ('python','optimized','spatial','native','native_spatial'):
            super().__init__(backend);self.trackers=[];return
        super().__init__('spatial');self.trackers=[];self.backend=backend
        tt=factory(backend,workers,self.trackers)
        if backend in ('batch','threads'):
            from .batching import make_infer
            infer=make_infer(tt)
        else:infer=clone(long_tracker.infer,TemplateTracker=tt)
        overrides=dict(infer=infer)
        if backend=='grid':
            from .grid import GridRoi
            overrides['RawRoi']=GridRoi
        self.process_frame=MethodType(clone(self.process_frame.__func__,**overrides),self)
        self.invoke=self.process_frame
        def process(index):
            self.close()
            return self.invoke(index)
        self.process_frame=process
    def close(self):
        for tracker in self.trackers:tracker.close()
        self.trackers.clear()
