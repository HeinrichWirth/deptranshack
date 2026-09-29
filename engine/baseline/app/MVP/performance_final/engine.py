"""Instance-local function globals; frozen functions and files are not patched."""
from types import FunctionType,MethodType
from MVP.final_pipeline.pipeline import Pipeline
from .direction import direction


def clone(fn,**overrides):
    copied=FunctionType(fn.__code__,dict(fn.__globals__,**overrides),fn.__name__,fn.__defaults__,fn.__closure__)
    copied.__kwdefaults__=fn.__kwdefaults__
    return copied


class Engine(Pipeline):
    def __init__(self,backend='python',orientation=True,profile=None):
        super().__init__(dict(engine='cache'))
        self.backend=backend;self.direction_info={};self.orientation=orientation
        overrides={}
        if orientation:
            def pose_basis(a,b):
                B,reason,info=direction(self.records,self.last_index);self.direction_info=info
                return B,reason
            overrides['initial_frame']=pose_basis
        if backend!='python' or profile is not None:
            from .hotspots import make_infer,make_roi
            fit_backend='native' if backend=='native_spatial' else 'optimized' if backend=='spatial' else backend
            overrides['infer']=make_infer(fit_backend,profile)
            overrides['RawRoi']=make_roi(profile)
            if backend in ('spatial','native_spatial'):
                from .spatial_cache import CachedSpatialRoi
                overrides['RawRoi']=CachedSpatialRoi
        self.process_frame=MethodType(clone(Pipeline.process_frame,**overrides),self)
