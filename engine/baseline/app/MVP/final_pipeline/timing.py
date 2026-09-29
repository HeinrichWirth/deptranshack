"""Direct stage boundary timing in COPY_MAIN, without Python line tracing."""
import time
from final_geometry import predict_geometry

def geometry_call(*args):
    started=time.perf_counter();timing={}
    result=predict_geometry(*args,_timing=timing)
    finished=time.perf_counter();middle=timing['middle']
    return result,dict(T_C4_SMOOTH=middle-started,T_STEP6_RAIL_RECONSTRUCTION=finished-middle)
