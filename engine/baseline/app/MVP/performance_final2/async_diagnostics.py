from . import OUT
from .engine import Engine
from .async_worker import remaining
import long_common as lc
import numpy as np

def main():
    rows=[]
    for case in lc.csv_read(OUT/'async_worst_cases.csv'):
        if case['backend']!='batch':continue
        run=case['run'];i=int(case['geometry_source_frame']);e=Engine('native_batch_spatial');e.start_run(lc.dataset()/run);r=e.process_frame(i);e.close()
        assert r['prediction'] is not None
        rows.append(dict(kind=case['kind'],run=run,current_frame=int(case['frame']),geometry_source_frame=i,
            source_geometry_horizon_m=remaining(r['prediction']),source_s_end=float(r['prediction']['s'][-1]),
            current_remaining_horizon_m=float(case['remaining_horizon']),source_age_s=float(case['geometry_age_seconds']),
            distance_since_source_m=float(case['distance_since_geometry_source']),current_speed_m_s=float(case['speed_m_s']),
            original_reason=r['summary']['original_reason'],source_frames=r['summary']['source_frames'],labels_used=False))
    lc.csv_write(OUT/'async_failure_diagnosis.csv',rows)

if __name__=='__main__':main()
