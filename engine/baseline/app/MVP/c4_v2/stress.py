"""Perturb only an experimental persistent tail, never source files."""
from . import OUT
from .persistent import PersistentEngine,stations,world
import long_common as lc
import numpy as np
from scipy.spatial import cKDTree


def main():
    run='roundT_pressureGate_roundT';rows=[];reference={}
    for cm in (0,1,3,5,10):
        e=PersistentEngine(**lc.load(OUT/'selected_config.json'))
        e.start_run(lc.dataset()/run)
        for i in range(41):
            if i==30 and cm:
                s=e.state;ss=stations(s.anchors);mask=ss>=ss[-1]-12
                tangent=s.anchors[-1]-s.anchors[-2];tangent/=np.linalg.norm(tangent);axis=np.cross([0,0,1],tangent);axis/=np.linalg.norm(axis)
                s.anchors[mask]+=axis*(cm/100)
            r=e.process_frame(i)
            if i<30:continue
            p=r['prediction'];w=None if p is None else world(p['C'],np.asarray(e.records[i]['lidar_pose_in_folder']))
            if cm==0:reference[i]=w
            delta=None if w is None or reference[i] is None else float(np.percentile(cKDTree(reference[i]).query(w)[0],95))
            rows.append(dict(run=run,frame=i,perturb_cm=cm,status=e.stats[-1]['status'],reason=e.stats[-1]['reason'],difference_p95_m=delta,available=p is not None))
        e.close();print('TAIL_STRESS',cm,flush=True)
    lc.csv_write(OUT/'tail_stress.csv',rows)

if __name__=='__main__':main()
