"""Same adapted direction, unoptimized frozen stages vs saved scheduled updates."""
from . import ROOT,OUT
from .engine import Engine
from .run_full import RUNS
import long_common as lc
import numpy as np
from .finalize import strip_times


def main():
    rows=[]
    for run in RUNS:
        root=OUT/'runs/scheduler_spatial'/run;b=lc.load(root/'INFERENCE_COMPLETE.json')
        choices=[r for r in b['rows'] if r['solve_attempted'] and r.get('direction',{}).get('mode')=='CAUSAL_ACCUMULATED_POSES']
        if not choices:continue
        # Geometry-independent, uniformly distributed adapter cases per complete run.
        selected=[choices[j] for j in np.unique(np.linspace(0,len(choices)-1,min(24,len(choices)),dtype=int))]
        engine=Engine('python');engine.start_run(lc.dataset()/run)
        for row in selected:
            i=row['frame'];r=engine.process_frame(i);folder=root/f'{i:06d}'
            meta=lc.load(folder/'summary.json')
            assert (r['prediction'] is not None)==meta['final_geometry_available']
            c4={k:v for k,v in r['c4'].items() if not isinstance(v,np.ndarray)}
            assert strip_times(lc.clean(c4))==strip_times(lc.load(folder/'c4.json'))
            for name,key in [('c4','c4'),('prediction','prediction'),('curve','c4_smooth'),('geometry_input','geometry_input')]:
                path=folder/(name+'.npz')
                if path.exists():
                    with np.load(path) as z:
                        for k in z.files:np.testing.assert_array_equal(z[k],r[key][k],err_msg=f'{run} {i} {name} {k}')
            with np.load(folder/'point_provenance.npz') as z:np.testing.assert_array_equal(z['rows'],r['provenance'])
            rows.append(dict(run=run,frame=i,bitwise_exact=True,direction=engine.direction_info))
            print('ADAPTED_EQ',run,i,flush=True)
    lc.save(OUT/'orientation_equivalence.json',dict(cases=len(rows),all_bitwise_exact=True,rows=rows,selection='up to 24 uniform indices per run, no quality-based selection'))


if __name__=='__main__':main()
