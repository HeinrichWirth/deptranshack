from . import ROOT,OUT
from .native_api import startup
import long_common as lc
import numpy as np
import time

def main():
    native=startup()
    with np.load(ROOT/'results_performance_final/profiles/fit_0000.npz') as z:c={k:z[k] for k in z.files}
    # Independent objects: mutable residual cache is never shared between workers.
    rows=[]
    for released in (False,True):
        for workers in (1,2,4,8,16):
            objects=[native.Problem(c['points'],c['template'],c['center']-.1,c['center']+.1,False) for _ in range(16)]
            def one(i):
                for _ in range(200):value=objects[i].fun(c['center'],released)
                return value
            pool=native.Pool(workers);t=time.perf_counter()
            try:result=pool.map(one,list(range(16)))
            finally:pool.close()
            elapsed=time.perf_counter()-t
            for a in result:np.testing.assert_array_equal(a,result[0])
            rows.append(dict(release_gil=released,workers=workers,calls=3200,seconds=elapsed,exact=True))
    lc.csv_write(OUT/'gil_scaling.csv',rows)

if __name__=='__main__':main()
