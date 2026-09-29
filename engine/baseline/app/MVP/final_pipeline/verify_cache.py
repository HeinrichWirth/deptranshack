"""Exact engineering equivalence, including the new RAW seed baseline."""
from . import ROOT
from .pipeline import Pipeline,write_frame
from .verify_raw import exact
import long_common as lc
import numpy as np
import time

OUT=ROOT/'results_final_pipeline'


def read_array(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def main():
    barrier=lc.load(OUT/'deployable_baseline/INFERENCE_COMPLETE.json')
    # Correct an integration omission discovered before optimization: preserve
    # the frozen 8m seed-only C4 fallback. Recompute just those two baselines.
    for j,rec in enumerate(barrier['results']):
        if rec['status']=='C4_SHORT_HORIZON':
            p=Pipeline(dict(engine='baseline'));p.start_run(lc.dataset()/rec['run']);r=p.process_frame(rec['frame'])
            write_frame(OUT/'deployable_baseline'/lc.key(rec['run'],rec['frame']),r)
            barrier['results'][j]=r['summary']
    lc.save(OUT/'deployable_baseline/INFERENCE_COMPLETE.json',barrier)
    rows=[]
    for run in sorted(set(r['run'] for r in barrier['starts'])):
        p=Pipeline(dict(engine='cache'));p.start_run(lc.dataset()/run)
        for rec in sorted((r for r in barrier['starts'] if r['run']==run),key=lambda r:r['frame']):
            i=rec['frame'];folder=OUT/'deployable_baseline'/lc.key(run,i)
            r=p.process_frame(i)
            before=lc.load(folder/'c4.json');before.update(read_array(folder/'c4.npz'))
            exact(r['c4'],before)
            exact(r['seed'],lc.load(folder/'seed.json'))
            if r['prediction'] is not None:
                exact(r['prediction'],read_array(folder/'prediction.npz'))
                exact(r['c4_smooth'],read_array(folder/'curve.npz'))
                exact(r['geometry_input'],read_array(folder/'geometry_input.npz'))
            np.testing.assert_array_equal(r['provenance'],read_array(folder/'point_provenance.npz')['rows'])
            row=dict(run=run,frame=i,bitwise_equal=True,max_difference=0,
                baseline=lc.load(folder/'summary.json')['timing'],optimized=r['summary']['timing'])
            rows.append(row);write_frame(OUT/'deployable_baseline_cache'/lc.key(run,i),r)
            print('CACHE_EXACT',len(rows),run,i,flush=True)
    lc.save(OUT/'equivalence/cache_40.json',dict(cases=len(rows),all_discrete_exact=True,
        all_arrays_bitwise_equal=True,max_numeric_difference=0,rows=rows))


if __name__=='__main__':main()
