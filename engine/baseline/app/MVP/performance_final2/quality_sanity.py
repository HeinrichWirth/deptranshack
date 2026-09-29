"""Post-inference only: unchanged downstream range metrics with approximate GT."""
from . import ROOT,OUT
import long_common as lc
import numpy as np
import time
from gt_reference import future_pair_reference
from evaluate_rails import measure

def main():
    barrier=lc.load(OUT/'FINAL_EQ_COMPLETE.json');rows=[]
    for case in lc.load(OUT/'benchmark_cohort.json')['final']:
        folder=OUT/'equivalence'/case['run']/f'{case["frame"]:06d}'
        if not (folder/'original.npz').exists():continue
        # Migration of this sprint's completed markers to the frozen evaluator's
        # file-hash contract. Predictions and original inference time unchanged.
        marker=lc.load(folder/'PREDICTION_COMPLETE.json')
        if 'files' not in marker:
            marker['files']={p.name:lc.sha(p) for p in folder.glob('*.npz')};marker['hashes_added_post_inference_ns']=time.time_ns()
            lc.save(folder/'PREDICTION_COMPLETE.json',marker)
        ref=future_pair_reference(case['run'],case['frame'],folder)
        if not len(ref['s']):continue
        outputs=[]
        for name in ('original','best'):
            with np.load(folder/(name+'.npz')) as z:p={k:z[k] for k in z.files}
            outputs.append(measure(p,ref,p['B']))
        a,b=outputs
        np.testing.assert_array_equal(a['error'],b['error'])
        for lo,hi in ((30,50),(50,75),(75,100)):
            take=a['valid']&(a['range']>=lo)&(a['range']<hi)
            rows.extend(dict(run=case['run'],frame=case['frame'],lo=lo,hi=hi,original_far=float(x),best_far=float(y))
                for x,y in zip(a['error'][take,1],b['error'][take,1]))
    table=[]
    for lo,hi in ((30,50),(50,75),(75,100)):
        rr=[r for r in rows if r['lo']==lo];a=[r['original_far'] for r in rr];b=[r['best_far'] for r in rr]
        table.append(dict(lo=lo,hi=hi,stations=len(rr),starts=len(set((r['run'],r['frame']) for r in rr)),
            original_p95=None if not rr else float(np.percentile(a,95)),best_p95=None if not rr else float(np.percentile(b,95)),
            max_difference=None if not rr else float(np.max(abs(np.array(a)-b)))))
    lc.csv_write(OUT/'quality_sanity.csv',table);lc.save(OUT/'quality_sanity.json',dict(inference_barrier=barrier,rows=table,
        reference='existing approximate future class1, read only AFTER inference, no tuning'))
    print('QUALITY_SANITY',table,flush=True)

if __name__=='__main__':main()
