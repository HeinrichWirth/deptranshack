"""Time only the new postprocessor, and compare withheld tails to saved geometry."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import sys,json,time,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT)]+([str(ROOT/'.runtime')] if os.name=='nt' else [])
import numpy as np
from COPY_MAIN.postprocess.linear_extension import extend_curves

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'COPY_MAIN/results/EXTENSION5');parser.add_argument('--length',type=float,choices=[5,20],default=5);parser.add_argument('--run',default='INPUT10HZ_SOLVE5_REPLAY');args=parser.parse_args()
out=args.output;out.mkdir(exist_ok=True)
folder=ROOT/'COPY_MAIN/results'/args.run/'payload/solves'
cases=[];read_begin=time.perf_counter()
for p in sorted(folder.glob('*/prediction.npz')):
    with np.load(p) as z:
        if len(z['s'])<3: continue
        s=z['s'];grid=np.unique(np.r_[np.arange(float(s[0]),float(s[-1]),.1),s[-1]])
        curves={k:np.round(np.column_stack([np.interp(grid,s,v[:,j]) for j in range(3)]),5) for k,v in [('left',z['pair'][:,0]),('right',z['pair'][:,1]),('contact',z['C']),('center',z['pair'].mean(axis=1))]}
    cases.append((int(p.parent.name),curves))
load_seconds=time.perf_counter()-read_begin
cold=[];hot=[];backtests=[]
for i,curves in cases:
    before={k:v.copy() for k,v in curves.items()}
    start=time.perf_counter_ns();result,detail=extend_curves(curves,length=args.length);cold.append((time.perf_counter_ns()-start)/1e6)
    n=len(curves['left'])
    for k,v in before.items():
        np.testing.assert_array_equal(curves[k],v)
        np.testing.assert_array_equal(result[k][:n],v)
    np.testing.assert_allclose(np.linalg.norm(result['left'][-1]-curves['left'][-1]),args.length,atol=1e-12)
    np.testing.assert_allclose(result['left'][n:]-result['right'][n:],np.broadcast_to(curves['left'][-1]-curves['right'][-1],(int(args.length*10),3)),atol=1e-12)
    center=(curves['left']+curves['right'])*.5;s=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(center,axis=0),axis=1))]
    cut=s[-1]-args.length
    if cut>=4:
        take=s<cut
        prefix={k:np.vstack((v[take], [np.interp(cut,s,v[:,j]) for j in range(3)])) for k,v in curves.items()}
        predicted, _=extend_curves(prefix,length=args.length)
        errors={k:float(np.linalg.norm(predicted[k][-1]-curves[k][-1])) for k in ('left','right','contact')}
        five_errors=[float(np.linalg.norm(predicted[k][len(prefix[k])+49]-np.array([np.interp(cut+5,s,curves[k][:,j]) for j in range(3)]))) for k in ('left','right','contact')]
        backtests.append(dict(frame=i,end_error_m=errors,max_end_error_m=max(errors.values()),error_at_5m_same_prefix=max(five_errors)))
for repeat in range(30):
    for i,curves in cases:
        start=time.perf_counter_ns();extend_curves(curves,length=args.length);hot.append((time.perf_counter_ns()-start)/1e6)
def stats(values):
    return dict(n=len(values),median=float(np.median(values)),p95=float(np.percentile(values,95)),p99=float(np.percentile(values,99)),maximum=float(np.max(values)))
report=dict(frames=len(cases),algorithm=f'one common linear tangent from last 3 m; anchored +{args.length:g} m; same terminal offsets',
            future_used=False,source_modified=False,load_seconds_excluded=load_seconds,
            first_call_per_frame_ms=stats(cold),repeated_calls_ms=stats(hot),
            calls_over_20ms=int(sum(v>=20 for v in cold+hot)),
            withheld_tail_proxy_error_m=stats([r['max_end_error_m'] for r in backtests]),
            same_prefix_error_at_5m=stats([r['error_at_5m_same_prefix'] for r in backtests]),
            accuracy_note=f'Withheld last {args.length:g} m of the saved model in the SAME frame, not measured GT or proof of hidden rail position.',
            worst_proxy=sorted(backtests,key=lambda r:r['max_end_error_m'],reverse=True)[:10])
(out/'BENCHMARK.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
