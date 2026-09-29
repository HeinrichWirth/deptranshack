"""Conditional development-only CR covariance calibration, then research freeze."""
from lc_common import *
from latent_inference import producer_hash

RANGES=[(0,30),(30,60),(60,1000)]

def calibrate(rows):
    out=[]
    for lo,hi in RANGES:
        rr=[r for r in rows if lo<=r['range_m']<hi and r['station']>=8 and r['cr_lateral'] is not None and r['cr_vertical'] is not None and np.isfinite(r['cr_lateral']) and np.isfinite(r['cr_vertical'])]
        # Quantiles first per run, then equal-run median. Floors complement the
        # local Hessian covariance, which remains anisotropic and may be larger.
        vals=[]
        for run in sorted(set(r['run'] for r in rr)):
            x=[r for r in rr if r['run']==run]
            vals.append([np.quantile([abs(r['cr_lateral']) for r in x],.975)/2.2414,np.quantile([abs(r['cr_vertical']) for r in x],.975)/2.2414])
        sigma=np.maximum(.005,np.median(vals,axis=0)) if vals else np.array([.05,.05])
        out.append(dict(lo=lo,hi=hi,sigma_v=float(sigma[0]),sigma_w=float(sigma[1]),stations=len(rr),runs=len(vals),fallback=not bool(vals)))
    return out

def main():
    require_reproduction();check_baseline();selection=load(OUT/'selection/phase_b.json');protocol=load(OUT/'protocol.json');rows=load(OUT/'phase_b/per_station.json');configs=[];checks=[]
    for name in selection['selected']:
        rr=[r for r in rows if r['method']==name and r.get('selection_allowed',True)];cal=calibrate(rr);cfg=dict(next(c for c in protocol['candidates'] if c['name']==name),calibration=cal);configs.append(cfg)
        for held in protocol['split']['development']:
            fit=calibrate([r for r in rr if r['run']!=held]);test=[r for r in rr if r['run']==held and r['station']>=8 and r['cr_lateral'] is not None and np.isfinite(r['cr_lateral'])];mahal=[]
            for r in test:
                c=next(x for x in fit if x['lo']<=r['range_m']<x['hi']);mahal.append((r['cr_lateral']/c['sigma_v'])**2+(r['cr_vertical']/c['sigma_w'])**2)
            checks.append(dict(method=name,heldout_run=held,n=len(mahal),floor_only_coverage95=float(np.mean(np.array(mahal)<=5.991464547)) if mahal else None,floor_only_coverage99=float(np.mean(np.array(mahal)<=9.210340372)) if mahal else None,calibration=fit,note='Floor-only diagnostic; actual covariance also includes per-profile Hessian and competing modes. No benchmark residuals used.'))
    configs.append(next(c for c in protocol['candidates'] if c['name']=='C0_CURRENT'));freeze=dict(time_ns=time.time_ns(),status='RESEARCH_ONLY_NOT_PRODUCTION',best=selection['best'],selected=selection['selected'],configs=configs,producer=producer_hash(),baseline=sha(OUT/'baseline_freeze.json'),protocol=sha(OUT/'protocol.json'),phase_b_selection=sha(OUT/'selection/phase_b.json'),calibration='Development-only range-conditional anisotropic CR uncertainty floors. Existing STEP6 roll/offset state and corridor scale remain unchanged. Floors do not replace larger Hessian uncertainty.',loro_calibration=checks,benchmark_runs=protocol['split']['validation'],benchmark_starts=len(protocol['benchmark']),benchmark_used_for_selection=False,benchmark_already_used_in_previous_research=True,fresh_holdout_consumed=False)
    if (OUT/'phase_e/INFERENCE_COMPLETE.json').exists():raise RuntimeError('Benchmark already opened: cannot refreeze or retune')
    save(OUT/'research_freeze.json',freeze);write_csv(OUT/'uncertainty_loro.csv',checks);save(STAGE/'config/research_freeze.json',freeze);print('FROZEN',freeze['selected'],'BEST',freeze['best'])

if __name__=='__main__':main()
