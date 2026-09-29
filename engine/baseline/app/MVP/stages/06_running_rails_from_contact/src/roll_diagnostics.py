"""Development-only signal tests; diagnostic GT never enters the predictor."""
from rr_common import *
from seed_input import read_input
from scipy.optimize import least_squares
import argparse

def corr(x,y):
    x=np.array(x);y=np.array(y);ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
    return float(np.corrcoef(x,y)[0,1]) if len(x)>=10 and np.std(x)>1e-5 and np.std(y)>1e-5 else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('group',default='phase_a',nargs='?');args=ap.parse_args();group=args.group;barrier=load(OUT/group/'EVALUATION_COMPLETE.json');data=[];slope_data=[]
    for path in (OUT/group/'B1_BISHOP').glob('*/prediction.json'):
        r=load(path)
        if r['status']!='AVAILABLE':continue
        m,a=read_input(OUT/r['input_folder'])
        with np.load(path.parent/'evaluation.npz') as z:e={k:z[k] for k in z.files}
        prof=m['profiles'];beta=np.array([np.nan if p.get('beta') is None else p['beta'] for p in prof]);good=np.array([p.get('informative',False) for p in prof]);seed=good&(a['s']<=8)
        beta0=np.median(beta[seed]) if seed.any() else np.nan;alpha0=m['seed']['x0'][0]
        dt=np.gradient(a['B'][:,:,0],a['s'],axis=0);kh=np.einsum('ij,ij->i',dt,a['B'][:,:,1]);kh0=np.median(kh[a['s']<=8])
        for k in np.flatnonzero(e['valid']&(a['s']>8)):
            actual=e['alpha_gt'][k];base=dict(run=r['run'],frame=r['i'],station=float(a['s'][k]),range_m=float(e['range'][k]),alpha_gt=actual,alpha0=alpha0,curvature=kh[k],curvature_delta=kh[k]-kh0,alpha_delta=actual-alpha0)
            slope_data.append(base)
            if good[k] and np.isfinite(beta0):data.append(dict(base,beta=beta[k],beta0=beta0,alpha_from_profile=beta[k]+alpha0-beta0,error=beta[k]+alpha0-beta0-actual,baseline_error=alpha0-actual,support=prof[k]['n'],sigma=prof[k]['sigma']))
    by_run=[]
    for run in sorted(set(r['run'] for r in data)):
        rr=[r for r in data if r['run']==run];rho=corr([r['beta']-r['beta0'] for r in rr],[r['alpha_delta'] for r in rr]);err=np.rad2deg([r['error'] for r in rr])
        by_run.append(dict(run=run,n=len(rr),starts=len(set(r['frame'] for r in rr)),correlation=rho,bias_deg=float(np.median(err)),median_abs_deg=float(np.median(abs(err))),p90_abs_deg=float(np.quantile(abs(err),.9)),p95_abs_deg=float(np.quantile(abs(err),.95)),baseline_p95_deg=float(np.quantile(abs(np.rad2deg([r['baseline_error'] for r in rr])),.95))))
    rho=[r['correlation'] for r in by_run if r['correlation'] is not None and r['n']>=20]
    profile_supported=bool(len(rho)>=3 and np.median(rho)>=.35)
    write_csv(OUT/'cr_profile_roll'/f'{group}_observations.csv',data);write_csv(OUT/'cr_profile_roll'/f'{group}_by_run.csv',by_run)
    curv=[];priors={}
    runs=sorted(set(r['run'] for r in slope_data))
    for held in runs:
        train=[r for r in slope_data if r['run']!=held];test=[r for r in slope_data if r['run']==held]
        xx=np.array([r['curvature_delta'] for r in train]);yy=np.array([r['alpha_delta'] for r in train]);fit=least_squares(lambda c:(c[0]*xx-yy)/.02,[0.],loss='soft_l1',bounds=(-20,20))
        coef=float(fit.x[0]);err=np.array([coef*r['curvature_delta']-r['alpha_delta'] for r in test]);base=np.array([r['alpha_delta'] for r in test]);sigma=float(np.std(coef*xx-yy))
        curv.append(dict(run=held,n=len(test),coef=coef,correlation=corr([r['curvature_delta'] for r in test],[r['alpha_delta'] for r in test]),baseline_p95_deg=float(np.quantile(abs(np.rad2deg(base)),.95)),prior_p95_deg=float(np.quantile(abs(np.rad2deg(err)),.95))))
        priors[held]=dict(coef=coef,residual_sigma=sigma)
    gains=[(r['baseline_p95_deg']-r['prior_p95_deg'])/max(.01,r['baseline_p95_deg']) for r in curv if r['n']>=20]
    curvature_supported=bool(len(gains)>=3 and np.median(gains)>.1 and np.mean(np.array(gains)>0)>=.75)
    write_csv(OUT/'curvature_roll'/f'{group}_observations.csv',slope_data);write_csv(OUT/'curvature_roll'/f'{group}_cross_run.csv',curv)
    save(OUT/'models'/f'{group}_signal_gates.json',dict(profile_supported=profile_supported,profile_median_run_correlation=float(np.median(rho)) if rho else None,profile_gate='At least 3 runs with >=20 observations; median within-start-change correlation >=0.35.',curvature_supported=curvature_supported,curvature_gate='LORO median p95 improvement >10%, positive in at least 75% of >=20-observation runs.',curvature_priors=priors,phase=group))
    print('SIGNAL GATES',profile_supported,curvature_supported,'profile median rho',np.median(rho) if rho else None,flush=True)

if __name__=='__main__':main()
