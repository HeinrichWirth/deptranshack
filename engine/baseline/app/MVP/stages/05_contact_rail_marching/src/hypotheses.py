"""Supplementary scientific diagnostics after prediction; no tuning."""
from common import *
from geometry import scatter,pca,transport
from evaluation import reference,curve_match
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

def main():
    # Translation of a transverse plane cannot change centered scatter for the
    # same points; midpoint affects crop/overlap rather than this objective itself.
    rng=np.random.default_rng(123);p=rng.normal(size=(500,3))*[5,.02,.04];B=np.eye(3)
    tests={name:[scatter(p-shift,B,name) for shift in (np.array([0,0,0]),np.array([4,1,2]),np.array([8,1,2]))] for name in ('trace','logdet','ellipse90','ellipse95','mad')}
    t,c=pca(p,B[:,0],robust=False);BB=transport(B,t);cov=np.cov(p,rowvar=False,bias=True)
    exact=float(np.trace(cov)-np.linalg.eigvalsh(cov)[-1]);observed=float(np.mean(np.sum(((p-p.mean(axis=0))@BB[:,1:])**2,axis=1)))
    save(OUT/'audit/geometric_hypotheses.json',dict(translation_invariance=tests,max_translation_difference=max(np.ptp(v) for v in tests.values()),
         raw_PCA_trace_identity=dict(theory=exact,measured=observed,difference=abs(exact-observed)),
         conclusion='Untrimmed covariance trace is minimized by first PCA component. Centered robust scatter is translation invariant; midpoint can improve overlap/cropping, not the scatter of identical points.'))
    rows=load(OUT/'analysis_rows.json');orientation=csv_read(OUT/'orientation_metrics.csv');strata=[]
    for r in rows:
        if not r.get('evaluation_eligible'):continue
        ss=[s for s in orientation if s['run']==r['run'] and int(s['start_frame'])==r['start_frame']]
        cur=[float(s['gt_curvature_per_m']) for s in ss if s.get('gt_curvature_per_m')];pitch=[abs(float(s['gt_pitch_deg'])) for s in ss if s.get('gt_pitch_deg')]
        curvature=float(np.median(cur)) if cur else None;grade=float(np.median(pitch)) if pitch else None
        strata.append(dict(run=r['run'],start_frame=r['start_frame'],reach=r['continuous_reach_5cm'],curvature=curvature,grade_deg=grade,
           curve_bin='unavailable' if curvature is None else 'near_straight' if curvature<.002 else 'mild' if curvature<.01 else 'strong',
           grade_bin='unavailable' if grade is None else '<0.5deg' if grade<.5 else '0.5-1.5deg' if grade<1.5 else '>=1.5deg'))
    csv_write(OUT/'curvature_grade_strata.csv',strata)
    ss=[]
    for kind in ('curve_bin','grade_bin'):
        for name in sorted(set(r[kind] for r in strata)):
            rr=[r for r in strata if r[kind]==name];ss.append(dict(kind=kind,bin=name,**percentiles([r['reach'] for r in rr])))
    csv_write(OUT/'curvature_grade_summary.csv',ss)
    # Drift association across all matched accepted/terminal steps; report
    # censoring and dependence, never treat steps as independent experiments.
    dr=[]
    for field in ('GT_anchor_error','tangent_angle_error_deg'):
        rr=[s for s in orientation if s.get(field) and s.get('range_far')]
        dr.append(dict(metric=field,n=len(rr),spearman_step=float(spearmanr([float(s['step_index']) for s in rr],[float(s[field]) for s in rr]).statistic) if len(rr)>3 else None,
                  spearman_range=float(spearmanr([float(s['range_far']) for s in rr],[float(s[field]) for s in rr]).statistic) if len(rr)>3 else None))
    save(OUT/'drift_summary.json',dr)
    print('HYPOTHESES',tests,'PCA',exact,observed,'strata',ss,flush=True)
if __name__=='__main__':main()
