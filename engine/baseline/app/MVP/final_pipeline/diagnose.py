"""Explain seed-only changes after evaluation; never updates a parameter."""
from . import ROOT
import long_common as lc
import numpy as np

OUT=ROOT/'results_final_pipeline'


def npz(p):
    with np.load(p) as z:return {k:z[k] for k in z.files}


def main():
    entries=[]
    for phase in ('deployable_development_cache','deployable_benchmark_cache'):
        rows=[r for r in lc.load(OUT/phase/'per_start_quality.json') if r.get('DEPLOYABLE_PIPELINE_far_p95') is not None]
        worst=sorted(rows,key=lambda r:r['DEPLOYABLE_PIPELINE_far_p95'],reverse=True)[:5]
        regressions=sorted(rows,key=lambda r:r['DEPLOYABLE_PIPELINE_far_p95']-r['RESEARCH_REFERENCE_far_p95'],reverse=True)[:5]
        newbad=[r for r in rows if r['DEPLOYABLE_PIPELINE_any_gt20cm'] and not r['RESEARCH_REFERENCE_any_gt20cm']]
        unique={(r['run'],r['frame']):r for r in worst+regressions+newbad}
        for (run,i),r in unique.items():
            key=lc.key(run,i);folder=OUT/phase/key
            raw=npz(folder/'prediction.npz');seed=lc.load(folder/'seed.json')
            oldgroup='phase_d' if phase=='deployable_development_cache' else 'phase_e'
            old=npz(ROOT/'results_latent_contact_reference'/oldgroup/'C4_SMOOTH'/key/'prediction.npz')
            oldseed=lc.load(ROOT/'results_running_rails_from_contact/c4_cr/seed8'/key/'input.json')['seed']
            ev=npz(folder/'evaluation.npz');v=ev['valid'];new_error=ev['raw_error'][v,1];old_error=ev['reference_error'][v,1]
            state_delta=raw['state'][0]-old['state'][0];disp=np.linalg.norm(raw['pair']-old['pair'],axis=2)
            item=dict(run=run,frame=i,phase=phase,old_sections=oldseed['sections'],raw_sections=seed['sections'],
                old_p95_cm=r['RESEARCH_REFERENCE_far_p95']*100,raw_p95_cm=r['DEPLOYABLE_PIPELINE_far_p95']*100,
                raw_max_cm=float(new_error.max()*100),old_max_cm=float(old_error.max()*100),
                first_gt20cm_range=float(ev['range'][v][np.flatnonzero(new_error>.2)[0]]) if np.any(new_error>.2) else None,
                initial_roll_delta_deg=float(np.rad2deg(state_delta[0])),initial_CR_to_near_delta_cm=float(state_delta[1]*100),
                initial_near_height_delta_cm=float(state_delta[2]*100),initial_gauge_delta_cm=float(state_delta[3]*100),
                initial_far_height_delta_cm=float(state_delta[4]*100),maximum_roll_delta_deg=float(np.rad2deg(abs(raw['state'][:,0]-old['state'][:,0]).max())),
                max_far_prediction_displacement_cm=float(disp[:,1].max()*100),
                C4_SMOOTH_max_change=float(abs(raw['C']-old['C']).max()),
                changed_only_by_seed_source=True,new_gt20cm_case=r['DEPLOYABLE_PIPELINE_any_gt20cm'] and not r['RESEARCH_REFERENCE_any_gt20cm'])
            entries.append(item)
    lc.save(OUT/'audit/worst_case_diagnosis.json',entries);lc.csv_write(OUT/'worst_case_diagnosis.csv',entries)
    for r in entries:
        if r['new_gt20cm_case'] or (r['phase']=='deployable_benchmark_cache' and r['frame'] in (51,112,67)):
            print(r,flush=True)


if __name__=='__main__':main()
