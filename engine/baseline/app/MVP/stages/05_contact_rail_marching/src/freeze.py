"""Lock research parameters before held-out; this is NOT a production release."""
from common import *

def main():
    selected=load(OUT/'calibration/selected.json');cfg=selected['config']
    if (OUT/'freeze.json').exists():
        assert load(OUT/'config.json')==cfg,'Research lock already exists; no post-heldout tuning'
        return
    save(OUT/'config.json',cfg)
    save(OUT/'protocol.json',dict(stage='STEP 5 contact rail marching',production_release=False,data_root=str(dataset()),
        inference='Only XYZ cloud T; sanitized pose T and T+1; frozen STEP1/FINAL STEP2 bootstrap. No classes or future clouds. No running rails after seed.',
        motion='Preserve frozen >=0.50 m reliability rule using ONLY T+1; insufficient displacement means START_UNAVAILABLE; no later-pose lookahead.',
        split=SPLIT,cohort=load(OUT/'audit/cohort.json'),selection=selected,
        evaluation='Near seed uses GT_CURRENT_T; all NEW points use GT_FUTURE_FUSED from T+2 onward, same run and continuous registration segment, 150m pose-travel horizon.',
        reference=dict(voxel_m=.005,component_cell_m=.15,component_neighbor_m=.37,curve_bin_m=.5,max_curve_gap_m=.8,
                       explanation='Evaluation-only template anchors and linked observed component. Fused points are registered annotations, not surveyed truth.'),
        reach='From the seed, consecutive accepted blocks need >=3 evaluated points, >=80% evaluable support and >=95% point distances within tolerance. First bad/unscorable block ends continuous reach. Euclidean range from sensor T, not arc length.',
        meaningful_cohort='All held-out frames inventoried and inferred; primary reach summaries require bootstrap and >=2 future frames with linked GT reaching >=12m. Unavailable starts and unavailable GT are reported separately.',
        ablation='Compact one-factor development sweep; no held-out parameter selection.',
        heldout_known_from_prior_stages=True,confidence='Heuristic support score, not calibrated probability.'))
    dependencies=locks();names=('tracker.py','geometry.py','bootstrap.py','run_study.py','common.py','evaluation.py')
    code={f'src/{name}':sha(STAGE/'src'/name) for name in names}
    save(OUT/'freeze.json',dict(kind='research_configuration_lock_before_heldout',production_release=False,time_ns=time.time_ns(),
        config_sha256=sha(OUT/'config.json'),protocol_sha256=sha(OUT/'protocol.json'),inference_sha256=code,dependencies=dependencies,
        template_sha256=sha(ROOT/'MVP/stages/04_contact_rail_final/contact_rail_step2/canonical_template.npz'),development_selection=selected))
    print('RESEARCH LOCK',selected['selected'],flush=True)
if __name__=='__main__':main()
