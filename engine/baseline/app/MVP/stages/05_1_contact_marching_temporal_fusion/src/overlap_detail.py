"""Reconstruct the frozen engine's second overlap gate from saved real support.

No prediction rerun and no changed gate. The original log records the first
overlap fit but not its residual at the proposed NEW anchor.
"""
from fusion_common import *
from tracker import TemplateTracker
from contact_rail_step2 import ContactRailDetector
from concurrent.futures import ProcessPoolExecutor,as_completed

def one(folder):
    p=load_prediction(folder)
    if p['reason']!='OVERLAP_INCONSISTENT' or not p['steps']:return None
    st=p['steps'][-1];r=dict(variant=p['variant'],run=p['run'],start_frame=p['start_frame'],step_index=st['step_index'],first_fit_p90=st.get('overlap_residual_p90'),confirmed_overlap_n=st.get('confirmed_overlap_n'),candidate_n=st.get('candidate_n'),new_support_unique=st['template_candidates'][0]['support_unique'] if st.get('template_candidates') else None,new_template_score=st.get('template_score'),new_template_residual=st.get('template_residual'),second_fit_p90=None,gate=CFG['overlap_tolerance'],failure_gate='OVERLAP_TEMPLATE_NO_FIT')
    if st.get('overlap_anchor') is not None:
        with np.load(folder/'provenance.npz') as z:xyz=z['xyz']
        order=np.argsort(p['indices']);ids=np.asarray(st['overlap_indices'],dtype=int);positions=order[np.searchsorted(p['indices'][order],ids)]
        np.testing.assert_array_equal(p['indices'][positions],ids)
        q=(xyz[positions]-np.asarray(st['plane_origin']))@np.asarray(st['basis']);tt=TemplateTracker(ContactRailDetector().template,p['seed']['side'],CFG)
        first=float(np.quantile(tt.distances(q[:,1:],np.asarray(st['overlap_anchor'])),.9))
        assert abs(first-st['overlap_residual_p90'])<1e-9
        if first>CFG['overlap_tolerance']:r['failure_gate']='CONFIRMED_OVERLAP_FIT'
        elif st.get('anchor_v') is not None:
            anchor=np.array([st['anchor_v'],st['anchor_w']]);second=float(np.quantile(tt.distances(q[:,1:],anchor),.9))
            r.update(second_fit_p90=second,anchor_delta_v=float(anchor[0]-st['overlap_anchor'][0]),anchor_delta_w=float(anchor[1]-st['overlap_anchor'][1]),failure_gate='NEW_ANCHOR_RECHECK' if second>CFG['overlap_tolerance'] else 'CHECK_REMAINING_LOG')
            assert second>CFG['overlap_tolerance'],r
    save(OUT/'diagnostics/overlap_detail'/p['variant']/(key(p['run'],p['start_frame'])+'.json'),r)
    return r

def main():
    names=load(OUT/'research_lock.json')['heldout_variants'];paths=[p.parent for name in names for p in (OUT/'heldout'/name).glob('*/prediction.json')];rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for j,f in enumerate(as_completed([pool.submit(one,p) for p in paths])):
            r=f.result()
            if r:rows.append(r)
            if j%2000==0:print('OVERLAP DETAIL',j+1,len(paths),flush=True)
    csv_write(OUT/'overlap_failure_detail.csv',rows)
    print('OVERLAP DETAIL DONE',len(rows),flush=True)
if __name__=='__main__':main()
