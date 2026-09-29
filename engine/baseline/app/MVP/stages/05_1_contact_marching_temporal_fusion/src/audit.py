from fusion_common import *
from fusion import CausalSource

def main():
    initialize();deps=dependency_hashes();save(OUT/'audit/previous_stage_sha.json',deps)
    dev=load(BASE_OUT/'calibration/sweep_protocol.json')['cohort'];held=[r for r in load(BASE_OUT/'audit/cohort.json') if r['split']=='validation']
    save(OUT/'audit/cohort.json',dict(development=dev,heldout=held))
    rows=[]
    for r in dev+held:
        i=r['frame'];src=CausalSource(r['run'],i)
        for distance in (.5,1.,2.,4.):
            ids,reason=src.history(dict(distance=distance));rows.append(dict(run=r['run'],frame=i,distance=distance,frames=len(ids),reason=reason))
    csv_write(OUT/'audit/history_inventory.csv',rows)
    save(OUT/'audit/history_summary.json',{str(d):percentiles([r['frames'] for r in rows if r['distance']==d]) for d in (.5,1.,2.,4.)})
    save(OUT/'protocol.json',dict(data_root=str(dataset()),split=SPLIT,baseline=CFG,development_starts=len(dev),heldout_starts=len(held),
        selection='Equal-run mean continuous reach 5cm minus 100*raw wrong structure rate; prefer fewer frames and RAW_CONCAT within 5% of best score. Require paired wins > losses for F6/F8 expansion. No heldout tuning.',
        history='Contiguous causal past prefix; Euclidean sensor displacement threshold, stop at first outside distance or invalid edge; no padding at run start; actual count reported.',
        gaps='Same STEP5 validity: dt>0 <=.3s, step<=5m, speed<=50m/s, pose ok/origin, no future-based poses.',
        thresholds_unchanged=True,production_release=False,self_train_mask='Only if a documented physical envelope is found; do not infer train geometry from class labels.'))
    print('HISTORY',load(OUT/'audit/history_summary.json'),flush=True)
if __name__=='__main__':main()
