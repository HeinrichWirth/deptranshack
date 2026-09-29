"""Sequential wall-clock benchmark, run after other heavy study jobs finish."""
from common import *
from bootstrap import production_seed,pose_record
from tracker import predict
from contact_rail_step2 import ContactRailDetector

def main():
    cfg=load(OUT/'config.json');railcfg=load(ROOT/'MVP/stages/01_rail2d/config/detector_v2.json');contact=ContactRailDetector()
    rows=load(OUT/'analysis_rows.json');valid=[r for r in rows if r['seed_available']];chosen=[valid[j] for j in np.unique(np.linspace(0,len(valid)-1,24).astype(int))]
    records=[];steps=[]
    for r in chosen:
        run=r['run'];i=r['start_frame'];mm=frames(run);t=time.perf_counter();xyz=read_geometry(run,mm[i]);io=(time.perf_counter()-t)*1000
        s=production_seed(xyz,pose_record(mm[i]),pose_record(mm[i+1]) if i+1<len(mm) else None,railcfg,contact)
        result=predict(xyz,s,contact.template,cfg);elapsed=(time.perf_counter()-t)*1000
        records.append(dict(run=run,start_frame=i,read_and_transform_ms=io,seed_ms=s['seed_ms'],march_ms=result['total_ms'],end_to_end_ms=elapsed,steps=sum(x['status']=='ACCEPTED' for x in result['steps']),points=len(result['indices'])))
        steps.extend(result['steps'])
    fields=('pca_ms','bishop_ms','refinement_ms','projection_ms','template_ms','update_ms','unattributed_ms','total_ms')
    summary=dict(starts=len(records),execution='sequential one process, BLAS/OMP/MKL one thread, detector only; no evaluation/plotting',
                 per_start={k:percentiles([r[k] for r in records]) for k in ('read_and_transform_ms','seed_ms','march_ms','end_to_end_ms')},
                 per_step={k:percentiles([s.get(k,0.) for s in steps]) for k in fields})
    csv_write(OUT/'runtime/sequential_per_start.csv',records);save(OUT/'runtime/sequential_summary.json',summary);print('BENCHMARK',summary,flush=True)
if __name__=='__main__':main()
