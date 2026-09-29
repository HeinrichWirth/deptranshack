"""Evaluation-only decomposition of NEW CR errors; never changes a prediction."""
from lc_common import *
from study_evaluation import align_prediction,BINS
from track_geometry import offsets,rails

def main():
    best=load(OUT/'research_freeze.json')['best'];rows=[]
    for group,oldgroup in [('phase_d','phase_b'),('phase_e','phase_e')]:
        for rec in load(OUT/group/'per_start.json'):
            if rec['method']!=best or rec['status']!='AVAILABLE':continue
            case=key(rec['run'],rec['frame']);folder=OUT/group/best/case
            def npz(path):
                with np.load(path) as z:return {k:z[k] for k in z.files}
            p=npz(folder/'prediction.npz');e=npz(folder/'evaluation.npz');old=npz(OLD_OUT/oldgroup/'B4_SPLINE_50__O1'/case/'prediction.npz');a,valid,C,B=align_prediction(p,old);valid &=e['valid'];out={name:np.full(len(C),np.nan) for name in ('new_production','new_oracle_alpha','new_oracle_offsets','new_all_oracle')}
            for j in np.flatnonzero(valid):
                oracle=offsets(C[j],B[j],e['gt'][j],int(p['q']));state=a['state'][j].copy();states={'new_production':state.copy(),'new_oracle_alpha':np.r_[oracle[0],state[1:]],'new_oracle_offsets':np.r_[state[0],oracle[1:]],'new_all_oracle':oracle}
                for name,x in states.items():
                    pair=rails(C[j],B[j],x,int(p['q']));delta=pair[1]-e['gt'][j,1];T=old['B'][j,:,0];out[name][j]=np.linalg.norm(delta-T*(delta@T))
            dest=OUT/'remaining_roll'/group/case;dest.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest/'errors.npz',s=old['s'],valid=valid,**out)
            for lo,hi in BINS:
                take=valid&(np.linalg.norm(old['C'],axis=1)>=lo)&(np.linalg.norm(old['C'],axis=1)<hi)&(old['s']>=8)
                if take.any():rows.append(dict(run=rec['run'],frame=rec['frame'],group=group,lo=lo,hi=hi,stations=int(sum(take)),**{k:float(np.percentile(v[take],95)) for k,v in out.items()}))
    write_csv(OUT/'remaining_roll_per_start.csv',rows);summary=[]
    for group in ('phase_d','phase_e'):
        for lo,hi in BINS:
            rr=[r for r in rows if r['group']==group and r['lo']==lo and r['hi']==hi]
            if rr:summary.append(dict(group=group,lo=lo,hi=hi,starts=len(rr),stations=sum(r['stations'] for r in rr),**{k:float(np.median([r[k] for r in rr])) for k in ('new_production','new_oracle_alpha','new_oracle_offsets','new_all_oracle')},note='Median per-start p95; all variants rebuilt algebraically in NEW CR frame and evaluated in shared old planes. GT used only for this diagnostic.'))
    write_csv(OUT/'remaining_roll.csv',summary);save(OUT/'remaining_roll.json',summary);print('REMAINING ROLL',len(rows))

if __name__=='__main__':main()
