"""CR derivative errors against future-class2 proxy, after inference only."""
from lc_common import *
from curve_geometry import plan_frame
from latent_inference import read_causal

def derivatives(s,C):
    v=np.gradient(C,s,axis=0);a=np.gradient(v,s,axis=0);sp=np.maximum(np.linalg.norm(v[:,:2],axis=1),1e-8);kh=(v[:,0]*a[:,1]-v[:,1]*a[:,0])/sp**3;grade=v[:,2]/sp;kv=np.gradient(grade,s);return kh,kv

def main():
    best=load(OUT/'research_freeze.json')['best'];rows=[]
    for group in ('phase_d','phase_e'):
        for r in load(OUT/group/'per_start.json'):
            if r['method'] not in ('C0_CURRENT',best) or r['status']!='AVAILABLE':continue
            folder=OUT/group/r['method']/key(r['run'],r['frame']);F=np.asarray(load(folder/'details.json')['frame_matrix'])
            with np.load(folder/'evaluation.npz') as z:s=z['station'];truth=z['cr_gt']@F;pred=z['aligned_new_C']@F;valid=np.isfinite(truth).all(axis=1)&np.isfinite(pred).all(axis=1)
            ids=np.flatnonzero(valid);groups=np.split(ids,np.flatnonzero((np.diff(ids)>1)|(np.diff(s[ids])>2.1)|(np.diff(s[ids])<1e-6))+1)
            for g in groups:
                if len(g)<5:continue
                kh,kv=derivatives(s[g],pred[g]);gh,gv=derivatives(s[g],truth[g])
                for local in range(2,len(g)-2):
                    j=g[local];rows.append(dict(group=group,method=r['method'],run=r['run'],frame=r['frame'],station=float(s[j]),range_m=float(np.linalg.norm(pred[j])),curvature_h_pred=float(kh[local]),curvature_h_GT_proxy=float(gh[local]),curvature_h_error=float(kh[local]-gh[local]),curvature_v_error=float(kv[local]-gv[local]),GT_eval_only=True))
    write_csv(OUT/'cr_curvature_eval.csv',rows);summary=[]
    for group in ('phase_d','phase_e'):
        for method in ('C0_CURRENT',best):
            for lo,hi in ((30,50),(50,75),(75,100)):
                rr=[r for r in rows if r['group']==group and r['method']==method and lo<=r['range_m']<hi];summary.append(dict(group=group,method=method,lo=lo,hi=hi,stations=len(rr),curvature_h_abs_p95=stats([abs(r['curvature_h_error']) for r in rr]).get('p95'),curvature_v_abs_p95=stats([abs(r['curvature_v_error']) for r in rr]).get('p95'),note='Finite-difference error vs imperfect class2 reference; gaps not bridged, 2 boundary stations per continuous component excluded.'))
    write_csv(OUT/'cr_curvature_summary.csv',summary);save(OUT/'cr_curvature_summary.json',summary);print('CR CURVATURE EVAL',len(rows))

if __name__=='__main__':main()
