"""Post-prediction LORO coverage with full local CR covariance plus held-run floors."""
from lc_common import *
from study_evaluation import align_prediction
from track_geometry import unit

def main():
    freeze=load(OUT/'research_freeze.json');best=freeze['best'];checks=[r for r in freeze['loro_calibration'] if r['method']==best];rows=[];protocol=load(OUT/'protocol.json')
    for check in checks:
        run=check['heldout_run'];cr=[];joint=[];width=[]
        for r in protocol['development']:
            if r['run']!=run:continue
            folder=OUT/'phase_b'/best/key(run,r['frame'])
            if load(folder/'prediction.json')['status']!='AVAILABLE':continue
            def npz(path):
                with np.load(path) as z:return {k:z[k] for k in z.files}
            p=npz(folder/'prediction.npz');e=npz(folder/'evaluation.npz');old=npz(OLD_OUT/'phase_b/B4_SPLINE_50__O1'/key(run,r['frame'])/'prediction.npz');a,ok,C,B=align_prediction(p,old)
            for j in range(len(old['s'])):
                if old['s'][j]<8 or not ok[j]:continue
                cal=next(c for c in check['calibration'] if c['lo']<=np.linalg.norm(old['C'][j])<c['hi']);cv=e['cr_cov'][j];delta=np.diag(np.maximum(np.array([cal['sigma_v'],cal['sigma_w']])**2-np.diag(cv),0));full=cv+delta;err=e['cr_error'][j]
                if np.isfinite(err).all():cr.append(float(err@np.linalg.pinv(full)@err))
                if not e['valid'][j]:continue
                G=old['B'][j,:,1:];dg=G@delta@G.T;b=unit(int(p['q'])*(e['gt'][j,0]-e['gt'][j,1]));n=np.cross(old['B'][j,:,0],b);T=np.column_stack((b,n));ms=[]
                for rail in (0,1):
                    cov=T.T@(a['rail_cov'][j,rail]+dg)@T;d=np.array([e['lat'][j,rail],e['vert'][j,rail]]);ms.append(float(d@np.linalg.pinv(cov)@d))
                    if rail==1:width.append(2*np.sqrt(5.991464547*np.maximum(np.diag(cov),0)))
                joint.append(max(ms))
        rows.append(dict(method=best,heldout_run=run,cr_n=len(cr),rail_n=len(joint),cr_coverage95=float(np.mean(np.array(cr)<=5.991464547)) if cr else None,cr_coverage99=float(np.mean(np.array(cr)<=9.210340372)) if cr else None,joint_coverage95=float(np.mean(np.array(joint)<=5.991464547)) if joint else None,joint_coverage99=float(np.mean(np.array(joint)<=9.210340372)) if joint else None,far_median_fullwidth95_v=float(np.median(np.array(width)[:,0])) if width else None,far_median_fullwidth95_w=float(np.median(np.array(width)[:,1])) if width else None,calibration_excluded_this_run=True,axis_note='Floors expressed in common evaluation transverse axes; avoids refitting. Same development corridor scale=1 as frozen STEP6 phase B.'))
    save(OUT/'uncertainty_loro_full.json',rows);write_csv(OUT/'uncertainty_loro_full.csv',rows);print('FULL LORO UNCERTAINTY',len(rows))

if __name__=='__main__':main()
