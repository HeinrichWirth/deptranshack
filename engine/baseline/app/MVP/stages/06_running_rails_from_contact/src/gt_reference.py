"""OFFLINE ONLY: independent two-component future-class1 reference, after predictions."""
from rr_common import *
from track_geometry import unit
from seed_input import head_center
from scipy.spatial import cKDTree

def valid_edge(a,b):
    dt=(int(b['header_time_ns'])-int(a['header_time_ns']))/1e9
    ds=np.linalg.norm(np.array(b['lidar_pose_in_folder'])[:3,3]-np.array(a['lidar_pose_in_folder'])[:3,3])
    return a.get('pose_status') in ('ok','origin') and b.get('pose_status') in ('ok','origin') and not a.get('pose_uses_future',False) and not b.get('pose_uses_future',False) and 0<dt<=.3 and ds<=5 and ds/dt<=50

def gt_forward(mm,i):
    P=np.asarray(mm[i]['lidar_pose_in_folder']);direction=None
    for j in range(i+1,min(i+12,len(mm))):
        if not valid_edge(mm[j-1],mm[j]):break
        d=np.asarray(mm[j]['lidar_pose_in_folder'])[:3,3]-P[:3,3]
        if np.linalg.norm(d)>=.5:direction=P[:3,:3].T@d;break
    if direction is None:
        for j in range(i-1,max(-1,i-12),-1):
            if not valid_edge(mm[j],mm[j+1]):break
            d=P[:3,3]-np.asarray(mm[j]['lidar_pose_in_folder'])[:3,3]
            if np.linalg.norm(d)>=.5:direction=P[:3,:3].T@d;break
    if direction is None:return None
    t=unit(direction);b=unit(np.cross([0,0,1.],t));n=np.cross(t,b)
    return np.column_stack((t,b,n))

def split_heads(q):
    """Label-only two-component separation; no predicted far rail is used."""
    if len(q)<10:return None
    centers=np.quantile(q[:,1],[.2,.8])
    for _ in range(12):
        ids=np.argmin(abs(q[:,1,None]-centers),axis=1)
        if min(np.bincount(ids,minlength=2))<5:return None
        update=np.array([np.median(q[ids==k,1]) for k in (0,1)])
        if np.max(abs(update-centers))<1e-5:break
        centers=update
    if centers[0]>centers[1]:ids=1-ids
    out=[head_center(q[ids==k]) for k in (0,1)]
    if any(v is None for v in out):return None
    heads=np.array([v[0] for v in out]);width=np.linalg.norm(heads[1,1:]-heads[0,1:])
    # Broad topology check, explicitly not a normative gauge constraint.
    if not .8<width<2.5:return None
    if max(np.linalg.norm(v[1]['spread'][1:]) for v in out)>.10:return None
    return heads,out

def prepare_run(task):
    run,group=task;barrier=load(OUT/group/'INFERENCE_COMPLETE.json')
    assert any(r['run']==run for r in barrier['starts'])
    path=OUT/'audit/future_class1'/run/'anchors.npz';meta=path.with_suffix('.json')
    if path.exists():return run,'cached'
    mm=frames(run);world=[];indices=[];uvalues=[];spreads=[];audit=[]
    for i,row in enumerate(mm):
        B=gt_forward(mm,i)
        if B is None:continue
        source=dataset()/run/row['file']
        with laspy.open(source) as f:h=f.header
        raw=np.memmap(source,dtype=h.point_format.dtype(),mode='r',offset=h.offset_to_point_data,shape=(h.point_count,))
        field='raw_classification' if h.point_format.id<6 else 'classification';mask=31 if h.point_format.id<6 else 255;ids=np.flatnonzero((raw[field]&mask)==1)
        xyz=np.column_stack([raw[a][ids].astype(float)*h.scales[k]+h.offsets[k] for k,a in enumerate(('X','Y','Z'))]);del raw
        P=np.asarray(row['lidar_pose_in_folder']);local=(xyz-P[:3,3])@P[:3,:3];uvw=local@B;valid=0
        for u in range(1,12):
            sel=(uvw[:,0]>=u)&(uvw[:,0]<u+1)&(abs(uvw[:,1])<4);res=split_heads(uvw[sel])
            if res is None:continue
            heads,diagnostic=res
            station=np.mean(heads[:,0]);heads[:,0]=station
            world.append((heads@B.T)@P[:3,:3].T+P[:3,3]);indices.append(i);uvalues.append(station)
            spreads.append([d[1]['spread'][1:] for d in diagnostic]);valid+=1
        audit.append(dict(frame=i,file=row['file'],class1_points=len(ids),sections=valid,bytes=source.stat().st_size,mtime_ns=source.stat().st_mtime_ns))
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,world=np.array(world).reshape(-1,2,3),frame=np.array(indices),u=np.array(uvalues),spread=np.array(spreads).reshape(-1,2,2))
    save(meta,dict(run=run,created_ns=time.time_ns(),prediction_barrier_ns=barrier['time_ns'],source_audit=audit,method='Future class1 in each source 1..12 m; independent two components and upper surface, no C4 future rail position.',normative_gauge_used=False))
    return run,len(world)

def future_pair_reference(run,i,barrier_folder):
    mark=load(barrier_folder/'PREDICTION_COMPLETE.json')
    for n,h in mark['files'].items():assert sha(barrier_folder/n)==h
    mm=frames(run);P=np.asarray(mm[i]['lidar_pose_in_folder']);end=i;travel=0.;path_s={i:0.}
    while end+1<len(mm) and travel<150:
        if not valid_edge(mm[end],mm[end+1]):break
        travel+=np.linalg.norm(np.asarray(mm[end+1]['lidar_pose_in_folder'])[:3,3]-np.asarray(mm[end]['lidar_pose_in_folder'])[:3,3]);end+=1;path_s[end]=travel
    with np.load(OUT/'audit/future_class1'/run/'anchors.npz') as z:
        keep=(z['frame']>=i+2)&(z['frame']<=end);w=z['world'][keep];f=z['frame'][keep];u=z['u'][keep]
    xyz=(w-P[:3,3])@P[:3,:3];station=np.array([path_s[int(k)] for k in f])+u
    bins=np.floor(station/.5).astype(int);pairs=[];ss=[];spread=[];support=[]
    for k in np.unique(bins):
        take=bins==k
        if take.sum()<2:continue
        p=np.median(xyz[take],axis=0);deviation=np.median(abs(xyz[take]-p),axis=0)*1.4826
        # Longitudinal spread belongs to station bin width, not rail disagreement.
        ss.append((k+.5)*.5);pairs.append(p);spread.append(deviation);support.append(int(take.sum()))
    return dict(s=np.array(ss),pair=np.array(pairs).reshape(-1,2,3),spread=np.array(spread).reshape(-1,2,3),support=np.array(support),future_end=end,future_frames=max(0,end-i-1),prediction_ns=mark['time_ns'],created_ns=time.time_ns())

def intersect_sections(C,B,reference):
    """Both rails intersect the SAME plane normal to CR tangent at each station."""
    pair=reference['pair'];ss=reference['s'];N=len(C);out=np.full((N,2,3),np.nan);unc=np.full((N,2),np.nan);chosen=np.full(N,np.nan)
    if len(ss)<2:return out,unc,chosen
    good=(np.diff(ss)<=1.)&(np.max(np.linalg.norm(np.diff(pair,axis=0),axis=2),axis=1)<1.5)
    previous=-np.inf
    for k,(center,BB) in enumerate(zip(C,B)):
        roots=[]
        for rail in (0,1):
            u=(pair[:,rail]-center)@BB[:,0];segments=np.flatnonzero(good&(u[:-1]*u[1:]<=0)&(abs(np.diff(u))>1e-5))
            rr=[]
            for j in segments:
                t=-u[j]/(u[j+1]-u[j]);p=pair[j,rail]+t*(pair[j+1,rail]-pair[j,rail]);station=ss[j]+t*(ss[j+1]-ss[j])
                if station>=previous-1:rr.append((station,p,j))
            roots.append(rr)
        candidates=[]
        for a in roots[0]:
            for b in roots[1]:
                if abs(a[0]-b[0])>.8:continue
                mid=(a[1]+b[1])/2;dist=np.linalg.norm(mid-center)
                if dist<5:candidates.append((dist,a,b))
        if not candidates:continue
        _,a,b=min(candidates,key=lambda x:x[0]);out[k]=[a[1],b[1]];chosen[k]=(a[0]+b[0])/2;previous=chosen[k]
        for r,entry in enumerate((a,b)):
            variance=np.mean(reference['spread'][entry[2]:entry[2]+2,r]**2,axis=0)
            transverse_variance=variance@(BB[:,1:]**2)
            unc[k,r]=max(.005,float(np.sqrt(np.sum(transverse_variance))))
    return out,unc,chosen
