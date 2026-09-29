"""Offline LAS adapter; the frozen detector dependencies are read-only."""
import os,sys
sys.dont_write_bytecode=True
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
STEP1=ROOT/'MVP/stages/01_rail2d';STEP2=ROOT/'MVP/stages/04_contact_rail_final'
sys.path[:0]=[str(ROOT/'.runtime'),str(STEP1/'src'),str(STEP2)]
import numpy as np,laspy,json,hashlib,csv,io,time
from rail2d_motion_v2 import MotionSelector
from rail2d_head_v2 import detect as detect_rails
from contact_rail_step2 import ContactRailDetector

def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def clean(x):
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.generic):return clean(x.item())
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def save(path,x):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
def digest(data):return hashlib.sha256(data).hexdigest()
def sha(path):return digest(Path(path).read_bytes())
def csv_write(path,rows):
    rows=list(rows);keys=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows([{k:clean(v) for k,v in r.items()} for r in rows])
def locks():
    out={}
    for folder in (STEP1,STEP2):
        m=load(folder/'MANIFEST.json')
        for r in m['files']:assert sha(folder/r['path'])==r['sha256'],str(folder/r['path'])
        out[folder.name]=dict(manifest_sha256=sha(folder/'MANIFEST.json'),files=len(m['files']))
    return out

def surface_support(vw,pred,cfg):
    """Export exactly the frozen accepted heads' surface evidence, no new width."""
    mask=np.zeros(len(vw),dtype=bool)
    if pred['status']!='ok':return mask
    roi=(vw[:,0]>=cfg['vmin'])&(vw[:,0]<=cfg['vmax'])&(vw[:,1]>=cfg['wmin'])&(vw[:,1]<=cfg['wmax'])
    for center in pred['pair']:
        delta=vw-center
        mask|=roi&(np.linalg.norm(delta,axis=1)<=np.hypot(cfg['surface_half_width_m'],cfg['surface_half_height_m']))&(abs(delta[:,0])<=cfg['surface_half_width_m'])&(abs(delta[:,1])<=cfg['surface_half_height_m'])
    return mask

def predict_cloud(cloud,row,axes,motion,cfg,contact):
    n=len(cloud.points);labels=np.zeros(n,dtype=np.uint8)
    result=dict(step1_status='refusal',step1_reason=motion.get('reason',''),step1_confidence=0.,step2_status='NOT_FOUND',step2_reason='running_pair_unavailable',step2_confidence=0.,motion=motion,axes=axes,
        source_points=n,step1_slab_points=0,step2_slab_points=0,overlap_points=0)
    if axes is None:return labels,result
    xyz=np.column_stack((cloud.x,cloud.y,cloud.z));pose=np.asarray(row['lidar_pose_in_folder']);local=(xyz-pose[:3,3])@pose[:3,:3]
    # Preserve STEP1's original operation order (u vector, then two-column vw).
    u=local@axes[:,0];ids=np.flatnonzero((u>=0)&(u<=8)&np.all(np.isfinite(local),axis=1));vw=local[ids]@axes[:,1:]
    p=detect_rails(vw,cfg,details=False);rails=p['pair'] if p['status']=='ok' else None
    result.update(step1_status=p['status'],step1_reason=p['reason'],step1_confidence=p['confidence'],step1_slab_points=len(ids),step1=p,rail_pair=rails)
    rail_mask=surface_support(vw,p,cfg)
    if rails is not None:assert int(np.sum(rail_mask))==p['surface_left']+p['surface_right'],'Export support differs from frozen head surface evidence'
    labels[ids[rail_mask]]=1
    # STEP2's frozen preparation used the full matrix multiplication. Its ROI
    # is inside the historical broad cache, so that cache is unnecessary here.
    uvw=local@axes;cid=np.flatnonzero((uvw[:,0]>=0)&(uvw[:,0]<=8)&np.all(np.isfinite(uvw),axis=1))
    q=contact.detect({'uvw':uvw[cid]},rails)
    chosen=cid[q['support_indices']] if q['status'] in ('LEFT','RIGHT') else np.empty(0,dtype=np.int64)
    overlap=int(np.count_nonzero(labels[chosen]));assert overlap==0,'Frozen rail head / contact support overlap: stop instead of assigning precedence'
    labels[chosen]=2
    result.update(step2_status=q['status'],step2_reason=q['diagnostics']['reason'],step2_confidence=q['confidence'],step2_side=q['side'],step2_v=q['anchor_v'],step2_w=q['anchor_w'],step2_slab_points=len(cid),
        step2_diagnostics=q['diagnostics'],step2_proposal_support_count=len(q['support_indices']),overlap_points=overlap)
    return labels,result

def layout(header):
    assert not header.are_points_compressed,'Only uncompressed LAS is supported'
    dtype=header.point_format.dtype();field='raw_classification' if header.point_format.id<6 else 'classification';mask=31 if header.point_format.id<6 else 255
    return dtype,header.offset_to_point_data,header.point_count,field,mask

def patch_bytes(raw,header,labels):
    dtype,offset,n,field,mask=layout(header);assert len(labels)==n and np.all(labels<=2)
    output=bytearray(raw);points=np.ndarray((n,),dtype=dtype,buffer=output,offset=offset)
    points[field]=(points[field]&(255^mask))|labels
    return output

def check_output(source,output,header,expected=None):
    dtype,offset,n,field,mask=layout(header);assert len(source)==len(output)
    before=np.frombuffer(source,dtype=dtype,count=n,offset=offset);after=np.frombuffer(output,dtype=dtype,count=n,offset=offset)
    labels=(after[field]&mask).copy();assert np.all(labels<=2)
    if expected is not None:np.testing.assert_array_equal(labels,expected)
    # All header/VLR/EVLR/trailing bytes AND every other point field must match.
    restored=bytearray(output);restore=np.ndarray((n,),dtype=dtype,buffer=restored,offset=offset)
    assert np.array_equal(restore[field]&(255^mask),before[field]&(255^mask)),'Classification flags changed'
    restore[field]=before[field]
    assert digest(restored)==digest(source),'Bytes other than classification changed'
    return np.bincount(labels,minlength=3).tolist()

def safe_destination(source,dest):
    source=Path(source).resolve();dest=Path(dest).resolve();workspace=(ROOT/'output').resolve()
    dest.relative_to(workspace)
    assert dest!=source and not dest.is_relative_to(source) and not source.is_relative_to(dest),'Source/destination must be separate siblings'
    return source,dest
