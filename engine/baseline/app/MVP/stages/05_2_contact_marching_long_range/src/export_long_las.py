"""Derived LAS with exact original point fields and separate research state."""
from long_common import *
from artifact_data import case_data
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse

EXTRA=[('state','u1'),('step','u2'),('source_age','u1'),('confidence','f4'),('continuity_score','f4'),('template_score','f4'),('track_prior_score','f4'),('source_point_index','u4')]

def records(run,c):
    header=None;arr=None;sources=[]
    for k in np.unique(c['source_frame']):
        p=dataset()/run/frames(run)[int(k)]['file']
        with laspy.open(p) as f:h=f.header
        if header is None:header=copy.deepcopy(h);arr=np.empty(len(c['xyz']),dtype=h.point_format.dtype())
        assert header.point_format.dtype()==h.point_format.dtype();np.testing.assert_array_equal(h.scales,header.scales);np.testing.assert_array_equal(h.offsets,header.offsets)
        raw=np.memmap(p,dtype=h.point_format.dtype(),mode='r',offset=h.offset_to_point_data,shape=(h.point_count,));ids=np.flatnonzero(c['source_frame']==k)
        arr[ids]=raw[c['source_row'][ids]];del raw
        sources.append(dict(file=str(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns))
    return header,arr,sources

def write(path,h,original,c,p,sel,prediction=True):
    cloud=laspy.LasData(copy.deepcopy(h));cloud.points=laspy.ScaleAwarePointRecord(original[sel].copy(),h.point_format,h.scales,h.offsets)
    names=set(cloud.point_format.dimension_names);cloud.add_extra_dims([laspy.ExtraBytesParams(name=n,type=t) for n,t in EXTRA if n not in names])
    n=len(original);fields=dict(state=np.zeros(n,dtype=np.uint8),step=np.full(n,65535,dtype=np.uint16),source_age=c['age_frames'].astype(np.uint8),source_point_index=c['source_point_index'])
    for name in ('confidence','continuity_score','template_score','track_prior_score'):fields[name]=np.full(n,np.nan,dtype=np.float32)
    if prediction:
        ids=p['indices'];fields['state'][ids]=p['point_state'];fields['step'][ids]=p['point_step']
        for name in ('confidence','continuity_score','template_score','track_prior_score'):
            if name in p:fields[name][ids]=p[name]
    for name,x in fields.items():setattr(cloud,name,x[sel])
    cloud.header.vlrs.append(laspy.VLR(user_id='DEPTRANS_5_2',record_id=52,description='Research state; original classes',record_data=b'Original registered-map XYZ/fields unchanged. state: 0 unselected, 1 tentative, 2 confirmed. GAP has no points. step=65535 unselected. source_age=T-source_frame. No production release.'))
    cloud.write(path);check=laspy.read(path)
    for f in original.dtype.names:np.testing.assert_array_equal(check.points.array[f],original[f][sel],err_msg=f)
    for f,x in fields.items():np.testing.assert_array_equal(np.asarray(getattr(check,f)),x[sel],err_msg=f)
    return dict(path=path.relative_to(OUT).as_posix(),points=len(sel),bytes=path.stat().st_size,original_fields_bitwise_verified=True,sha256=sha(path))

def one(e):
    target=OUT/'las'/e['id'];manifest=target/'EXPORT.json'
    if manifest.exists():return load(manifest)
    p,c,ref,info=case_data(e);h,original,sources=records(e['run'],c);target.mkdir(parents=True,exist_ok=True)
    n=len(original);allids=np.arange(n);conf=p['indices'][p['point_state']==2];tent=p['indices'][p['point_state']==1]
    files=[]
    for suffix,ids,predict in [('input_T',np.flatnonzero(c['age_frames']==0),False),('fused_evidence',allids,False),('prediction_confirmed',conf,True),('prediction_tentative',tent,True),('overlay',allids,True)]:
        files.append(write(target/(suffix+'.las'),h,original,c,p,ids,predict))
    for s in sources:
        stat=Path(s['file']).stat();assert stat.st_size==s['size'] and stat.st_mtime_ns==s['mtime_ns']
    result=dict(id=e['id'],variant=e['variant'],run=e['run'],start_frame=e['start_frame'],files=files,sources=sources,tags=e['tags']);save(manifest,result);return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();items=load(OUT/'las/examples.json');results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        ff=[pool.submit(one,e) for e in items]
        for j,f in enumerate(as_completed(ff),1):
            r=f.result();results.append(r)
            if j%10==0 or j==len(ff):print('LAS',j,len(ff),r['id'],flush=True)
    save(OUT/'las/index.json',results);csv_write(OUT/'las_exports.csv',[dict(id=r['id'],variant=r['variant'],run=r['run'],start_frame=r['start_frame'],**f) for r in results for f in r['files']])

if __name__=='__main__':main()
