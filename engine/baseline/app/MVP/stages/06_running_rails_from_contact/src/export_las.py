"""Display-only synthetic rail samples + bitwise preserved observed C4 records."""
from rr_common import *
from scipy.interpolate import CubicSpline
from concurrent.futures import ProcessPoolExecutor
import copy

EXTRA=[('predicted_geometry','u1'),('rail_id','u1'),('station_s','f4'),('sigma_lateral','f4'),('sigma_vertical','f4'),('alpha','f4'),('alpha_sigma','f4'),('source_method','u2'),('cr_state','u1'),('observed_or_predicted','u1'),('src_frame','i4'),('src_row','i8')]

def original_records(run,c):
    header=None;arr=None;sources=[]
    for k in np.unique(c['source_frame']):
        path=dataset()/run/frames(run)[int(k)]['file']
        with laspy.open(path) as f:h=f.header
        if header is None:header=copy.deepcopy(h);arr=np.empty(len(c['xyz']),dtype=h.point_format.dtype())
        assert h.point_format.dtype()==header.point_format.dtype();np.testing.assert_array_equal(h.scales,header.scales);np.testing.assert_array_equal(h.offsets,header.offsets)
        raw=np.memmap(path,dtype=h.point_format.dtype(),mode='r',offset=h.offset_to_point_data,shape=(h.point_count,));ids=np.flatnonzero(c['source_frame']==k);arr[ids]=raw[c['source_row'][ids]];del raw
        sources.append(dict(path=str(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns))
    return header,arr,sources

def one(item):
    run,i=item['run'],item['frame'];name=item['method'];folder=OUT/item['group']/name/key(run,i)
    with np.load(folder/'prediction.npz') as z:p={k:z[k] for k in z.files}
    record=load(folder/'prediction.json');scale=load(OUT/'configs.json')[name]['corridor_scale']/record['config']['corridor_scale']
    old,cr=read_c4(run,i);h,raw,sources=original_records(run,cr);h=copy.deepcopy(h);present=set(h.point_format.dimension_names)
    for n,t in EXTRA:
        if n not in present:h.add_extra_dim(laspy.ExtraBytesParams(name=n,type=t))
    h.vlrs.append(laspy.VLR(user_id='DEPTRANS_STEP6',record_id=60,description='Synthetic rails explicitly marked',record_data=json.dumps(dict(predicted_geometry='0 original C4 selected LiDAR record, 1 synthetic rail',rail_id='0 CR, 1 near, 2 far',station_s='m, common CR station',alpha='radians in Bishop frame',source_method={1:name},observed_or_predicted='0 measured, 1 predicted geometry',cr_state={1:'tentative',2:'CR supported',3:'RAILS_INFERRED_FROM_CR_STATE_GAP'},synthetic_sampling='0.1 m, display only',zero_other_synthetic_fields='not measurements',GT='not included')).encode()))
    station=np.unique(np.r_[np.arange(0,p['s'][-1],.1),p['s'][-1]]);points=CubicSpline(p['s'],p['pair'],axis=0)(station);P=np.asarray(frames(run)[i]['lidar_pose_in_folder']);world=points.reshape(-1,3)@P[:3,:3].T+P[:3,3];n=len(world);nr=len(raw)
    syn=laspy.LasData(copy.deepcopy(h));syn.points=laspy.ScaleAwarePointRecord.zeros(n,header=h);syn.x=world[:,0];syn.y=world[:,1];syn.z=world[:,2];syn.classification=np.ones(n,dtype=np.uint8)
    syn.predicted_geometry=np.ones(n,dtype=np.uint8);syn.observed_or_predicted=np.ones(n,dtype=np.uint8);syn.rail_id=np.tile([1,2],len(station));syn.station_s=np.repeat(station,2);syn.source_method=np.ones(n,dtype=np.uint16);syn.src_frame=np.full(n,-1,dtype=np.int32);syn.src_row=np.full(n,-1,dtype=np.int64)
    for field in ('sigma_lateral','sigma_vertical'):
        syn[field]=np.column_stack([np.interp(station,p['s'],p[field][:,j])*scale for j in (0,1)]).ravel()
    syn.alpha=np.repeat(np.interp(station,p['s'],p['state'][:,0]),2);syn.alpha_sigma=np.repeat(np.interp(station,p['s'],p['alpha_sigma']),2);idx=np.minimum(np.searchsorted(p['s'],station),len(p['s'])-1);syn.cr_state=np.repeat(p['cr_state'][idx],2)
    real=laspy.LasData(copy.deepcopy(h));real.points=laspy.ScaleAwarePointRecord.zeros(nr,header=h)
    for field in raw.dtype.names:real.points.array[field]=raw[field]
    real.src_frame=cr['source_frame'].astype(np.int32);real.src_row=cr['source_row'].astype(np.int64);real.station_s=np.full(nr,np.nan);real.cr_state=old['point_state'];real.source_method=np.ones(nr,dtype=np.uint16)
    for f in ('sigma_lateral','sigma_vertical','alpha','alpha_sigma'):real[f]=np.full(nr,np.nan)
    overlay=laspy.LasData(copy.deepcopy(h));overlay.points=laspy.ScaleAwarePointRecord(np.concatenate((real.points.array,syn.points.array)),h.point_format,h.scales,h.offsets)
    dest=OUT/'las'/key(run,i);dest.mkdir(parents=True,exist_ok=True);outputs=[]
    for suffix,cloud in (('cr_input',real),('rails_predicted',syn),('overlay',overlay)):
        path=dest/(suffix+'.las');cloud.write(path);check=laspy.read(path)
        np.testing.assert_array_equal(check.predicted_geometry,cloud.predicted_geometry);np.testing.assert_array_equal(check.rail_id,cloud.rail_id)
        if suffix!='rails_predicted':
            for f in raw.dtype.names:np.testing.assert_array_equal(check.points.array[f][:nr],raw[f],err_msg=f)
        outputs.append(dict(run=run,frame=i,method=name,path=path.relative_to(OUT).as_posix(),points=len(cloud.points),bytes=path.stat().st_size,sha256=sha(path),original_fields_verified=suffix!='rails_predicted',synthetic_geometry_marked=True))
    for src in sources:
        stat=Path(src['path']).stat();assert (stat.st_size,stat.st_mtime_ns)==(src['size'],src['mtime_ns'])
    save(dest/'EXPORT.json',dict(case=item,files=outputs,sources=sources,world_coordinates='Original registered folder coordinate system',source_measurements_intact=True))
    return outputs

def main():
    items=load(OUT/'gallery/cases.json');selected=[]
    for tag in ('worst_far','median','best','high_roll','curve','grade'):
        selected+= [r for r in items if tag in r['tags'] and r['id'] not in {v['id'] for v in selected}][:5]
    assert len(selected)>=20;rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for j,rr in enumerate(pool.map(one,selected),1):rows+=rr;print('LAS',j,len(selected),flush=True)
    write_csv(OUT/'las_exports.csv',rows);save(OUT/'las/index.json',dict(cases=len(selected),files=rows))

if __name__=='__main__':main()
