"""Representative exports. Original LAS records copied bit-for-bit by provenance."""
from lc_common import *
from curve_geometry import grid
import copy
from concurrent.futures import ProcessPoolExecutor

def records(run,pr):
    header=None;raw=None;sources=[]
    for source in np.unique(pr['source_frame']):
        path=dataset()/run/frames(run)[int(source)]['file']
        with frozen.laspy.open(path) as f:h=f.header
        if header is None:header=copy.deepcopy(h);raw=np.empty(len(pr['xyz']),dtype=h.point_format.dtype())
        assert h.point_format.dtype()==header.point_format.dtype();np.testing.assert_array_equal(h.scales,header.scales);np.testing.assert_array_equal(h.offsets,header.offsets)
        mapped=np.memmap(path,mode='r',dtype=h.point_format.dtype(),offset=h.offset_to_point_data,shape=(h.point_count,));ids=np.flatnonzero(pr['source_frame']==source);raw[ids]=mapped[pr['source_row'][ids]];del mapped;sources.append(dict(path=str(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns))
    return header,raw,sources

def one(item):
    run,i=item['run'],item['frame'];folder=OUT/item['group']/item['method']/item['id'];oldgroup='phase_b' if item['group']=='phase_d' else 'phase_e'
    with np.load(folder/'prediction.npz') as z:p={k:z[k] for k in z.files}
    with np.load(OLD_OUT/oldgroup/'B4_SPLINE_50__O1'/item['id']/'prediction.npz') as z:old={k:z[k] for k in z.files}
    c4,pr=read_c4(run,i);h,raw,sources=records(run,pr);h=copy.deepcopy(h);extras=[('geometry_type','u1'),('lc_rail_id','u1'),('lc_station','f4'),('lc_source_frame','i4'),('lc_source_row','i8'),('lc_cr_state','u1')]
    for name,typ in extras:
        assert name not in h.point_format.dimension_names,('New export field collides with source field',name)
        h.add_extra_dim(frozen.laspy.ExtraBytesParams(name=name,type=typ))
    h.vlrs.append(frozen.laspy.VLR(user_id='DEPTRANS_STEP61',record_id=61,description='Observed and synthetic geometry',record_data=json.dumps(dict(geometry_type={0:'original selected C4 LiDAR record',1:'old STEP6 CR curve',2:'new latent CR curve',3:'old running rail prediction',4:'new running rail prediction'},synthetic_points='Display-only samples at 0.1m by piecewise interpolation of saved predictions; zeroed measurement fields are not sensor values.',coordinates='Original registered recording frame; no parent-frame conversion.',original_fields='Bitwise preserved for geometry_type=0',GT_included=False,method=item['method'])).encode()))
    def blank(n):
        las=frozen.laspy.LasData(copy.deepcopy(h));las.points=frozen.laspy.ScaleAwarePointRecord.zeros(n,header=h);return las
    real=blank(len(raw))
    for field in raw.dtype.names:real.points.array[field]=raw[field]
    real.geometry_type=np.zeros(len(raw),np.uint8);real.lc_source_frame=pr['source_frame'];real.lc_source_row=pr['source_row'];real.lc_station=np.full(len(raw),np.nan);real.lc_cr_state=c4['point_state'];pose=np.asarray(frames(run)[i]['lidar_pose_in_folder'])
    def synthetic(pred,rail,kind):
        ss=grid(float(pred['s'][-1]),.1);curve=pred['pair'] if rail else pred['C'];shape=curve.shape[1:];flat=curve.reshape(len(curve),-1);xyz=np.column_stack([np.interp(ss,pred['s'],flat[:,j]) for j in range(flat.shape[1])]).reshape((-1,3));world=xyz@pose[:3,:3].T+pose[:3,3];las=blank(len(world));las.x=world[:,0];las.y=world[:,1];las.z=world[:,2];las.geometry_type=np.full(len(world),kind,np.uint8);las.classification=np.full(len(world),1 if rail else 2,np.uint8);las.lc_source_frame=np.full(len(world),-1,np.int32);las.lc_source_row=np.full(len(world),-1,np.int64);las.lc_station=np.repeat(ss,2) if rail else ss;las.lc_rail_id=np.tile([1,2],len(ss)) if rail else np.zeros(len(ss),np.uint8);return las
    clouds={'raw_cr_points':real,'old_c4_curve':synthetic(old,False,1),'new_latent_cr_curve':synthetic(p,False,2),'rails_old':synthetic(old,True,3),'rails_new':synthetic(p,True,4)};overlay=blank(0);overlay.points=frozen.laspy.ScaleAwarePointRecord(np.concatenate([v.points.array for v in clouds.values()]),h.point_format,h.scales,h.offsets);clouds['overlay']=overlay;dest=OUT/'las'/item['id'];dest.mkdir(parents=True,exist_ok=True);rows=[]
    for name,cloud in clouds.items():
        path=dest/(name+'.las');cloud.write(path);check=frozen.laspy.read(path);np.testing.assert_array_equal(check.geometry_type,cloud.geometry_type)
        if name in ('raw_cr_points','overlay'):
            for field in raw.dtype.names:np.testing.assert_array_equal(check.points.array[field][:len(raw)],raw[field],err_msg=field)
        rows.append(dict(run=run,frame=i,method=item['method'],file=name,path=path.relative_to(OUT).as_posix(),points=len(check.points),bytes=path.stat().st_size,sha256=sha(path),original_fields_verified=name in ('raw_cr_points','overlay'),synthetic_explicitly_marked=True))
    for source in sources:
        stat=Path(source['path']).stat();assert stat.st_size==source['size'] and stat.st_mtime_ns==source['mtime_ns']
    save(dest/'EXPORT.json',dict(case=item,files=rows,sources=sources,original_fields_preserved=True));return rows

def main():
    cases=load(OUT/'gallery/cases.json');chosen=[]
    for tag in ('old-failure','worst','best','median','curve','grade','sparse'):
        chosen +=[r for r in cases if tag in r['tags'] and r['id'] not in {x['id'] for x in chosen}][:11 if tag=='old-failure' else 3]
    rows=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for j,rr in enumerate(pool.map(one,chosen),1):rows+=rr;print('LAS',j,len(chosen),flush=True)
    write_csv(OUT/'las_exports.csv',rows);save(OUT/'las/index.json',dict(cases=len(chosen),files=rows))

if __name__=='__main__':main()
