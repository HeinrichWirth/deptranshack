"""Offline display export of current raw T and its already computed prediction."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,sqlite3,time,hashlib,argparse
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path[:0]=[str(ROOT/'.runtime'),str(HERE/'adapter')]
import numpy as np
import laspy
from cdr_cloud import decode

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='INPUT10HZ_SOLVE5_REPLAY');parser.add_argument('--destination',default='results_copy_main_10hz_train_las');parser.add_argument('--id',default='COPY_MAIN_10HZ');parser.add_argument('--label',default='COPY_MAIN · 10 Гц · каждый 5-й кадр');args=parser.parse_args()
    source=(HERE/'results'/args.run/'payload').resolve();dest=(ROOT/args.destination).resolve()
    assert source.is_relative_to(HERE/'results') and dest.is_relative_to(ROOT) and dest!=ROOT
    dest.mkdir(exist_ok=True)
    if (dest/'index.json').exists():raise ValueError('Export already exists; do not overwrite')
    db=ROOT/'datas/test_synthetic/cloud_with_fake_obj/cloud_with_fake_obj_0.db3';before=db.stat()
    jobs=json.loads((source/'solves.json').read_text());mapping=json.loads((source/'source_index_map.json').read_text())['frames']
    frames=[];audit=[];begin=time.perf_counter();conn=sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True)
    for ordinal,job in enumerate(sorted(jobs,key=lambda r:r['frame'])):
        i=job['frame'];identity=mapping[i];original=identity['source_frame_index'];folder=source/'solves'/f'{i:06d}'
        blob=conn.execute('SELECT data FROM messages WHERE id=?',(identity['message_id'],)).fetchone()[0]
        sensor,points=decode(blob);xyz=np.column_stack([points[k] for k in ('x','y','z')]);ids=np.flatnonzero(np.isfinite(xyz).all(axis=1)&np.any(xyz!=0,axis=1))
        xyz=xyz[ids];n=len(ids);synthetic=[];layers=[];stations=[]
        summary=json.loads((folder/'summary.json').read_text());horizon=0.
        if summary['final_geometry_available']:
            with np.load(folder/'prediction.npz') as pred:
                s=pred['s'];horizon=float(s[-1]);grid=np.unique(np.r_[np.arange(float(s[0]),horizon,.1),s[-1]])
                for layer,values in ((21,pred['pair'][:,0]),(22,pred['pair'][:,1]),(20,pred['C']),(23,pred['pair'].mean(axis=1))):
                    synthetic.append(np.column_stack([np.interp(grid,s,values[:,k]) for k in range(3)]));layers.extend([layer]*len(grid));stations.extend(grid)
        m=len(layers);header=laspy.LasHeader(point_format=3,version='1.2');header.scales=np.full(3,1e-5);header.offsets=np.zeros(3)
        extra=[('display_layer','uint8'),('synthetic_geometry','uint8'),('forecast_station_m','float64'),('source_frame_index','int32'),('pipeline_frame_index','int32'),('point_index','uint32'),('source_row','int64'),('sensor_time_ns','int64'),('intensity_raw','float32'),('sensor_x','float32'),('sensor_y','float32'),('sensor_z','float32')]
        header.add_extra_dims([laspy.ExtraBytesParams(name=name,type=typ) for name,typ in extra])
        las=laspy.LasData(header,laspy.ScaleAwarePointRecord.zeros(n+m,header=header))
        allxyz=np.vstack([xyz,*synthetic]) if m else xyz;las.x,las.y,las.z=allxyz.T
        las.forecast_station_m=np.full(n+m,np.nan);las.source_row=np.r_[ids.astype('int64'),np.full(m,-1,dtype='int64')]
        las.source_frame_index=np.r_[np.full(n,original,dtype='int32'),np.full(m,-1,dtype='int32')]
        las.pipeline_frame_index=np.full(n+m,i,dtype='int32');las.point_index=np.r_[ids.astype('uint32'),np.full(m,2**32-1,dtype='uint32')]
        las.sensor_time_ns=np.r_[np.full(n,sensor['header_time_ns'],dtype='int64'),np.full(m,np.iinfo('int64').min,dtype='int64')]
        las.intensity_raw=np.r_[points['intensity'][ids],np.full(m,np.nan,dtype='float32')]
        for k,name in enumerate(('sensor_x','sensor_y','sensor_z')):las[name]=np.r_[xyz[:,k],np.full(m,np.nan,dtype='float32')]
        las.intensity[:n]=np.clip(np.nan_to_num(points['intensity'][ids]),0,65535).astype('uint16')
        display=np.zeros(n+m,dtype='uint8')
        with np.load(folder/'point_provenance.npz') as z:prov=z['rows'];prov=prov[prov['source_frame']==i]
        ids_to_row=np.full(len(points),-1,dtype='int64');ids_to_row[ids]=np.arange(n)
        for label,take in ((1,prov['first_found_stage']==1),(2,(prov['used_by_stage_mask']&6)!=0)):
            at=ids_to_row[prov['source_point_index'][take]];assert np.all(at>=0);display[at]=label
        display[n:]=layers;las.display_layer=display;las.classification=display
        las.synthetic_geometry[n:]=1;las.forecast_station_m[n:]=stations
        palette={0:(125,139,157),1:(72,202,139),2:(240,45,180),21:(0,200,255),22:(255,125,35),20:(255,210,45),23:(155,130,220)}
        for layer,rgb in palette.items():
            mask=display==layer
            for channel,c in zip(('red','green','blue'),rgb):las[channel][mask]=c*257
        provenance=dict(source_db=str(db.relative_to(ROOT)),source_message_id=identity['message_id'],source_frame_index=original,pipeline_frame_index=i,source_payload_sha256=hashlib.sha256(blob).hexdigest(),prediction_complete_sha256=sha(folder/'PREDICTION_COMPLETE.json'),real_clouds=[original],history_clouds_not_exported=True,coordinate_frame='current source LiDAR axes; origin is actual source LiDAR',xyz_quantization_m=1e-5,raw_XYZI_preserved_in_extra_fields=True)
        las.vlrs.append(laspy.VLR(user_id='DEPTRANS_VIEW',record_id=1,description='Raw T and saved forecast T',record_data=json.dumps(provenance).encode()))
        path=dest/f'frame_{original:06d}_with_predictions.las';las.write(path)
        check=laspy.read(path)
        assert len(check.points)==n+m
        np.testing.assert_array_equal(check.point_index[:n],ids)
        for k,name in enumerate(('sensor_x','sensor_y','sensor_z')):np.testing.assert_array_equal(check[name][:n],xyz[:,k])
        np.testing.assert_array_equal(check.intensity_raw[:n],points['intensity'][ids])
        np.testing.assert_allclose(np.column_stack((check.x,check.y,check.z)),allxyz,rtol=0,atol=5.001e-6)
        frames.append(dict(index=ordinal,pipeline_frame_index=i,source_frame_index=original,file=path.name,name=path.name,real_points=n,synthetic_points=m,available=bool(m),horizon=horizon,status=summary['status'],reason=summary['original_reason'],origin=[0,0,0],up=[0,0,1],timestamp_ns=str(sensor['header_time_ns']),source_clouds=[original],coordinate_frame='current source LiDAR axes',message_id=identity['message_id']))
        audit.append(dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),raw_fields_exact=True,real_points=n,synthetic_points=m,**provenance))
        if ordinal%20==0:print('LAS_EXPORT',ordinal+1,'/',len(jobs),flush=True)
    conn.close();after=db.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    save(dest/'index.json',dict(version=2,runs=[dict(id=args.id,label=args.label,role='ros2_replay',frames=frames)]))
    save(dest/'audit.json',dict(frames=audit,source_unchanged=True,seconds=time.perf_counter()-begin,total_bytes=sum(r['bytes'] for r in audit)))
    print('COMPLETE',len(frames),'LAS',sum(r['bytes'] for r in audit),'bytes',flush=True)
if __name__=='__main__':main()
