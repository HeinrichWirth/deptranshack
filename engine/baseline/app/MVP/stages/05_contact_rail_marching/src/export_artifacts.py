"""Scientific LAS copies: original classes retained, predictions in Extra Bytes."""
from common import *
from evaluation import load_prediction
import gzip,copy

EXTRAS=[('pred_cr','u1'),('pred_step','u2'),('pred_confidence','f4'),('pred_local_s','f4'),('pred_template_residual','f4'),('pred_anchor_error_eval','f4')]

def export_las(case):
    r=case['row'];run=r['run'];i=r['start_frame'];k=key(run,i);row=frames(run)[i]
    source=dataset()/run/row['file'];before=sha(source);cloud=laspy.read(source);pred=load_prediction(OUT/'heldout'/k);ids=pred['indices'];xyz=np.load(OUT/'cache'/k/'xyz.npy')
    ev=load(OUT/'heldout'/k/'evaluation.json');errors={s['step_index']:s.get('GT_anchor_error') for s in ev['steps']}
    overlay=laspy.LasData(copy.deepcopy(cloud.header));overlay.points=cloud.points.copy()
    overlay.add_extra_dims([laspy.ExtraBytesParams(name=name,type=typ) for name,typ in EXTRAS])
    overlay.pred_cr=np.zeros(len(cloud.points),dtype=np.uint8);overlay.pred_cr[ids]=1
    overlay.pred_step=np.full(len(cloud.points),65535,dtype=np.uint16);overlay.pred_step[ids]=pred['point_step']
    for name in ('pred_confidence','pred_local_s','pred_template_residual','pred_anchor_error_eval'):overlay[name]=np.full(len(cloud.points),np.nan,dtype=np.float32)
    overlay.pred_confidence[ids]=pred['confidence'];overlay.pred_local_s[ids]=pred['s_from_seed'];overlay.pred_template_residual[ids]=pred['template_residual']
    overlay.pred_anchor_error_eval[ids]=np.array([errors.get(int(s)) if errors.get(int(s)) is not None else np.nan for s in pred['point_step']],dtype=np.float32)
    overlay.header.vlrs.append(laspy.VLR(user_id='DEPTRANS_STEP5',record_id=5,description='Research CR marching; map XYZ',record_data=json.dumps(dict(run=run,start_frame=i,coordinates='original registered map of this run',original_classification_preserved=True,prediction_in_extra_bytes=True,seed_step=0,unpredicted_step=65535,anchor_error_eval_only=True)).encode()))
    path=OUT/'las'/f'{k}_overlay.las';overlay.write(path)
    only=laspy.LasData(copy.deepcopy(overlay.header));only.points=overlay.points[ids].copy();map_path=OUT/'las'/f'{k}_predicted_cr_map.las';only.write(map_path)
    sensor=laspy.LasData(copy.deepcopy(only.header));sensor.header.offsets=np.zeros(3)
    sensor.points=laspy.ScaleAwarePointRecord(only.points.array.copy(),sensor.header.point_format,sensor.header.scales,np.zeros(3))
    sensor.x=xyz[ids,0];sensor.y=xyz[ids,1];sensor.z=xyz[ids,2]
    sensor.header.vlrs.append(laspy.VLR(user_id='DEPTRANS_STEP5',record_id=6,description='Coordinates sensor T override',record_data=b'XYZ in sensor T coordinates. All original non-XYZ point fields retained.'))
    sensor_path=OUT/'las'/f'{k}_predicted_cr_sensorT.las';sensor.write(sensor_path)
    restored=laspy.read(path)
    assert len(restored.points)==len(cloud.points)
    for field in cloud.points.array.dtype.names:np.testing.assert_array_equal(cloud.points.array[field],restored.points.array[field],err_msg=field)
    reread=laspy.read(map_path)
    for field in cloud.points.array.dtype.names:np.testing.assert_array_equal(cloud.points.array[field][ids],reread.points.array[field],err_msg=field)
    sr=laspy.read(sensor_path);xx=np.c_[sr.x,sr.y,sr.z];assert np.max(abs(xx-xyz[ids]))<=.000051
    assert sha(source)==before
    return dict(run=run,start_frame=i,source=source.as_posix(),source_sha256=before,source_points=len(cloud.points),predicted_points=len(ids),
                overlay=path.relative_to(OUT).as_posix(),predicted_map=map_path.relative_to(OUT).as_posix(),predicted_sensorT=sensor_path.relative_to(OUT).as_posix(),
                verification='all original fields bitwise equal; sensor XYZ <=0.051mm rounding; source unchanged',tags=';'.join(case['tags']))

def point_csv():
    path=OUT/'per_predicted_point.csv.gz';count=0
    if path.exists() and (OUT/'audit/point_csv.json').exists():return
    fields=['run','start_frame','source_row','source_point_index','source_frame_index','march_step','sensor_x','sensor_y','sensor_z','range_from_lidar','s_from_seed','local_u','local_v','local_w','template_residual','confidence','anchor_v','anchor_w','future_or_current_seed_GT_distance','GT_evaluable']
    with gzip.open(path,'wt',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(fields)
        for folder in sorted((OUT/'heldout').iterdir()):
            if not (folder/'prediction.json').exists():continue
            pred=load_prediction(folder)
            if not len(pred['indices']):continue
            run=pred['run'];i=pred['start_frame'];xyz=np.load(OUT/'cache'/key(run,i)/'xyz.npy');row=frames(run)[i];cloud=laspy.read(dataset()/run/row['file'])
            with np.load(folder/'point_evaluation.npz') as z:distance=z['distance'];valid=z['evaluable']
            pointid=np.asarray(cloud.point_index);frameid=np.asarray(cloud.frame_index)
            for j,ix in enumerate(pred['indices']):
                p=xyz[ix];uv=pred['local_uvw'][j];a=pred['point_anchor'][j]
                vals=[run,i,int(ix),int(pointid[ix]),int(frameid[ix]),int(pred['point_step'][j]),*p,float(np.linalg.norm(p)),pred['s_from_seed'][j],*uv,pred['template_residual'][j],pred['confidence'][j],*a,distance[j],bool(valid[j])]
                writer.writerow(['' if isinstance(x,(float,np.floating)) and not np.isfinite(x) else x for x in vals]);count+=1
    save(OUT/'audit/point_csv.json',dict(rows=count,file=path.name,bytes=path.stat().st_size))

def main():
    examples=load(OUT/'gallery/examples.json');chosen=[]
    for tag in ('best','median','early','sparsity','curve','grade','wrong','gtgap','orientation'):
        for e in [x for x in examples if tag in x['tags']][:3]:
            if e not in chosen:chosen.append(e)
    for e in examples:
        if len(chosen)>=15:break
        if e not in chosen:chosen.append(e)
    rows=[]
    for e in chosen:rows.append(export_las(e));print('LAS',len(rows),len(chosen),flush=True)
    csv_write(OUT/'las_exports.csv',rows);save(OUT/'las/index.json',rows);point_csv()
if __name__=='__main__':main()
