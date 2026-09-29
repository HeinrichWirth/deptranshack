"""Real source records only. Classes/original fields untouched in derived copies."""
from fusion_common import *
from fusion import CausalSource,assemble

PROVENANCE=[('source_frame_delta','i1'),('source_point_index','u4'),('age_seconds','f4'),('age_distance','f4')]
PREDICTED=[('pred_cr','u1'),('pred_step','u2'),('pred_confidence','f4')]

def original_records(run,c):
    parts=[];header=None;records=[]
    for frame in dict.fromkeys(c['source_frame'].tolist()):
        row=frames(run)[frame];path=dataset()/run/row['file'];cloud=laspy.read(path);ids=np.flatnonzero(c['source_frame']==frame)
        if header is None:header=copy.deepcopy(cloud.header)
        assert cloud.points.array.dtype==header.point_format.dtype();np.testing.assert_array_equal(cloud.header.scales,header.scales);np.testing.assert_array_equal(cloud.header.offsets,header.offsets)
        parts.append((ids,cloud.points.array[c['source_row'][ids]].copy()));records.append(dict(source=str(path),sha256=sha(path)))
    array=np.empty(len(c['xyz']),dtype=header.point_format.dtype())
    for ids,p in parts:array[ids]=p
    return header,array,records

def write_cloud(path,header,original,c,pred=None):
    cloud=laspy.LasData(copy.deepcopy(header));cloud.points=laspy.ScaleAwarePointRecord(original.copy(),header.point_format,header.scales,header.offsets)
    names=set(cloud.point_format.dimension_names);extra=PROVENANCE+(PREDICTED if pred is not None else [])
    cloud.add_extra_dims([laspy.ExtraBytesParams(name=n,type=t) for n,t in extra if n not in names]);age=c['age_frames'];assert np.max(age)<=127
    cloud.source_frame_delta=-age.astype(np.int8);cloud.source_point_index=c['source_point_index'];cloud.age_seconds=c['age_seconds'];cloud.age_distance=c['age_distance']
    if pred is not None:
        ids=pred['indices'];cloud.pred_cr=np.zeros(len(original),dtype=np.uint8);cloud.pred_cr[ids]=1;cloud.pred_step=np.full(len(original),65535,dtype=np.uint16);cloud.pred_step[ids]=pred['point_step'];cloud.pred_confidence=np.full(len(original),np.nan,dtype=np.float32);cloud.pred_confidence[ids]=pred['confidence']
    cloud.header.vlrs.append(laspy.VLR(user_id='DEPTRANS_5_1',record_id=51,description='Causal fusion; original classes',record_data=b'Original registered-map XYZ and fields; causal source provenance in Extra Bytes. Prediction never overwrites classification.'))
    cloud.write(path);read=laspy.read(path)
    for field in original.dtype.names:np.testing.assert_array_equal(original[field],read.points.array[field],err_msg=field)
    np.testing.assert_array_equal(read.source_frame_delta,-age.astype(np.int8));return path.relative_to(OUT).as_posix()

def main():
    result=[]
    for e in load(OUT/'las/examples.json'):
        r=e['row'];run=r['run'];i=r['start_frame'];name=r['variant'];stem=name+'__'+key(run,i);pred=load_prediction(OUT/'heldout'/name/key(run,i));seed=seed_from_cache(run,i);c,_=assemble(CausalSource(run,i),pred['fusion']['config'],seed);header,original,sources=original_records(run,c)
        raw=write_cloud(OUT/'las'/(stem+'_fused_input_map.las'),header,original,c)
        overlay=write_cloud(OUT/'las'/(stem+'_prediction_overlay.las'),header,original,c,pred)
        current,bseed=assemble(CausalSource(run,i),dict(frames=1),seed);h,a,src=original_records(run,current);baseline=load_prediction(OUT/'heldout/F0'/key(run,i));comparison=write_cloud(OUT/'las'/(stem+'_current_only_comparison.las'),h,a,current,baseline)
        for source in sources:assert sha(Path(source['source']))==source['sha256']
        result.append(dict(variant=name,run=run,start_frame=i,tags=';'.join(e['tags']),fused_input=raw,prediction_overlay=overlay,current_only_comparison=comparison,points=len(original),predicted=len(pred['indices']),sources=sources,original_fields_bitwise_verified=True));print('LAS',len(result),stem,flush=True)
    save(OUT/'las/index.json',result);csv_write(OUT/'las_exports.csv',result)
if __name__=='__main__':main()
