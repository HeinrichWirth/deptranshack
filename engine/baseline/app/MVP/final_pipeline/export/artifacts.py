"""Post-inference compact provenance + exactly two diagnostic LAS runs.

Original labels are read only here, for copying/auditing after inference.
"""
from .. import ROOT
import sys
sys.path.insert(0,str(ROOT/'MVP/stages/06_pipeline_performance/vendor'))
import pyarrow as pa
import pyarrow.parquet as pq
import long_common as lc
import numpy as np
import laspy
import time
from pathlib import Path
from geometry_math import cross_axes

OUT=ROOT/'results_final_pipeline'
EXTRAS={'original_classification':'uint8','first_found_stage':'uint8','used_by_stage_mask':'uint16',
    'pipeline_component':'uint8','synthetic_geometry':'uint8','source_point_index':'int64',
    'march_step':'int32','stage_confidence':'float32','pipeline_frame_index':'int32','source_frame_index':'int32'}


def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def folders_for(run):
    result={}
    # Full runs supersede sampled starts only when identical frozen config was used.
    for phase in ('deployable_development_cache','deployable_benchmark_cache','deployable_curved_run_cache'):
        directory=OUT/phase
        if not (directory/'INFERENCE_COMPLETE.json').exists():continue
        for folder in directory.glob(run+'__*'):
            if (folder/'PREDICTION_COMPLETE.json').exists():result[int(folder.name.rsplit('__',1)[1])]=folder
    return sorted(result.items())


def unique_provenance(items):
    chunks=[];found=[]
    for index,folder in items:
        lc.check_marker(folder)
        rows=npz(folder/'point_provenance.npz')['rows']
        if len(rows):chunks.append(rows);found.append(np.full(len(rows),index,np.int32))
    if not chunks:return None,None
    rows=np.concatenate(chunks);first_frame=np.concatenate(found)
    keys=(rows['source_frame'].astype('uint64')<<np.uint64(32))|rows['source_row'].astype('uint64')
    order=np.argsort(keys,kind='stable');rows=rows[order];keys=keys[order];first_frame=first_frame[order]
    starts=np.r_[0,np.flatnonzero(keys[1:]!=keys[:-1])+1]
    result=rows[starts].copy();result['used_by_stage_mask']=np.bitwise_or.reduceat(rows['used_by_stage_mask'],starts)
    return result,first_frame[starts]


def add_extras(las):
    existing=set(las.point_format.dimension_names)
    las.add_extra_dims([laspy.ExtraBytesParams(name=name,type=dtype) for name,dtype in EXTRAS.items() if name not in existing])


def distances_and_preview(run,records,rows,found):
    distance=np.empty(len(rows),float);sample=[];stages=[]
    for frame in np.unique(rows['source_frame']):
        mask=rows['source_frame']==frame;ids=rows['source_row'][mask]
        assert len(np.unique(rows['source_point_index'][mask]))==len(ids),'Duplicate point identity within source frame'
        path=lc.dataset()/run/records[int(frame)]['file']
        with laspy.open(path) as f:h=f.header
        raw=np.memmap(path,dtype=h.point_format.dtype(),mode='r',offset=h.offset_to_point_data,shape=(h.point_count,))
        xyz=np.column_stack([raw[a][ids].astype(float)*h.scales[j]+h.offsets[j] for j,a in enumerate(('X','Y','Z'))]);del raw
        indices=np.flatnonzero(mask);times=found[mask]
        for T in np.unique(times):
            take=times==T;pose=np.asarray(records[int(T)]['lidar_pose_in_folder'])
            distance[indices[take]]=np.linalg.norm((xyz[take]-pose[:3,3])@pose[:3,:3],axis=1)
        preview=(ids.astype('uint64')+np.uint64(int(frame)*17))%53==0
        sample.append(xyz[preview]);stages.append(rows['first_found_stage'][mask][preview])
    (OUT/'visualizations').mkdir(exist_ok=True)
    np.savez_compressed(OUT/'visualizations'/(run+'_real_preview.npz'),world=np.concatenate(sample),stage=np.concatenate(stages),
                        sampling_note=np.array('Deterministic 1/53 sample of unique selected points; visualization only'))
    return distance


def export_real(run,records,rows,first_frame,role):
    target=OUT/'las'/role/run;target.mkdir(parents=True,exist_ok=True)
    audit=[];counts=[];started=time.perf_counter()
    frames=rows['source_frame'] if rows is not None else np.empty(0,dtype=int)
    for i,record in enumerate(records):
        source=lc.dataset()/run/record['file'];dest=target/(Path(record['file']).stem+'_pipeline.las')
        original=laspy.read(source);n=len(original.points)
        raw_sha=lc.sha(source)
        selected=rows[frames==i] if rows is not None else None
        new=laspy.LasData(original.header.copy(),original.points.copy());add_extras(new)
        new.original_classification=np.asarray(original.classification,dtype=np.uint8)
        new.classification=np.zeros(n,dtype=np.uint8)
        new.first_found_stage=np.zeros(n,dtype=np.uint8);new.used_by_stage_mask=np.zeros(n,dtype=np.uint16)
        new.pipeline_component=np.zeros(n,dtype=np.uint8);new.synthetic_geometry=np.zeros(n,dtype=np.uint8)
        new.source_point_index=np.asarray(original.point_index,dtype=np.int64) if 'point_index' in original.point_format.dimension_names else np.arange(n,dtype=np.int64)
        new.pipeline_frame_index=np.full(n,i,dtype=np.int32);new.march_step=np.full(n,-1,dtype=np.int32)
        if 'source_frame_index' not in original.point_format.dimension_names:new.source_frame_index=np.full(n,i,dtype=np.int32)
        new.stage_confidence=np.full(n,np.nan,dtype=np.float32)
        if selected is not None and len(selected):
            ix=selected['source_row'];assert np.all(ix<n)
            np.testing.assert_array_equal(new.source_point_index[ix],selected['source_point_index'])
            new.classification[ix]=np.take(np.array([0,10,11,12],dtype=np.uint8),selected['first_found_stage'])
            for name in ('first_found_stage','used_by_stage_mask','pipeline_component','march_step'):new[name][ix]=selected[name]
            new.stage_confidence[ix]=selected['confidence'].astype(np.float32)
        new.write(dest)
        check=laspy.read(dest)
        assert len(check.points)==n
        for name in original.point_format.dimension_names:
            if name=='classification':continue
            np.testing.assert_array_equal(np.asarray(check[name]),np.asarray(original[name]),err_msg=name)
        np.testing.assert_array_equal(check.original_classification,original.classification)
        for name in EXTRAS:np.testing.assert_array_equal(np.asarray(check[name]),np.asarray(new[name]),err_msg=name)
        np.testing.assert_array_equal(check.classification,new.classification)
        assert lc.sha(source)==raw_sha,'Source LAS changed'
        audit.append(dict(run=run,frame=i,points=n,source_sha256=raw_sha,output_sha256=lc.sha(dest),
            output=str(dest.relative_to(OUT)),all_original_fields_exact=True,xyz_exact=True,original_classification_preserved=True))
        counts.append(dict(run=run,frame=i,raw_N=n,STEP1_N=int(np.sum(check.classification==10)),
            STEP2_N=int(np.sum(check.classification==11)),C4_N=int(np.sum(check.classification==12)),
            unique_selected_N=int(np.sum(check.classification!=0)),semantics='run-wide first chronological discovery'))
        if (i+1)%50==0:print('LAS_ROUNDTRIP',run,i+1,len(records),flush=True)
    lc.save(OUT/'audit'/f'las_{run}.json',dict(frames=len(audit),rows=audit,seconds=time.perf_counter()-started,passed=True))
    return counts


def geometry(run,records,items,role,writer):
    lasparts=[];counts=[];npzfolder=OUT/'provenance/geometry'/run;npzfolder.mkdir(parents=True,exist_ok=True)
    for index,folder in items:
        meta=lc.load(folder/'summary.json')
        if not meta['final_geometry_available']:continue
        p=npz(folder/'prediction.npz');curve=npz(folder/'curve.npz')
        pose=np.asarray(records[index]['lidar_pose_in_folder']);R,t=pose[:3,:3],pose[:3,3]
        q=int(p['q']);left=1 if q>0 else 0
        cb,cn=curve['B'][:,:,1],curve['B'][:,:,2];bb,nn=cross_axes(p['B'],p['state'][:,0])
        sigma=lambda cov,d:np.sqrt(np.maximum(np.einsum('ni,nij,nj->n',d,cov,d),0))
        pieces=[(4,20,4,curve['C'],sigma(curve['covariance'],cb),sigma(curve['covariance'],cn)),(5,21,5,p['pair'][:,left],p['sigma_lateral'][:,left],p['sigma_vertical'][:,left]),
                (6,22,5,p['pair'][:,1-left],p['sigma_lateral'][:,1-left],p['sigma_vertical'][:,1-left]),
                (7,23,6,p['center'],sigma(p['center_cov'],bb),sigma(p['center_cov'],nn))]
        # Store scientific native samples. Dense samples are visualization only.
        for component,classification,stage,points,sv,sw in pieces:
            world=points@R.T+t;n=len(points)
            table=pa.table(dict(run=pa.DictionaryArray.from_arrays(pa.array(np.zeros(n,dtype=np.int8)),pa.array([run])),
                frame=np.full(n,index,np.int32),station_s=p['s'],geometry_type=np.full(n,component,np.uint8),
                x=world[:,0],y=world[:,1],z=world[:,2],sigma_lateral=np.full(n,np.nan) if sv is None else sv,
                sigma_vertical=np.full(n,np.nan) if sw is None else sw,created_by_stage=np.full(n,stage,np.uint8)))
            writer.write_table(table)
            if role:
                s=np.unique(np.r_[np.arange(0,float(p['s'][-1]),.1),p['s'][-1]])
                dense=np.column_stack([np.interp(s,p['s'],world[:,j]) for j in range(3)])
                lasparts.append(dict(xyz=dense,frame=index,component=component,classification=classification,stage=stage))
                counts.append(dict(run=run,frame=index,component=component,samples=len(dense)))
    if role:
        n=sum(len(p['xyz']) for p in lasparts);header=laspy.LasHeader(point_format=3,version='1.2')
        header.scales=np.array([.001,.001,.001]);header.offsets=np.asarray(records[0]['lidar_pose_in_folder'])[:3,3]
        las=laspy.LasData(header);las.points=laspy.ScaleAwarePointRecord.zeros(n,header=header);add_extras(las)
        xyz=np.concatenate([p['xyz'] for p in lasparts]) if n else np.empty((0,3))
        las.x,las.y,las.z=xyz.T;las.synthetic=np.ones(n,dtype=np.uint8);las.synthetic_geometry=np.ones(n,dtype=np.uint8)
        las.classification=np.concatenate([np.full(len(p['xyz']),p['classification'],np.uint8) for p in lasparts]) if n else np.empty(0,np.uint8)
        las.first_found_stage=np.concatenate([np.full(len(p['xyz']),p['stage'],np.uint8) for p in lasparts]) if n else np.empty(0,np.uint8)
        las.pipeline_component=np.concatenate([np.full(len(p['xyz']),p['component'],np.uint8) for p in lasparts]) if n else np.empty(0,np.uint8)
        las.pipeline_frame_index=np.concatenate([np.full(len(p['xyz']),p['frame'],np.int32) for p in lasparts]) if n else np.empty(0,np.int32)
        las.source_point_index=np.full(n,-1,np.int64);las.march_step=np.full(n,-1,np.int32);las.stage_confidence=np.full(n,np.nan,np.float32)
        las.source_frame_index=np.full(n,-1,np.int32)
        dest=OUT/'las'/role/(run+'_pipeline_geometry.las');las.write(dest)
        check=laspy.read(dest);assert len(check.points)==n and np.all(check.synthetic_geometry==1)
        assert set(np.unique(check.classification))<={20,21,22,23}
        for field in las.points.array.dtype.names:np.testing.assert_array_equal(las.points.array[field],check.points.array[field],err_msg=field)
        lc.save(OUT/'audit'/f'geometry_las_{run}.json',dict(points=n,sha256=lc.sha(dest),roundtrip_exact=True,
            visual_sampling_m=.1,coordinate_quantization_m=.001,extrapolation=False))
    return counts


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--geometry-only',action='store_true');args=ap.parse_args()
    selected={r['run']:r['role'] for r in lc.load(OUT/'selected_las_runs.json')['runs']}
    runs=lc.load(ROOT/'results_latent_contact_reference/protocol.json')['split']
    allruns=runs['development']+runs['validation']
    (OUT/'provenance').mkdir(parents=True,exist_ok=True);(OUT/'las').mkdir(exist_ok=True)
    point_writer=None;geo_writer=None;counts=[];synthetic=[];point_summary=[];distance_counts=[]
    geometry_schema=pa.schema([('run',pa.dictionary(pa.int8(),pa.string())),('frame',pa.int32()),('station_s',pa.float64()),
        ('geometry_type',pa.uint8()),('x',pa.float64()),('y',pa.float64()),('z',pa.float64()),('sigma_lateral',pa.float64()),
        ('sigma_vertical',pa.float64()),('created_by_stage',pa.uint8())])
    geo_writer=pq.ParquetWriter(OUT/'geometry_provenance.parquet',geometry_schema,compression='zstd')
    if args.geometry_only:
        for run in allruns:geometry(run,lc.frames(run),folders_for(run),None,geo_writer)
        geo_writer.close()
        mark=lc.load(OUT/'audit/EXPORT_COMPLETE.json');mark['geometry_rows']=pq.ParquetFile(OUT/'geometry_provenance.parquet').metadata.num_rows
        mark['native_uncertainty_all_components_exported']=True;lc.save(OUT/'audit/EXPORT_COMPLETE.json',mark)
        return
    for run in allruns:
        items=folders_for(run);records=lc.frames(run);rows,found=unique_provenance(items)
        if run in selected:assert [i for i,_ in items]==list(range(len(records))),'Diagnostic run must be complete'
        if rows is not None:
            np.savez_compressed(OUT/'provenance'/(run+'.npz'),rows=rows,first_found_frame=found)
            distance=distances_and_preview(run,records,rows,found)
            n=len(rows);data={name:pa.array(rows[name]) for name in rows.dtype.names}
            data.update(source_run=pa.DictionaryArray.from_arrays(pa.array(np.zeros(n,dtype=np.int8)),pa.array([run])),
                source_file=pa.DictionaryArray.from_arrays(pa.array(rows['source_frame']),pa.array([r['file'] for r in records])),
                first_found_frame=pa.array(found),sensor_distance_at_first_found_T=pa.array(distance))
            table=pa.table(data)
            if point_writer is None:point_writer=pq.ParquetWriter(OUT/'point_provenance.parquet',table.schema,compression='zstd')
            point_writer.write_table(table)
            point_summary.append(dict(run=run,unique_selected=n,STEP1_N=int(np.sum(rows['first_found_stage']==1)),
                STEP2_N=int(np.sum(rows['first_found_stage']==2)),C4_N=int(np.sum(rows['first_found_stage']==3))))
            for lo,hi in ((0,10),(10,20),(20,30),(30,40),(40,50),(50,75),(75,100)):
                for stage in (2,3):distance_counts.append(dict(run=run,lo=lo,hi=hi,first_found_stage=stage,
                    unique_points=int(np.sum((distance>=lo)&(distance<hi)&(rows['first_found_stage']==stage))),
                    distance='Euclidean sensor distance at first discovery frame T'))
        if run in selected:counts.extend(export_real(run,records,rows,found,selected[run]))
        synthetic.extend(geometry(run,records,items,selected.get(run),geo_writer))
        print('PROVENANCE_EXPORTED',run,0 if rows is None else len(rows),flush=True)
    if point_writer:point_writer.close()
    geo_writer.close()
    lc.csv_write(OUT/'point_stage_counts_run_unique.csv',point_summary)
    lc.csv_write(OUT/'point_stage_counts_full_runs.csv',counts);lc.csv_write(OUT/'synthetic_counts.csv',synthetic)
    lc.csv_write(OUT/'stage_distance_counts_run_unique.csv',distance_counts)
    point_rows=pq.ParquetFile(OUT/'point_provenance.parquet').metadata.num_rows
    assert point_rows==sum(r['unique_selected'] for r in point_summary)
    lc.save(OUT/'audit/EXPORT_COMPLETE.json',dict(time_ns=time.time_ns(),point_rows=point_rows,
        selected_runs=list(selected),geometry_rows=pq.ParquetFile(OUT/'geometry_provenance.parquet').metadata.num_rows))


if __name__=='__main__':main()
