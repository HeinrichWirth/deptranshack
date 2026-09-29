"""RAW partial-chain gate, separate from offline geometry/GT forensic audit."""
from . import ROOT
from .pipeline import Pipeline, write_frame
import long_common as lc
import numpy as np
import time
import json
import hashlib
import cProfile
import pstats
import io

OUT = ROOT/'results_final_pipeline'


def normalized(value):
    if isinstance(value, dict):
        return {k:normalized(v) for k,v in value.items() if not k.endswith(('_ms','_ns'))}
    if isinstance(value, np.ndarray):
        return dict(dtype=str(value.dtype), shape=list(value.shape), sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value, (list, tuple)):
        return [normalized(v) for v in value]
    if isinstance(value, np.generic):
        return normalized(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def exact(a,b,path='root'):
    if isinstance(a,dict):
        for k,v in a.items():
            if k.endswith(('_ms','_ns')):
                continue
            assert k in b, (path,k,'missing')
            exact(v,b[k],path+'.'+k)
    elif isinstance(a,np.ndarray):
        np.testing.assert_array_equal(a,np.asarray(b),err_msg=path)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),(path,len(a),len(b))
        for i,(v,w) in enumerate(zip(a,b)):
            exact(v,w,path+f'[{i}]')
    elif isinstance(a,(float,np.floating)):
        assert a==b or (b is not None and np.isnan(a) and np.isnan(b)),(path,a,b)
    else:
        assert a==b,(path,a,b)


def main():
    rows=[];hashes={};counts=[];distance=[]
    cases=lc.load(ROOT/'results_final_geometry_rescue/protocol.json')['screen']
    for r in cases:
        pipeline=Pipeline(dict(engine='baseline',partial_audit=True,pose_policy='frozen_delayed_pose_audit',mode='online'))
        pipeline.start_run(lc.dataset()/r['run'])
        result=pipeline.process_frame(r['frame'])
        p,c=result['c4'],result['cloud']
        folder=ROOT/'results_contact_marching_long_range/phase_d_combinations/C4'/lc.key(r['run'],r['frame'])
        lc.check_marker(folder)
        expected=lc.read_prediction(folder)
        exact(p,expected)
        with np.load(folder/'provenance.npz') as z:
            for field in ('xyz','source_frame','source_row','source_point_index'):
                np.testing.assert_array_equal(c[field][p['indices']],z[field],err_msg=field)
        prov=result['provenance']
        assert np.all(prov['source_frame']<=r['frame'])
        assert np.all(prov['used_by_stage_mask']<8)
        row=dict(result['summary'],c4_exact=True,raw_point_provenance_exact=True)
        rows.append(row)
        hashes[lc.key(r['run'],r['frame'])]=dict(c4=normalized(p),provenance=normalized(prov))
        counts.append(dict(run=r['run'],frame=r['frame'],raw_N=row['source_points'],
            STEP1_N=int(np.sum(prov['first_found_stage']==1)),STEP2_N=int(np.sum(prov['first_found_stage']==2)),
            C4_N=int(np.sum(prov['first_found_stage']==3)),unique_selected_N=len(prov),
            semantics='unique within this start, not deduplicated across the run'))
        key=(prov['source_frame'].astype('uint64')<<np.uint64(32))|prov['source_row'].astype('uint64')
        selected_key=c['keys'][p['indices']]
        found=prov['first_found_stage'][np.searchsorted(key,selected_key)]
        d=c['sensor_distance_at_T'][p['indices']]
        for lo,hi in ((0,10),(10,20),(20,30),(30,40),(40,50),(50,75),(75,100)):
            for stage in (2,3):
                distance.append(dict(run=r['run'],frame=r['frame'],lo=lo,hi=hi,first_found_stage=stage,
                    N=int(np.sum((d>=lo)&(d<hi)&(found==stage))),distance='Euclidean sensor distance at T'))
        write_frame(OUT/'baseline_runtime'/lc.key(r['run'],r['frame']),result)
        lc.save(OUT/'BASELINE_RUNTIME.json',dict(scope='RAW_TO_C4_ONLY_NOT_FINAL_GEOMETRY',rows=rows))
        print('RAW_EQUIVALENT',len(rows),r['run'],r['frame'],round(row['timing']['T_TOTAL'],3),'seconds',flush=True)
    lc.save(OUT/'BASELINE_OUTPUT_HASHES.json',hashes)
    lc.csv_write(OUT/'point_stage_counts.csv',counts)
    lc.csv_write(OUT/'stage_distance_counts.csv',distance)
    # A distinct profiler call is deliberately excluded from latency statistics.
    (OUT/'profiles').mkdir(exist_ok=True)
    r=cases[0];pipeline=Pipeline(dict(engine='baseline',partial_audit=True,pose_policy='frozen_delayed_pose_audit',mode='research'))
    pipeline.start_run(lc.dataset()/r['run'])
    profile=cProfile.Profile();profile.enable()
    result=pipeline.process_frame(r['frame'])
    profile.disable();profile.dump_stats(str(OUT/'profiles/raw_to_c4.prof'))
    assert normalized(result['c4'])==hashes[lc.key(r['run'],r['frame'])]['c4']
    assert normalized(result['provenance'])==hashes[lc.key(r['run'],r['frame'])]['provenance']
    for ordering in ('cumulative','tottime'):
        stream=io.StringIO();pstats.Stats(profile,stream=stream).sort_stats(ordering).print_stats(50)
        (OUT/'profiles'/f'top50_{ordering}.txt').write_text(stream.getvalue(),encoding='utf-8')
    lc.save(OUT/'audit/raw_gate.json',dict(cases=len(rows),c4_bitwise_equal=True,
        discrete_equal=True,selected_points_exact=True,mode_parity_cases=1,mode_parity=True,
        profiling_output_parity=True,final_geometry_available=False,
        scope='delayed T+1 pose RAW partial chain; not strict online, not complete E2E'))


if __name__=='__main__':main()
