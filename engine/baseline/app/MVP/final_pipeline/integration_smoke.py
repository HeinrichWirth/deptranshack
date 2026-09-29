"""Final API/label barrier and full timing-boundary checks."""
from . import ROOT
from .pipeline import Pipeline,write_frame
from .verify_raw import exact
import long_common as lc
import numpy as np
import laspy
import tempfile
from pathlib import Path
import time

OUT=ROOT/'results_final_pipeline'


def compare_saved(result,folder):
    for key,name in (('prediction','prediction.npz'),('c4_smooth','curve.npz')):
        with np.load(folder/name) as z:exact(result[key],{k:z[k] for k in z.files})


def main():
    times=[]
    cases=[('roundT_squareT_pressureGate_squareT',44,'deployable_benchmark_cache'),
        ('roundT_pressureGate_roundT',169,'deployable_curved_run_cache'),
        ('squareT_platform_squareT_switch',67,'deployable_benchmark_cache'),
        ('new_data_part_01d1',555,'deployable_development_cache')]
    for run,i,phase in cases:
        p=Pipeline(dict(mode='online'));p.start_run(lc.dataset()/run);r=p.process_frame(i)
        compare_saved(r,OUT/phase/lc.key(run,i))
        start=time.perf_counter();write_frame(OUT/'online_smoke'/lc.key(run,i),r,mode='online');serialization=time.perf_counter()-start
        row=dict(run=run,frame=i,**r['summary']['timing'],**p.startup_times,T_SERIALIZATION_FULL=serialization,
            T_DIAGNOSTICS=0.,T_LAS_EXPORT=0.,T_TOTAL_WITH_STARTUP_AND_SERIALIZATION=sum(p.startup_times.values())+r['summary']['timing']['T_TOTAL']+serialization)
        times.append(row)
        assert not (OUT/'online_smoke'/lc.key(run,i)/'c4.json').exists(),'Online emitted research debug tables'
    # Change annotation labels in a separate copied LAS, and deliberately omit
    # the T+1 cloud. The only supplied T+1 input is its sanitized pose metadata.
    run='roundT_squareT_pressureGate_squareT';records=lc.frames(run)
    base=Pipeline();base.start_run(lc.dataset()/run);a=base.process_frame(0)
    with tempfile.TemporaryDirectory(prefix='label_blind_',dir=OUT/'audit') as temp:
        folder=Path(temp).resolve();assert folder.is_relative_to((OUT/'audit').resolve())
        cloud=laspy.read(lc.dataset()/run/records[0]['file'])
        cloud.classification=(np.arange(len(cloud.points),dtype=np.uint32)%32).astype(np.uint8)
        cloud.write(folder/records[0]['file'])
        lc.save(folder/'probe.frames.json',dict(frames=records[:2]))
        test=Pipeline();test.start_run(folder);b=test.process_frame(0)
        for key in ('step1','step2','seed','c4','prediction','c4_smooth'):exact(a[key],b[key])
        np.testing.assert_array_equal(a['provenance'],b['provenance'])
        assert not (folder/records[1]['file']).exists()
    lc.save(OUT/'audit/final_api_checks.json',dict(timing_cases=4,old_predictions_bitwise_equal=True,
        annotation_permutation_no_effect=True,T_plus_1_cloud_absent_and_not_needed=True,
        generic_run_directory_supported=True,online_has_no_research_debug=True,
        evidence='real RAW frame 0 copied to a new directory with changed classes, unchanged measurements, and only two pose rows'))
    lc.save(OUT/'timing_supplement.json',dict(scope='four successful RAW cases, full startup and serialization boundaries; not substituted for the 32-start headline benchmark',rows=times))
    lc.csv_write(OUT/'timing_supplement.csv',times)


if __name__=='__main__':main()
