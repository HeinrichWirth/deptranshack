"""Separate full RAW profile, excluded from headline latency."""
from . import ROOT
from .pipeline import Pipeline
from .verify_raw import exact
import long_common as lc
import numpy as np
import cProfile
import pstats
import io

OUT=ROOT/'results_final_pipeline'


def main():
    profiler=cProfile.Profile();profiler.enable()
    p=Pipeline(dict(engine='cache'));p.start_run(lc.dataset()/'new_data_part_01a1');r=p.process_frame(330)
    profiler.disable();directory=OUT/'profiles';directory.mkdir(exist_ok=True)
    profiler.dump_stats(str(directory/'full_raw_pipeline.prof'))
    with np.load(OUT/'deployable_baseline_cache/new_data_part_01a1__000330/prediction.npz') as z:
        exact(r['prediction'],{k:z[k] for k in z.files})
    for ordering in ('cumulative','tottime'):
        text=io.StringIO();pstats.Stats(profiler,stream=text).sort_stats(ordering).print_stats(50)
        (directory/f'full_top50_{ordering}.txt').write_text(text.getvalue(),encoding='utf-8')
    stats=pstats.Stats(profiler);rows=[]
    for (file,line,name),(primitive,calls,selftime,cumulative,callers) in stats.stats.items():
        rows.append(dict(file=file,line=line,function=name,calls=calls,self_seconds=selftime,
            cumulative_seconds=cumulative,percent_profile_total=100*cumulative/stats.total_tt))
    lc.csv_write(directory/'full_function_times.csv',sorted(rows,key=lambda r:-r['cumulative_seconds']))
    lc.save(directory/'PROFILE_COMPLETE.json',dict(output_equivalent=True,scope='full RAW file+configuration+compute, profiling overhead, not headline benchmark',
        stage_times=r['summary']['timing'],profile_seconds=stats.total_tt))


if __name__=='__main__':main()
