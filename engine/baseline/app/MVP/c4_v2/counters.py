"""Measured native call counters include validation, history ingestion and march."""
from . import OUT
from .persistent import PersistentEngine
import long_common as lc


class MarchProxy:
    def __init__(self,target):self.target=target;self.ingest=0
    def add_frame(self,*args):self.ingest+=1;return self.target.add_frame(*args)
    def __getattr__(self,key):return getattr(self.target,key)


def main():
    rows=[];config=lc.load(OUT/'selected_config.json')
    for run in ('roundT_pressureGate_roundT','roundT_squareT_pressureGate_squareT','squareT_platform_squareT_switch'):
        e=PersistentEngine(**config);e.engine.marcher=MarchProxy(e.engine.marcher);e.start_run(lc.dataset()/run)
        for i in range(45):
            a=e.engine.marcher.stats();v=e.validation_solver.stats();ingest=e.engine.marcher.ingest
            r=e.process_frame(i);b=e.engine.marcher.stats();w=e.validation_solver.stats()
            row=dict(run=run,frame=i,status=r['persistent_stats']['status'],native_march_calls=b['native_updates']-a['native_updates'],native_ingest_calls=e.engine.marcher.ingest-ingest,validation_batch_calls=w['python_native_calls']-v['python_native_calls'])
            row['python_native_calls']=row['native_march_calls']+row['native_ingest_calls']+row['validation_batch_calls']
            for key in ('candidate_fits','residual_evaluations','native_tasks'):row[key]=b[key]-a[key]+w[key]-v[key]
            row['native_march_steps']=b['native_march_steps']-a['native_march_steps'];rows.append(row)
        e.close()
    lc.csv_write(OUT/'native_counters.csv',rows)

if __name__=='__main__':main()
