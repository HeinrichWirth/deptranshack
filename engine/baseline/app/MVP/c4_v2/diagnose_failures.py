"""Fixed-config full-native replays of worst held-out regressions; no tuning."""
from . import ROOT,OUT
from .native_engine import NativeEngine
from .evaluate import read,matched
from gt_reference import future_pair_reference
import long_common as lc
import numpy as np
import time


def main():
    root=OUT/'benchmark/locked_r8_o8';groups={}
    for r in lc.csv_read(root/'matched_stations.csv'):
        groups.setdefault((r['run'],int(r['frame'])),[]).append(r)
    rank=sorted(groups,key=lambda key:np.percentile([float(r['far_variant']) for r in groups[key]],95)-np.percentile([float(r['far_reference']) for r in groups[key]],95),reverse=True)[:16]
    rows=[];config=lc.load(OUT/'selected_config.json')
    for run,i in rank:
        engine=NativeEngine(config['threads'],config['iterations'],config['grid']);engine.start_run(lc.dataset()/run);result=engine.process_frame(i);engine.close()
        p=result['prediction'];folder=OUT/'failure_diagnostics'/run/f'{i:06d}';folder.mkdir(parents=True,exist_ok=True)
        if p is not None:np.savez_compressed(folder/'native_full.npz',**p)
        lc.save(folder/'PREDICTION_COMPLETE.json',dict(time_ns=time.time_ns(),files={x.name:lc.sha(x) for x in folder.glob('*.npz')},diagnostic_selection_only=True,configuration_changed=False))
        if p is None:rows.append(dict(run=run,frame=i,full_native_available=False));continue
        source=root/run/f'{i:06d}';a=read(source/'reference.npz');b=read(source/'prediction.npz');ref=future_pair_reference(run,i,folder)
        ep=matched(a,b,ref);en=matched(a,p,ref);valid=ep['valid']&en['valid'];valid&=(ep['ranges']>=30)&(ep['ranges']<100)
        if not valid.any():continue
        offset=(ep['variant_pair'][:,1]-ep['variant_cr'])-(a['pair'][:,1]-a['C']);normal=a['B'][:,:,0];offset-=np.einsum('ij,ij->i',offset,normal)[:,None]*normal
        cr=np.linalg.norm(ep['variant_cr']-a['C'],axis=1);relative=np.linalg.norm(offset,axis=1)
        row=dict(run=run,frame=i,full_native_available=True,stations=int(valid.sum()),reference_far_p95=float(np.percentile(ep['reference'][valid,1],95)),persistent_far_p95=float(np.percentile(ep['variant'][valid,1],95)),native_full_far_p95=float(np.percentile(en['variant'][valid,1],95)),cr_disagreement_p95=float(np.nanpercentile(cr[valid],95)),relative_far_cr_disagreement_p95=float(np.nanpercentile(relative[valid],95)))
        row['dominant_difference']='relative rail/CR seed continuation' if row['relative_far_cr_disagreement_p95']>row['cr_disagreement_p95'] else 'persistent CR geometry'
        rows.append(row);print('FAILURE_DIAGNOSTIC',run,i,row,flush=True)
    lc.csv_write(OUT/'failure_diagnosis.csv',rows)

if __name__=='__main__':main()
