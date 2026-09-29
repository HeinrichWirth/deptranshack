from . import ROOT,OUT
from .engine import Engine
from .hotspots import Profile
import long_common as lc
import numpy as np
import cProfile,pstats,time

COHORT=[('roundT_squareT_pressureGate_squareT',100),('roundT_pressureGate_roundT',72),
        ('squareT_platform_squareT_switch',100),('new_data_part_01d1',550)]


def main():
    (OUT/'profiles').mkdir(parents=True,exist_ok=True);events=[];fits=[];optimizers=[];captures=[];summary=[];history=[]
    for run,i in COHORT:
        prof=Profile();e=Engine(profile=prof);e.start_run(lc.dataset()/run)
        original_read=e.cache.read
        def audited_read(k,index,allow_disk=True):
            metadata_cached=k in e.cache.source_meta
            part,timing=original_read(k,index,allow_disk)
            history.append(dict(run=run,frame=i,source_frame=k,points=len(part['xyz']),
                current_pose_transform_count=1,source_pose_transform_count=0 if metadata_cached else 1,
                metadata_cached=metadata_cached,**{name:timing[name] for name in ('read_ms','transform_ms')}))
            return part,timing
        e.cache.read=audited_read
        cp=cProfile.Profile();cp.enable();r=e.process_frame(i);cp.disable()
        cp.dump_stats(str(OUT/'profiles'/f'{run}__{i}.prof'))
        with (OUT/'profiles'/f'{run}__{i}_top50.txt').open('w',encoding='utf-8') as f:
            pstats.Stats(cp,stream=f).sort_stats('cumulative').print_stats(50);pstats.Stats(cp,stream=f).sort_stats('tottime').print_stats(50)
        for collection,target in ((prof.events,events),(prof.fits,fits),(prof.optimizers,optimizers)):
            target.extend(dict(run=run,frame=i,**v) for v in collection)
        captures.extend(prof.captures)
        summary.append(dict(run=run,frame=i,optimizers=len(prof.optimizers),fits=len(prof.fits),timing=r['summary']['timing']))
        print('PROFILE',run,i,'optimizers',len(prof.optimizers),'fits',len(prof.fits),flush=True)
    lc.save(OUT/'profiles/summary.json',summary);lc.csv_write(OUT/'c4_spatial_stats.csv',events)
    lc.csv_write(OUT/'history_transform_audit.csv',history)
    lc.csv_write(OUT/'c4_optimizer_stats.csv',optimizers);lc.csv_write(OUT/'profiles/fits.csv',fits)
    lc.save(OUT/'profiles/events.json',events)
    for j,c in enumerate(captures):
        np.savez_compressed(OUT/'profiles'/f'fit_{j:04d}.npz',points=c['points'],center=c['center'],template=c['template'],gate=c['gate'],fixed=c['fixed'])
    lc.save(OUT/'profiles/capture_config.json',dict(cfg=captures[0]['cfg'],count=len(captures)))
    profile_rows=[dict(operation=k,calls=len(rr),sum_seconds=sum(r['seconds'] for r in rr),
        median_ms=float(np.median([r['seconds'] for r in rr])*1000),p95_ms=float(np.percentile([r['seconds'] for r in rr],95)*1000),
        overlapping='inclusive wrappers overlap; never sum this table') for k in sorted(set(r['operation'] for r in events)) if (rr:=[r for r in events if r['operation']==k])]
    for name,key,rr in [('history_loading','T_LAS_IO',[r['timing'] for r in summary]),
                        ('history_transforms','T_TRANSFORMS',[r['timing'] for r in summary]),
                        ('least_squares','seconds',optimizers),('residual_callback','residual_seconds',optimizers)]:
        values=[r[key] for r in rr]
        profile_rows.append(dict(operation=name,calls=len(values),sum_seconds=sum(values),median_ms=float(np.median(values))*1000,
            p95_ms=float(np.percentile(values,95))*1000,overlapping='inclusive; history includes current and past'))
    for name in ('approx_derivative','pca','transport','infer','distances','fit'):
        hits=[]
        for path in (OUT/'profiles').glob('*.prof'):
            for (file,line,fn),v in pstats.Stats(str(path)).stats.items():
                if fn==name and (name not in ('distances','fit') or file.endswith('tracker.py')):hits.append(v)
        if hits:profile_rows.append(dict(operation=name,calls=sum(v[1] for v in hits),sum_seconds=sum(v[3] for v in hits),
            self_seconds=sum(v[2] for v in hits),median_ms=None,p95_ms=None,overlapping='cProfile cumulative; per-call durations unavailable'))
    lc.csv_write(OUT/'c4_profile.csv',profile_rows)


if __name__=='__main__':main()
