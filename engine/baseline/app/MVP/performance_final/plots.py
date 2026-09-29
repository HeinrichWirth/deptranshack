from . import ROOT,OUT
import sys
sys.path.insert(0,str(ROOT/'results_rail2d/_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import long_common as lc
from .run_full import RUNS


def main():
    fig,a=plt.subplots(2,2,figsize=(15,10),layout='constrained');r=lc.csv_read(OUT/'stateful_availability.csv')
    short=['straight','curved','platform/switch','01a2','01d1'];x=np.arange(len(r));bottom=np.zeros(len(r))
    for key,label,color in [('fresh','fresh','#2076ba'),('reuse_low_motion','reuse <0.5 m','#25a568'),('stale','failed update → stale','#eb9a21'),('true_no_geometry','no state','#c14c56')]:
        v=np.array([float(y[key])/int(y['raw_frames'])*100 for y in r]);a[0,0].bar(x,v,bottom=bottom,label=label,color=color);bottom+=v
    a[0,0].set_xticks(x,short);a[0,0].set(ylabel='% raw frames',title='Five complete runs: state availability');a[0,0].legend(fontsize=9)
    vals=lc.csv_read(OUT/'python_optimization_ablation.csv');x=np.arange(len(vals))
    a[0,1].bar(x-.17,[float(r['median_ms']) for r in vals],.34,label='median');a[0,1].bar(x+.17,[float(r['p95_ms']) for r in vals],.34,label='p95')
    a[0,1].set_xticks(x,[r['backend'] for r in vals],rotation=18);a[0,1].set(ylabel='ms / solve',title='Clean serial ablation, same 32 starts');a[0,1].legend()
    runs=lc.csv_read(OUT/'scheduler_runs.csv');x=np.arange(5)
    for shift,mode,color in [(-.18,'OLD','#7b8ea3'),(.18,'NEW','#227fc0')]:
        rr=[next(r for r in runs if r['run']==run and r['mode']==mode) for run in RUNS]
        a[1,0].bar(x+shift,[float(r['effective_CPU_ms_raw_frame']) for r in rr],.36,color=color,label=mode)
    a[1,0].set_xticks(x,short);a[1,0].set(ylabel='process CPU ms / raw frame',title='OLD cheap refusals vs NEW useful solves + reuse');a[1,0].legend()
    for j,run in enumerate(RUNS):
        rows=lc.load(OUT/'runs/scheduler_spatial'/run/'INFERENCE_COMPLETE.json')['rows']
        distance=np.array([r['distance_since_geometry_update'] if r['distance_since_geometry_update'] is not None else np.nan for r in rows])
        a[1,1].plot(np.arange(len(rows)),distance,lw=1,label=short[j])
    a[1,1].axhline(.5,c='black',ls=':',lw=1);a[1,1].set(ylabel='m since last successful update',xlabel='raw frame ordinal',title='State freshness: failed refresh remains explicit');a[1,1].legend(fontsize=9)
    for ax in a.flat:ax.grid(axis='y',alpha=.15)
    target=OUT/'performance_overview.png';fig.savefig(target,dpi=150);plt.close(fig)
    path=OUT/'REPORT_PERFORMANCE_FINAL.html';page=path.read_text(encoding='utf-8')
    page=page.replace('</main>','<p><a href="performance_overview.png"><img src="performance_overview.png" alt="Measured performance and state availability"></a></p></main>')
    path.write_text(page,encoding='utf-8')


if __name__=='__main__':main()
