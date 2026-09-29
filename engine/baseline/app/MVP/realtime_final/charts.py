"""Scientific summary charts; plotting only, never used by inference."""
from . import ROOT,OUT
import sys
sys.path.insert(0,str(ROOT/'results_rail2d/_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import long_common as lc

def main():
    m=lc.load(OUT/'METRICS.json');q=lc.load(OUT/'FINAL_LINUX_QUALITY.json');dest=OUT/'figures';dest.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(15,10));ax=axes[0,0]
    rr=[r for r in m['ablations'] if r.get('median') is not None];labels=['Original\nLAS','Memory +\nshared index','Selective\nmaterialization','Exact\nROI']
    ax.bar(labels,[r['median'] for r in rr],color=['#8c99aa','#6a86c0','#229ba0','#1c7964']);ax.scatter(labels,[r['p95'] for r in rr],color='#c74f3f',label='p95',zorder=4);ax.set(ylabel='Full solve, ms',title='32 paired development starts');ax.legend()
    ax=axes[0,1];rr=[r for r in m['workers'] if r['file'].startswith('development')];rr.sort(key=lambda r:(r['workers'],r['threads']));x=np.arange(len(rr))
    ax.bar(x-.18,[r['publish_hz'] for r in rr],.36,label='Geometry publications/s',color='#167d9a');ax.bar(x+.18,[r['solve_hz'] for r in rr],.36,label='Completed solves/s',color='#9db7c9');ax.set_xticks(x,[f'{r["workers"]}×{r["threads"]}' for r in rr]);ax.set(title='10 Hz input ≠ 10 Hz geometry',xlabel='Solve workers × candidate threads',ylabel='Hz');ax.legend(fontsize=8)
    ax=axes[1,0];rr=[r for r in q['ranges'] if r['component']=='far'];x=np.arange(len(rr));ax.bar(x-.18,[100*r['reference_p95'] for r in rr],.36,color='#8293aa',label='Linux reference');ax.bar(x+.18,[100*r['native_p95'] for r in rr],.36,color='#208c78',label='Linux final native');ax.set_xticks(x,[f'{r["lo"]}–{r["hi"]} m\n{r["stations"]} sections / {r["starts"]} starts' for r in rr]);ax.set(ylabel='Far rail p95, cm',title='240 starts, same evaluation sections');ax.legend()
    ax=axes[1,1];rr=lc.load(OUT/'performance_bind.json')['rows'];ax.scatter([r['counters']['native_roi_points'] for r in rr],[r['c4_ms'] for r in rr],c=[r['step_count'] for r in rr],cmap='viridis',s=45);ax.set(xlabel='Cumulative exact ROI rows / solve',ylabel='LIVE_MEMORY C4, ms',title='48 fixed starts: workload and time')
    for ax in axes.flat:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('FINAL REALTIME — exact geometry, measured runtime',fontsize=18);fig.tight_layout();fig.savefig(dest/'summary.png',dpi=145);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(15,8),sharex=True)
    files=sorted((OUT/'live').glob('confirmation*__2x4.json'))
    for p in files:
        r=lc.load(p);a=r['arrivals'];x=np.arange(len(a))*.1;y=[np.nan if v['remaining_horizon'] is None else v['remaining_horizon'] for v in a]
        axes[0].plot(x,y,label=r['summary']['run']);axes[1].plot(x,[v['raw_consumer_ms'] for v in a],label=r['summary']['run'])
    axes[0].axhline(0,color='#b24b41',ls='--');axes[0].set(ylabel='Remaining horizon, m',title='2 workers × 4 candidate threads — real-paced 10 Hz arrivals');axes[0].legend()
    axes[1].axhline(100,color='#b24b41',ls='--',label='100 ms input period');axes[1].set(ylabel='Raw service incl. arrival prep, ms',xlabel='Replay wall schedule, seconds');axes[1].legend()
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(dest/'live_horizon.png',dpi=145);plt.close(fig);print('CHARTS_COMPLETE')
if __name__=='__main__':main()
