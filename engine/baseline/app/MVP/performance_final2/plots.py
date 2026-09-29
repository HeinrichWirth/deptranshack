from . import ROOT,OUT
import sys
sys.path.insert(0,str(ROOT/'results_rail2d/_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import long_common as lc

def main():
    dest=OUT/'figures';dest.mkdir(exist_ok=True)
    rows=lc.csv_read(OUT/'waterfall.csv');paired=lc.csv_read(OUT/'paired_variants.csv')
    fig,ax=plt.subplots(2,1,figsize=(13,10),layout='constrained')
    vals=[float(r['median_ms']) for r in rows];last=0
    for j,(r,v) in enumerate(zip(rows,vals)):
        if j==0 or j==len(rows)-1:ax[0].bar(j,v,color='#3775a8')
        else:ax[0].bar(j,abs(v-last),bottom=min(v,last),color='#b35057' if v>last else '#36a08a')
        ax[0].text(j,max(v,last)+25,f'{v:.0f}',ha='center',fontsize=10);last=v
    ax[0].set_xticks(range(len(rows)),[r['optimization'] for r in rows],rotation=20)
    ax[0].set(ylabel='Full median ms',title='Chronological waterfall: observed values, timing drift included (not causal gains)')
    x=np.arange(len(paired));ax[1].bar(x-.18,[float(r['median_ms']) for r in paired],.36,label='Full solve median',color='#3775a8')
    ax[1].bar(x+.18,[float(r['C4_median_ms']) for r in paired],.36,label='C4 median',color='#36a08a')
    ax[1].set_xticks(x,[r['backend'] for r in paired],rotation=15);ax[1].set(ylabel='ms',title='Counterbalanced 32 starts: final selection evidence')
    ax[1].axhline(100,ls=':',color='#b35057',label='100 ms geometry target');ax[1].legend()
    fig.savefig(dest/'waterfall.png',dpi=150);fig.savefig(dest/'waterfall.pdf');plt.close(fig)
    paths=sorted((OUT/'async').glob('*/*/result.json'));runs=list(dict.fromkeys(p.parent.name for p in paths))
    fig,axes=plt.subplots(len(runs),2,figsize=(15,3.5*len(runs)),layout='constrained',squeeze=False)
    for j,run in enumerate(runs):
        for path in paths:
            if path.parent.name!=run:continue
            b=lc.load(path);r=b['frames'];mode=r[0]['backend'];t=[x['source_seconds'] for x in r]
            axes[j,0].plot(t,[x.get('geometry_age_seconds',np.nan) for x in r],label=mode,lw=1)
            axes[j,1].plot(t,[x.get('remaining_horizon',np.nan) for x in r],label=mode,lw=1)
        axes[j,0].set(title=run,ylabel='Geometry source age, s',xlabel='Source time, s');axes[j,0].legend()
        axes[j,1].set(title='Remaining horizon in current frame',ylabel='m',xlabel='Source time, s');axes[j,1].axhline(0,color='red',ls=':');axes[j,1].legend()
    fig.savefig(dest/'async_age.png',dpi=150);fig.savefig(dest/'async_age.pdf');plt.close(fig)

if __name__=='__main__':main()
