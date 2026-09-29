"""Actual distant returns in evaluation-only canonical GT coordinates."""
from long_common import *
sys.path.insert(0,str(ROOT/'results_rail2d/_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from contact_rail_step2 import ContactRailDetector

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--force',action='store_true');args=ap.parse_args()
    candidates=[]
    for folder in (OUT/'observability').iterdir():
        if not (folder/'ranges.json').exists():continue
        row=load(folder/'ranges.json')[-1]
        if row['max_observable_range_5cm']>=98:
            budgets=load(folder/'budgets.json');n=sum(b['unique_xyz'] for b in budgets if b['history']==16 and b['center_m']>=90)
            candidates.append(dict(row,far_positions=n,folder=folder.name))
    chosen=sorted(candidates,key=lambda r:-r['far_positions'])[:20];index=[]
    for row in chosen:
        path=OUT/'observability'/row['folder'];target=path/'profile_80_120m.png'
        if args.force or not target.exists():
            with np.load(path/'near_gt_points.npz') as z:c={k:z[k] for k in z.files}
            s=seed_from_cache(row['run'],row['start_frame']);tp=ContactRailDetector().template*np.array([s['side'],1])
            visible=(c['distance']<=.05)&np.any(abs(c['range'][:,None]-np.array([15,60,80,100,110,120]))<=2,axis=1)
            bounds=np.r_[tp,c['local_vw'][visible]];low=bounds.min(axis=0)-.025;high=bounds.max(axis=0)+.025
            fig,axs=plt.subplots(2,3,figsize=(13,8.8))
            for ax,h in zip(axs.flat,(15,60,80,100,110,120)):
                m=(abs(c['range']-h)<=2)&(c['distance']<=.05);labelled=m&c['class2'];unlabelled=m&~c['class2'];q=c['local_vw']
                ax.plot(tp[:,0],tp[:,1],'.',ms=1,c='#1f2933',label='Канонический профиль')
                ax.scatter(q[unlabelled,0],q[unlabelled,1],s=12,c='#afbbc4',alpha=.5,label='Близко к GT, без class2')
                ax.scatter(q[labelled,0],q[labelled,1],s=12,c=c['age_frames'][labelled],cmap='viridis',vmin=0,vmax=15,label='Исходный class2; цвет=возраст')
                ax.set(title=f'{h}±2 м · {int(m.sum())} возвратов\n{len(np.unique(c["xyz"][m],axis=0))} XYZ · {len(np.unique(c["source_frame"][m]))} источн. · class2 {int(labelled.sum())}',xlabel='v относительно GT, м',ylabel='w относительно GT, м',xlim=(low[0],high[0]),ylim=(low[1],high[1]))
                ax.set_aspect('equal');ax.grid(alpha=.2)
            handles,labels=axs[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.945),ncol=3,fontsize=8)
            fig.suptitle(row['run']+' / '+frames(row['run'])[row['start_frame']]['file']+' · T…T−15\nРеальные измерения; GT используется только здесь, после inference. Близость к GT ≠ доказанная принадлежность CR.',fontsize=10)
            fig.tight_layout(rect=(0,0,1,.90));fig.savefig(target,dpi=135);plt.close(fig)
        index.append(dict(row,image=target.relative_to(OUT).as_posix()))
    save(OUT/'observability/profile_examples.json',index);print('FAR PROFILES',len(index))

if __name__=='__main__':main()
