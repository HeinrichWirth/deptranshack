"""Comparable scientific slices, whole-track views and spatial animations."""
from long_common import *
sys.path.insert(0,str(ROOT/'results_rail2d/_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image
from io import BytesIO
from artifact_data import case_data
from contact_rail_step2 import ContactRailDetector
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse

def subset(ids,n=10000):return ids if len(ids)<=n else ids[np.linspace(0,len(ids)-1,n,dtype=int)]

def baseline_points(e):
    group=e.get('group','phase_e_benchmark')
    folder=OUT/group/'B0'/key(e['run'],e['start_frame'])
    if not folder.exists():folder=OUT/'phase_b_full_dev/B0'/key(e['run'],e['start_frame'])
    with np.load(folder/'provenance.npz') as z:return z['xyz']

def section(e,p,c,ref,base,st,gt=True):
    C=np.asarray(st.get('plane_origin',p['seed']['anchor']));B=np.asarray(st.get('basis',p['seed']['basis']));W=float(st.get('window',p['config'].get('window',8)))
    a=np.asarray(st.get('overlap_anchor',[0,0]));tp=ContactRailDetector().template*np.array([p['seed']['side'],1]);lo=a+tp.min(axis=0)-.17;hi=a+tp.max(axis=0)+.17
    q=(c['xyz']-C)@B;g=(ref['future']-C)@B;b=(base-C)@B
    visible_gt=(g[:,0]>=0)&(g[:,0]<=W)&(np.linalg.norm(g[:,1:]-a,axis=1)<=1.)
    if visible_gt.any():
        lo=np.minimum(lo,np.quantile(g[visible_gt,1:],.05,axis=0)-.05);hi=np.maximum(hi,np.quantile(g[visible_gt,1:],.95,axis=0)+.05)
    crop=(q[:,0]>=0)&(q[:,0]<=W)&np.all((q[:,1:]>=lo)&(q[:,1:]<=hi),axis=1);gm=(g[:,0]>=0)&(g[:,0]<=W)&np.all((g[:,1:]>=lo)&(g[:,1:]<=hi),axis=1)
    bm=(b[:,0]>=0)&(b[:,0]<=W)&np.all((b[:,1:]>=lo)&(b[:,1:]<=hi),axis=1)
    fig,axs=plt.subplots(2,2,figsize=(12,8.8),sharex=True,sharey=True)
    selected=p['indices'];upto=p['point_step']<=st.get('step_index',999);display_state=p['point_state'].copy()
    for earlier in p['steps']:
        if earlier.get('promoted_at',-1)>st.get('step_index',999):display_state[p['point_step']==earlier['step_index']]=1
    for col in (0,1):
        ids=np.flatnonzero(crop&((c['age_frames']==0) if col==0 else True));show=subset(ids)
        for row in (0,1):
            ax=axs[row,col];ax.scatter(q[show,1],q[show,2],c=c['age_frames'][show],cmap='viridis',vmin=0,vmax=max(1,c['age_frames'].max()),s=3,alpha=.65)
            ax.plot(tp[:,0]+a[0],tp[:,1]+a[1],'.',color='#202a36',ms=1,label='Шаблон old overlap')
            if row:
                gs=subset(np.flatnonzero(gm),1600);ax.scatter(g[gs,1],g[gs,2],s=10,facecolors='none',edgecolors='#009e73',alpha=.35,label='Будущий GT (оценка)')
            if col==0:ax.scatter(b[bm,1],b[bm,2],s=17,facecolors='none',edgecolors='#d1495b',lw=.65,label='B0: подтверждено')
            else:
                for state,color,label in [(2,'#d1495b','Подтверждено'),(1,'#f29f05','Временно не подтверждено')]:
                    ix=selected[(display_state==state)&upto&crop[selected]]
                    ax.scatter(q[ix,1],q[ix,2],s=19,facecolors='none',edgecolors=color,lw=.8,label=label)
            ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),xlabel='v, м',ylabel='w, м',title=('Текущий T · B0' if col==0 else e['variant']+' · выбранная история')+(' · GT' if row else ' · без GT')+f' · {len(ids)} точек')
            if row and not gm.any():ax.text(.03,.97,'GT вне показанной области\nили отсутствует',transform=ax.transAxes,va='top',fontsize=8,color='#a54d30')
            ax.set_aspect('equal');ax.grid(alpha=.2)
    handles={}
    for ax in axs.flat:
        hh,ll=ax.get_legend_handles_labels();handles.update(zip(ll,hh))
    fig.legend(handles.values(),handles.keys(),loc='upper center',bbox_to_anchor=(.5,.918),ncol=4,fontsize=8)
    reach=e.get('continuous_reach_5cm');reachtext=f'{reach:.2f} м' if reach is not None else 'не оценивается'
    fig.suptitle(f"{e['run']} / {frames(e['run'])[e['start_frame']]['file']}\n{e['variant']} · шаг {st.get('step_index',0)} · {st.get('state',st.get('status',''))} · {st.get('failure_reason',p['reason'])}\nНепрерывная дальность ≤5 см: {reachtext} · окно {W:g} м · цвет: возраст источника T…T−{int(c['age_frames'].max())}",fontsize=10.5)
    fig.tight_layout(rect=(0,0,1,.88));return fig

def overview(e,p,c,ref,base):
    B=np.asarray(p['seed']['basis']);q=c['xyz']@B;pred=q[p['indices']];b=base@B;g=ref['future']@B
    fig,axes=plt.subplots(2,1,figsize=(14,8),sharex=True)
    extent=max(30.,min(132.,(e.get('max_confirmed_observed_range') or 0)+15))
    for ax,axis,title in [(axes[0],1,'Вид сверху в базисе seed'),(axes[1],2,'Вид сбоку в базисе seed')]:
        allp=np.r_[pred,b];low=allp[:,axis].min()-.4;high=allp[:,axis].max()+.4
        m=(q[:,0]>=0)&(q[:,0]<=extent)&(q[:,axis]>=low)&(q[:,axis]<=high);ids=subset(np.flatnonzero(m),18000)
        ax.scatter(q[ids,0],q[ids,axis],s=.4,c='#adb6bd',alpha=.4,label='Реальные входные точки')
        gm=(g[:,0]>=0)&(g[:,0]<=extent)&(g[:,axis]>=low)&(g[:,axis]<=high);gi=subset(np.flatnonzero(gm),12000)
        ax.scatter(g[gi,0],g[gi,axis],s=1,c='#129370',alpha=.45,label='GT для оценки')
        ax.scatter(b[:,0],b[:,axis],s=3,c='#2864b4',alpha=.7,label='B0')
        for st,color,label in [(2,'#dc493d','Подтверждено'),(1,'#e99f00','Tentative')]:
            ix=p['point_state']==st;ax.scatter(pred[ix,0],pred[ix,axis],s=4,c=color,label=label)
        for s in p['steps']:
            if s.get('state')=='GAP':
                u=np.asarray(s['plane_origin'])@B[:,0];ax.axvspan(u+4,u+8,color='#e99f00',alpha=.18)
        ax.set(xlim=(0,extent),ylim=(low,high),ylabel=('v' if axis==1 else 'w')+', м',title=title+' · поперечный масштаб увеличен');ax.grid(alpha=.2)
    axes[0].legend(ncol=5,fontsize=8);axes[1].set_xlabel('Продольная координата u, м; это не отдельные синтетические точки')
    fig.suptitle(f"{e['run']} / {frames(e['run'])[e['start_frame']]['file']} · {e['variant']}\nПодтверждённые наблюдения {e.get('max_confirmed_observed_range',0):.2f} м · поиск {e.get('max_search_hypothesis_range') or 0:.2f} м",fontsize=11)
    fig.tight_layout(rect=(0,0,1,.94));return fig

def long_sections(e,p,c,ref):
    B=np.asarray(p['seed']['basis']);pred=c['xyz'][p['indices']];rr=np.linalg.norm(pred,axis=1);fig,axs=plt.subplots(3,4,figsize=(16,11))
    for ax,h in zip(axs.flat,range(10,121,10)):
        available=[s for s in p['steps'] if s.get('anchor_3d') is not None]
        st=min(available,key=lambda s:abs(np.linalg.norm(s['anchor_3d'])-h)) if available else dict(plane_origin=p['seed']['anchor'],basis=B)
        C=np.asarray(st['plane_origin']);BB=np.asarray(st['basis']);q=(c['xyz']-C)@BB;g=(ref['future']-C)@BB
        radii=np.linalg.norm(c['xyz'],axis=1);gr=(abs(np.linalg.norm(ref['future'],axis=1)-h)<=2)&np.all(abs(g[:,1:])<1.2,axis=1)
        low=np.array([-.3,-.3]);high=np.array([.3,.3])
        if gr.any():low=np.minimum(low,np.quantile(g[gr,1:],.02,axis=0)-.03);high=np.maximum(high,np.quantile(g[gr,1:],.98,axis=0)+.03)
        m=(abs(radii-h)<=2)&np.all((q[:,1:]>=low)&(q[:,1:]<=high),axis=1);ids=subset(np.flatnonzero(m),5000)
        ax.scatter(q[ids,1],q[ids,2],s=3,c=c['age_frames'][ids],cmap='viridis',vmin=0,vmax=max(1,c['age_frames'].max()))
        gm=gr&np.all((g[:,1:]>=low)&(g[:,1:]<=high),axis=1);gi=subset(np.flatnonzero(gm),1000)
        ax.scatter(g[gi,1],g[gi,2],s=8,facecolors='none',edgecolors='#159776',alpha=.25)
        for state,color in [(2,'#d64245'),(1,'#dd9900')]:
            ids=p['indices'][(abs(rr-h)<=2)&(p['point_state']==state)];ax.scatter(q[ids,1],q[ids,2],s=18,facecolors='none',edgecolors=color)
        ax.set(title=f'{h}±2 м · {int(m.sum())} входных точек',xlabel='v, м',ylabel='w, м',xlim=(low[0],high[0]),ylim=(low[1],high[1]));ax.set_aspect('equal');ax.grid(alpha=.2)
    fig.suptitle(e['variant']+' · '+e['run']+' / '+frames(e['run'])[e['start_frame']]['file']+'\nСрезы по реальной дальности; пустые интервалы остаются пустыми',fontsize=11)
    fig.tight_layout(rect=(0,.045,.94,.94))
    cb=fig.colorbar(ScalarMappable(norm=Normalize(0,max(1,c['age_frames'].max())),cmap='viridis'),cax=fig.add_axes([.95,.22,.012,.55]))
    cb.set_label('Возраст источника, кадров; 0 = T')
    fig.legend(handles=[Line2D([],[],marker='o',linestyle='none',markerfacecolor='none',markeredgecolor=color,label=label) for color,label in [('#159776','GT: только для оценки'),('#d64245','Confirmed: реальные наблюдения'),('#dd9900','Tentative: реальные наблюдения')]],loc='lower center',ncol=3,frameon=False,fontsize=10)
    return fig

def one(task):
    e,animate=task;folder=OUT/'gallery'/e['id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'COMPLETE.json').exists():return e['id'],'cached'
    p,c,ref,info=case_data(e);base=baseline_points(e)
    st=next((s for s in reversed(p['steps']) if s.get('plane_origin') is not None),dict(plane_origin=p['seed']['anchor'],basis=p['seed']['basis']))
    fig=section(e,p,c,ref,base,st);fig.savefig(folder/'terminal.png',dpi=125);plt.close(fig)
    fig=overview(e,p,c,ref,base);fig.savefig(folder/'track.png',dpi=125);plt.close(fig)
    files=['terminal.png','track.png']
    if any(t in e['tags'] for t in ('100m_failure','negative_100m_detail','catastrophic_review')):
        from collect_long import first_failure_step
        first=first_failure_step(p,OUT/e['folder'])
        if first and first.get('plane_origin') is not None:
            fig=section(e,p,c,ref,base,first);fig.savefig(folder/'first_failure.png',dpi=125);plt.close(fig);files.append('first_failure.png')
    if (e.get('max_confirmed_observed_range') or 0)>=100:
        fig=long_sections(e,p,c,ref);fig.savefig(folder/'sections_10m.png',dpi=125);plt.close(fig);files.append('sections_10m.png')
        reference_path=BASE/'future_gt'/(key(e['run'],e['start_frame'])+'_reference_v2.npz')
        save(OUT/'100m_cases'/(e['id']+'.json'),dict(example=e,source_frames=p['input']['source_frames'],states=p['steps'],hypotheses=p.get('hypotheses'),curve=p.get('curve'),reason=p['reason'],provenance=str(OUT/e['folder']/'provenance.npz'),input_T=str(dataset()/e['run']/frames(e['run'])[e['start_frame']]['file']),reference=info,reference_npz=str(reference_path),reference_sha256=sha(reference_path)))
    gif=None
    if animate:
        eligible=[s for s in p['steps'] if s.get('plane_origin') is not None];chosen=[eligible[j] for j in np.unique(np.linspace(0,len(eligible)-1,min(14,len(eligible)),dtype=int))] if eligible else []
        images=[]
        for st in chosen:
            fig=section(e,p,c,ref,base,st);buf=BytesIO();fig.savefig(buf,format='png',dpi=75);plt.close(fig);buf.seek(0);images.append(Image.open(buf).convert('RGB'))
        if len(images)>=2:
            gif=OUT/'animations'/(e['id']+'.gif');images[0].save(gif,save_all=True,append_images=images[1:],duration=800,loop=0);gif=gif.relative_to(OUT).as_posix()
    save(folder/'COMPLETE.json',dict(files=files,animation=gif,display_sampling_only=True));return e['id'],len(files)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--refresh-long-sections',action='store_true');a=ap.parse_args();items=load(OUT/'gallery/examples.json')
    if a.refresh_long_sections:
        for e in items:
            if (e.get('max_confirmed_observed_range') or 0)<100:continue
            p,c,ref,info=case_data(e);fig=long_sections(e,p,c,ref)
            fig.savefig(OUT/'gallery'/e['id']/'sections_10m.png',dpi=125);plt.close(fig)
            print('REFRESH SECTIONS',e['id'],flush=True)
        return
    featured=list(dict.fromkeys(e['id'] for tag in ('best_reach','best_gains','median','worst_losses') for e in [r for r in items if tag in r['tags']][:2]))
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        ff=[pool.submit(one,(e,e['id'] in featured)) for e in items]
        for j,f in enumerate(as_completed(ff),1):
            value=f.result()
            if j%20==0 or j==len(ff):print('RENDER',j,len(ff),value,flush=True)

if __name__=='__main__':main()
