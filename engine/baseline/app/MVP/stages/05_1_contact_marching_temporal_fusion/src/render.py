"""Scientific comparison figures and GIFs; subsampling is display-only."""
from fusion_common import *
sys.path.insert(0,str(ROOT/'results_rail2d/_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from fusion import CausalSource,assemble
from assess import reference_readonly
from tracker import TemplateTracker
from contact_rail_step2 import ContactRailDetector
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse

COLORS=['#cf572a','#297da3','#aa8d16','#8755a2','#21866f','#cf648c','#6d777b','#3f54b6','#795548']
def subset(ids,n=16000):return ids if len(ids)<=n else ids[np.linspace(0,len(ids)-1,n).astype(int)]
def age_scatter(ax,x,y,age,s=3):
    for a in np.unique(age):
        m=age==a;ax.scatter(x[m],y[m],s=s,c=COLORS[min(int(a),8)],alpha=.75,label='T' if a==0 else f'T−{int(a)}',rasterized=True)

def prepare(e):
    r=e['row'];run=r['run'];i=r['start_frame'];name=r['variant'];pred=load_prediction(OUT/'heldout'/name/key(run,i));base=load_prediction(OUT/'heldout/F0'/key(run,i));seed=seed_from_cache(run,i)
    c,_=assemble(CausalSource(run,i),pred['fusion']['config'],seed);ref,_=reference_readonly(run,i);return r,pred,base,c,ref

def slice_figure(r,pred,base,c,ref,step=None,overlap_only=False):
    st=step or (pred['steps'][-1] if pred['steps'] else {})
    if st.get('plane_origin') is None:
        st=next((s for s in reversed(pred['steps']) if s.get('plane_origin') is not None),None)
        if st is None:st=dict(step_index=0,status='Базис seed для диагностики: новый frame недоступен',failure_reason=pred['reason'],plane_origin=np.asarray(pred['seed']['anchor'])+np.asarray(pred['seed']['basis'])[:,0]*4,basis=pred['seed']['basis'])
    C=np.asarray(st['plane_origin']);B=np.asarray(st['basis']);q=(c['xyz']-C)@B;a=np.asarray(st.get('overlap_anchor',[0,0]));tt=TemplateTracker(ContactRailDetector().template,pred['seed']['side'],CFG)
    width=4 if overlap_only else 8
    lower=a+tt.lo-.12;upper=a+tt.hi+.12;crop=(q[:,0]>=0)&(q[:,0]<=width)&np.all((q[:,1:]>=lower)&(q[:,1:]<=upper),axis=1)
    gt=(ref['future']-C)@B;gm=(gt[:,0]>=0)&(gt[:,0]<=width)&np.all((gt[:,1:]>=lower)&(gt[:,1:]<=upper),axis=1)
    fig,axes=plt.subplots(2,2,figsize=(12,9),sharex=True,sharey=True)
    for col,current in enumerate((True,False)):
        ids=np.flatnonzero(crop&((c['age_frames']==0) if current else True));show=subset(ids)
        for line in (0,1):
            ax=axes[line,col];age_scatter(ax,q[show,1],q[show,2],c['age_frames'][show],3)
            ax.plot(tt.template[:,0]+a[0],tt.template[:,1]+a[1],'.',c='#111827',ms=1.6,label='Шаблон overlap')
            if line:
                shown_gt=subset(np.flatnonzero(gm),1500);ax.scatter(gt[shown_gt,1],gt[shown_gt,2],s=6,facecolors='none',edgecolors='#10a171',alpha=.24,label='Будущий GT')
            # Raw main variants preserve current T rows at the beginning.
            # The left panel must show actual baseline decisions, not the
            # current-source subset of the fused detector's decision.
            chosen=base['indices'] if current else pred['indices'][pred['point_step']<=st['step_index']]
            chosen=chosen[crop[chosen]]
            if len(chosen):ax.scatter(q[chosen,1],q[chosen,2],s=18,facecolors='none',edgecolors='#d61f38',lw=.6,label='Принятый support')
            ax.set(xlim=(lower[0],upper[0]),ylim=(lower[1],upper[1]),xlabel='v, м',ylabel='w, м',title=('Только T' if current else r['variant']+f" · {c['meta']['actual_frames']} кадр.")+(' · с GT' if line else ' · без GT')+f' · {len(ids)} точек ROI');ax.set_aspect('equal');ax.grid(alpha=.2)
    handles={}
    for ax in axes.flat:
        hh,ll=ax.get_legend_handles_labels();handles.update(zip(ll,hh))
    fig.legend(handles.values(),handles.keys(),loc='upper center',bbox_to_anchor=(.5,.925),ncol=6,fontsize=8)
    reach=r.get('continuous_reach_5cm');reach_label=f"≤5 см: T {reach-r.get('delta_reach',0):.2f} м → {reach:.2f} м" if r.get('evaluation_eligible') and reach is not None else 'Дальность не оценивается: недостаточный reference'
    window_label='Подтверждённый baseline overlap 0–4 м' if overlap_only else 'Одинаковый срез'
    fig.suptitle(f"{r['run']} / {frames(r['run'])[r['start_frame']]['file']}\n{window_label} · шаг {st['step_index']} · {st.get('status')} · {st.get('failure_reason','')}\n{reach_label}",fontsize=11)
    fig.tight_layout(rect=(0,0,1,.96));return fig

def draw3d(r,pred,base,c):
    B=np.asarray(pred['seed']['basis']);q=c['xyz']@B;pp=q[pred['indices']];current_ids=np.flatnonzero(c['age_frames']==0)
    # Current rows keep original ordering in raw main variants.
    base_xyz=np.load(BASE_OUT/'cache'/key(r['run'],r['start_frame'])/'xyz.npy');bp=base_xyz[base['indices']]@B
    union=np.r_[pp,bp];low=np.min(union,axis=0)-[1,.35,.30];high=np.max(union,axis=0)+[2,.35,.30];crop=np.all((q>=low)&(q<=high),axis=1)
    fig=plt.figure(figsize=(14,6))
    for col,(p,title) in enumerate(((bp,'T only'),(pp,r['variant']))):
        ax=fig.add_subplot(1,2,col+1,projection='3d');ids=subset(np.flatnonzero(crop&((c['age_frames']==0) if col==0 else True)),7000)
        ax.scatter(q[ids,0],q[ids,1],q[ids,2],s=.6,c='#8fa9b4',alpha=.1)
        if col==0:ax.scatter(p[:,0],p[:,1],p[:,2],s=2,c=COLORS[0])
        else:
            ages=c['age_frames'][pred['indices']]
            for age in np.unique(ages):
                m=ages==age;ax.scatter(p[m,0],p[m,1],p[m,2],s=2,c=COLORS[min(int(age),8)],label='T' if age==0 else f'T−{age}')
            ax.legend(loc='upper center',bbox_to_anchor=(.5,1.02),ncol=4,fontsize=8)
        ax.set(xlabel='u, м',ylabel='v, м',zlabel='w, м',title=title,xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]));ax.set_box_aspect((5,1.2,1));ax.view_init(elev=22,azim=-62)
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(3));ax.zaxis.set_major_locator(MaxNLocator(4));ax.tick_params(labelsize=9)
    fig.suptitle(f"{r['run']} · {r['start_frame']+1}\nРеальные точки; цвет — возраст кадра. Поперечный масштаб увеличен.",fontsize=11);fig.subplots_adjust(left=.03,right=.97,bottom=.07,top=.86,wspace=.08);return fig

def one(e,animation=False):
    r,pred,base,c,ref=prepare(e);stem=r['variant']+'__'+key(r['run'],r['start_frame']);folder=OUT/('animations' if animation else 'gallery')
    if animation:
        images=[]
        for st in pred['steps']:
            if st.get('plane_origin') is None:continue
            fig=slice_figure(r,pred,base,c,ref,st);buf=BytesIO();fig.savefig(buf,format='png',dpi=80);plt.close(fig);buf.seek(0);images.append(Image.open(buf).convert('RGB'))
        if not images:return stem,'no steps'
        images[0].save(folder/(stem+'.gif'),save_all=True,append_images=images[1:]+[images[-1]]*2,duration=800,loop=0);return stem,len(images)
    fig=slice_figure(r,pred,base,c,ref);fig.savefig(folder/(stem+'_slices.png'),dpi=135);plt.close(fig)
    fig=draw3d(r,pred,base,c);fig.savefig(folder/(stem+'_3d.png'),dpi=135);plt.close(fig)
    if 'blur' in e['tags']:
        known=next((s for s in reversed(base['steps']) if s['status']=='ACCEPTED' and s.get('overlap_anchor') is not None),None)
        if known:
            fig=slice_figure(r,pred,base,c,ref,known,overlap_only=True);fig.savefig(folder/(stem+'_blur_overlap.png'),dpi=135);plt.close(fig)
    if r.get('raw_wrong_structure_trigger') or r.get('wrong_structure_suspected'):
        ev=load(OUT/'heldout'/r['variant']/key(r['run'],r['start_frame'])/'evaluation.json')
        bad=[s for s in ev['steps'] if s['status']=='ACCEPTED' and (s.get('point_p95') or 0)>.20]
        if bad:
            worst=max(bad,key=lambda s:s['point_p95']);st=next(s for s in pred['steps'] if s['step_index']==worst['step_index'])
            fig=slice_figure(r,pred,base,c,ref,st);fig.savefig(folder/(stem+'_suspect_slices.png'),dpi=135);plt.close(fig)
    return stem,'rendered'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--animations',action='store_true');ap.add_argument('--workers',type=int,default=2);a=ap.parse_args();examples=load(OUT/('animations' if a.animations else 'gallery')/'examples.json')
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for j,f in enumerate(as_completed([pool.submit(one,e,a.animations) for e in examples])):print('RENDER',j+1,len(examples),f.result(),flush=True)
if __name__=='__main__':main()
