"""Static scientific plots and local HTML gallery, after inference/evaluation."""
from . import ROOT
import sys
sys.path.insert(0,str(ROOT/'results_rail2d/_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import long_common as lc
import html
from .frame_data import FrameData

OUT=ROOT/'results_final_pipeline'
COLORS={1:'#168455',2:'#2265c6',3:'#e68a16'}


def npz(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def case(folder,category):
    meta=lc.load(folder/'summary.json');run,i=meta['run'],meta['frame'];records=lc.frames(run)
    c4=lc.load(folder/'c4.json');B=np.array(c4['seed']['basis']);pose=np.asarray(records[i]['lidar_pose_in_folder'])
    p=npz(folder/'prediction.npz');curve=npz(folder/'curve.npz');prov=npz(folder/'point_provenance.npz')['rows']
    points=np.empty((len(prov),3));current=None
    for frame in np.unique(prov['source_frame']):
        data=FrameData.read(lc.dataset()/run/records[int(frame)]['file'],int(frame))
        take=prov['source_frame']==frame
        points[take]=((data.world[prov['source_row'][take]]-pose[:3,3])@pose[:3,:3])@B
        if frame==i:current=(data.world-pose[:3,3])@pose[:3,:3]@B
    ev=npz(folder/'evaluation.npz') if (folder/'evaluation.npz').exists() else None
    if ev is None and (folder/'evaluation_unpaired.npz').exists():ev=npz(folder/'evaluation_unpaired.npz')
    C=curve['C']@B;rails=np.einsum('nri,ij->nrj',p['pair'],B);center=p['center']@B
    fig=plt.figure(figsize=(18,14),layout='constrained')
    axes=[fig.add_subplot(4,2,k+1,projection='3d' if k==3 else None) for k in range(8)]
    for ax in axes:
        if ax.name!='3d':ax.grid(alpha=.2)
    ax=axes[0]
    if current is not None:
        subset=current[(current[:,0]>=0)&(current[:,0]<=max(10,C[:,0].max()))&(abs(current[:,1])<3)&(current[:,2]>-2)&(current[:,2]<0)]
        if len(subset)>15000:subset=subset[np.linspace(0,len(subset)-1,15000,dtype=int)]
        ax.scatter(subset[:,0],subset[:,1],s=.25,c='#aeb6c1',alpha=.25,rasterized=True)
    for stage,color in COLORS.items():
        take=prov['first_found_stage']==stage;ax.scatter(points[take,0],points[take,1],s=2,c=color,label=f'First STEP {stage}: {take.sum()}',rasterized=True)
    ax.set(title='A · First-found stage, реальные точки',xlabel='u, м',ylabel='v, м');ax.legend(fontsize=8)
    ax=axes[1]
    for stage in (2,3):
        take=prov['first_found_stage']==stage;ax.scatter(points[take,0],points[take,1],s=3,c=COLORS[stage],label='STEP2 bootstrap' if stage==2 else 'C4 marching')
    ax.plot(C[:,0],C[:,1],color='#14243c',lw=1.2,label='C4_SMOOTH');ax.legend(fontsize=8)
    ax.set(title='B · CR bootstrap → marching → reference',xlabel='u, м',ylabel='v, м')
    ax=axes[2]
    for j,label in enumerate(('near','far')):ax.plot(rails[:,j,0],rails[:,j,1],label='RAW '+label)
    ax.plot(center[:,0],center[:,1],':',color='#633cab',label='center')
    if ev is not None:
        gt=np.einsum('nri,ij->nrj',ev['gt'],B)
        for j in (0,1):ax.plot(gt[:,j,0],gt[:,j,1],'--',color='#999999',lw=.9,label='approximate GT' if j==0 else None)
    ax.set(title='C · Восстановленные ходовые рельсы: план',xlabel='u, м',ylabel='v, м');ax.legend(fontsize=8)
    ax=axes[3]
    ax.plot(C[:,0],C[:,1],C[:,2],color='#14243c',label='CR reference')
    for j in (0,1):ax.plot(rails[:,j,0],rails[:,j,1],rails[:,j,2],label=('near','far')[j])
    ax.plot(center[:,0],center[:,1],center[:,2],':',color='#633cab');ax.set_box_aspect((3,1,.7));ax.view_init(22,-67)
    ax.set(title='D · 3D, local T coordinates',xlabel='u, м',ylabel='v, м',zlabel='w, м');ax.legend(fontsize=8)
    ax=axes[4]
    ax.plot(C[:,0],C[:,2],label='CR')
    for j in (0,1):ax.plot(rails[:,j,0],rails[:,j,2],label=('near','far')[j])
    ax.set(title='E · Высота: без продолжения за C4',xlabel='u, м',ylabel='w, м');ax.legend(fontsize=8)
    ax=axes[5]
    if ev is not None:
        mask=ev['valid'];ax.plot(ev['range'][mask],ev['raw_error'][mask,1]*100,label='RAW far')
        if 'reference_error' in ev:ax.plot(ev['range'][mask],ev['reference_error'][mask,1]*100,'--',label='old GT-seed far')
        ax.axhline(20,color='#b83232',ls=':',label='20 cm');ax.legend(fontsize=8)
    else:ax.text(.5,.5,'Независимый расчётный эталон\nдля этого старта отсутствует',ha='center',va='center',transform=ax.transAxes)
    ax.set(title='F · Ошибка к приближённому GT',xlabel='дальность, м',ylabel='ошибка, см')
    ax=axes[6]
    world=curve['C']@pose[:3,:3].T;s=curve['s'];d=np.gradient(world,s,axis=0);heading=np.unwrap(np.arctan2(d[:,1],d[:,0]));kappa=np.gradient(heading,s)
    ax.plot(s,kappa*1000,color='#14243c',label='curvature × 1000');right=ax.twinx();right.plot(s,np.rad2deg(heading-heading[0]),color='#c56a1c',alpha=.7,label='heading change')
    ax.set(title='G · Кривизна и изменение курса',xlabel='станция, м',ylabel='κ × 1000, 1/м');right.set_ylabel('Δ heading, °',color='#c56a1c')
    ax=axes[7]
    near=points[(prov['first_found_stage']==1)]
    ax.scatter(near[:,1],near[:,2],s=5,c=COLORS[1],alpha=.5,label='STEP1 surface support')
    seed=lc.load(folder/'seed.json');heads=np.array([r['heads'] for r in seed['observations']])@B
    ax.scatter(heads[:,:,1].ravel(),heads[:,:,2].ravel(),s=25,c='#e14c32',marker='x',label='1m robust 3D heads')
    ax.set(title=f'H · Seed без GT: {seed["sections"]} метровых сечений',xlabel='v, м',ylabel='w, м');ax.legend(fontsize=8)
    error=''
    if ev is not None and ev['valid'].any():error=f' · far p95 {np.percentile(ev["raw_error"][ev["valid"],1],95)*100:.2f} см'
    fig.suptitle(f'{category} · {run} / index {i:06d}\nRAW → final · C4 horizon {p["s"][-1]:.1f} м{error} · core {meta["timing"]["T_TOTAL"]*1000:.0f} мс',fontsize=15)
    target=OUT/'visualizations'/(folder.name+'.png');target.parent.mkdir(parents=True,exist_ok=True);fig.savefig(target,dpi=140);plt.close(fig)
    return dict(run=run,frame=i,category=category,image=target.relative_to(OUT).as_posix(),folder=folder.relative_to(OUT).as_posix())


def main():
    selections=[]
    for phase in ('deployable_development_cache','deployable_benchmark_cache'):
        path=OUT/phase/'per_start_quality.json'
        if not path.exists():continue
        rows=[r for r in lc.load(path) if r.get('paired') and r.get('DEPLOYABLE_PIPELINE_far_p95') is not None]
        rows.sort(key=lambda r:r['DEPLOYABLE_PIPELINE_far_p95']);N=len(rows)
        choices=[('ЛУЧШИЕ',r) for r in rows[:2]]+[('СРЕДНИЕ',r) for r in rows[max(0,N//2-1):N//2+1]]+[('ХУДШИЕ',r) for r in rows[-4:]]
        deltas=sorted(rows,key=lambda r:r['DEPLOYABLE_PIPELINE_far_p95']-r['RESEARCH_REFERENCE_far_p95'],reverse=True)
        choices+=[('SEED: НАИБОЛЬШАЯ РЕГРЕССИЯ',r) for r in deltas[:2]]
        for category,r in choices:
            folder=OUT/phase/lc.key(r['run'],r['frame'])
            if any(x['folder']==folder.relative_to(OUT).as_posix() for x in selections):continue
            selections.append(case(folder,category))
            print('PLOT',phase,r['run'],r['frame'],category,flush=True)
    phase='deployable_curved_run_cache';path=OUT/phase/'per_start_quality.json'
    if path.exists():
        rr=lc.load(path);rr=[r for r in rr if r.get('DEPLOYABLE_PIPELINE_far_p95') is not None or r.get('unpaired_far_p95') is not None]
        rr.sort(key=lambda r:r.get('DEPLOYABLE_PIPELINE_far_p95') if r.get('DEPLOYABLE_PIPELINE_far_p95') is not None else r['unpaired_far_p95'])
        for label,j in [('CURVED RUN · ЛУЧШИЙ',0),('CURVED RUN · СРЕДНИЙ',len(rr)//2),('CURVED RUN · ХУДШИЙ',len(rr)-1)]:
            if rr:
                r=rr[j];selections.append(case(OUT/phase/lc.key(r['run'],r['frame']),label))
    cards=[]
    for r in selections:
        label=f'{r["category"]} · {r["run"]} / {r["frame"]}'
        cards.append(f'<article id="{r["run"]}__{r["frame"]:06d}"><h2>{html.escape(label)}</h2><a href="{r["image"]}"><img loading="lazy" src="{r["image"]}" alt="{html.escape(label)}"></a><p><a href="{r["folder"]}/summary.json">Метаданные</a> · <a href="{r["folder"]}/seed.json">RAW seed</a></p></article>')
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>FINAL RAW PIPELINE — examples</title><style>body{font:16px system-ui;background:#edf1f6;color:#14243c;margin:25px}main{max-width:1450px;margin:auto}article{background:white;padding:20px;margin:24px 0;border-radius:12px}img{width:100%;height:auto}a{color:#225eba}h2{font-size:20px}</style><main><h1>FINAL RAW PIPELINE — лучшие, средние, худшие</h1><p>Сравниваются DEPLOYABLE_PIPELINE без GT и RESEARCH_REFERENCE со старым GT-seed. Разметка и регистрация приблизительны. Графики показывают сохранённые предсказания; GT подключён только после inference. Дальность ограничена C4.</p><p><a href="REPORT_FINAL_PIPELINE.html">Полный отчёт</a></p>'''+''.join(cards)+'</main></html>'
    (OUT/'gallery.html').write_text(page,encoding='utf-8');lc.save(OUT/'visualizations/cases.json',selections)


if __name__=='__main__':main()
