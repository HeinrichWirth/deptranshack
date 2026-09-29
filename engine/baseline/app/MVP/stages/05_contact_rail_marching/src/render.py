"""Static scientific plots, terminal slices and 3D propagation animations."""
from common import *
sys.path.insert(0,str(ROOT/'results_rail2d/_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from evaluation import load_prediction,reference
from contact_rail_step2 import ContactRailDetector
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,io

plt.rcParams.update({'font.size':10,'axes.titlesize':11,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
COL=dict(background='#abb5bf',overlap='#176aaf',new='#dd6829',future='#12977c',current='#a33aa4',template='#182c45')

def sample(p,n=12000):return p[::max(1,int(np.ceil(len(p)/n)))]

def slice_panel(ax,xyz,pred,step,ref,gt=False):
    C=np.array(step.get('plane_origin',pred['seed']['anchor']));B=np.array(step.get('basis',pred['seed']['basis']));uvw=(xyz-C)@B
    W=pred['config']['window'];a=np.array([step.get('anchor_v',0.),step.get('anchor_w',0.)]);oa=np.array(step.get('overlap_anchor',a))
    slab=(uvw[:,0]>=0)&(uvw[:,0]<=W);roi=slab&(abs(uvw[:,1]-oa[0])<.30)&(abs(uvw[:,2]-oa[1])<.28)
    q=sample(uvw[roi]);ax.scatter(q[:,1],q[:,2],s=3,c=COL['background'],alpha=.42,linewidths=0,label='Текущее облако T')
    known=np.array(step.get('overlap_indices',[]),dtype=int)
    if len(known):ax.scatter(uvw[known,1],uvw[known,2],s=9,c=COL['overlap'],alpha=.6,linewidths=0,label='Подтверждённое перекрытие')
    new=np.array(step.get('support_indices',[]),dtype=int)
    if len(new):ax.scatter(uvw[new,1],uvw[new,2],s=9,c=COL['new'],alpha=.8,linewidths=0,label='Новые принятые точки')
    else:
        unknown=roi&(uvw[:,0]>=W/2);q=sample(uvw[unknown]);ax.scatter(q[:,1],q[:,2],s=6,c=COL['new'],alpha=.55,linewidths=0,label='Новая половина: кандидаты')
    if gt:
        for name,label,color in (('future','Будущая разметка',COL['future']),('current','Разметка T',COL['current'])):
            pp=(ref[name]-C)@B;m=(pp[:,0]>=0)&(pp[:,0]<=W)&(abs(pp[:,1]-oa[0])<.30)&(abs(pp[:,2]-oa[1])<.28)
            pp=sample(pp[m],6000);ax.scatter(pp[:,1],pp[:,2],s=4,c=color,alpha=.4,linewidths=0,label=label)
    tp=ContactRailDetector().template*np.array([pred['seed']['side'],1])+a
    ax.scatter(tp[:,0],tp[:,1],s=1,c=COL['template'],alpha=.5,label='Шаблон кандидата')
    ax.scatter(*a,c='#c12743',s=90,marker='+',linewidths=2,label='Предложенная опора')
    ax.scatter(*oa,c=COL['overlap'],s=55,marker='x',linewidths=1.8,label='Опора перекрытия')
    comp=step.get('competing_candidate')
    if comp:ax.scatter(*comp['anchor'],edgecolors='#813ac2',s=100,marker='D',facecolors='none',label='Конкурент')
    ax.set(xlim=(oa[0]-.30,oa[0]+.30),ylim=(oa[1]-.28,oa[1]+.28),xlabel='v, м',ylabel='w, м',aspect='equal',title='Только вход детектора' if not gt else 'Разметка наложена после поиска')
    ax.grid(alpha=.18);ax.legend(fontsize=7,loc='upper left',framealpha=.85)

def slice_image(run,i,pred,xyz,ref,step,gt,path):
    fig,ax=plt.subplots(figsize=(8.8,7.3));slice_panel(ax,xyz,pred,step,ref,gt)
    conf=step.get('confidence');reason=step.get('failure_reason') or step.get('status') or pred['reason'];frame=frames(run)[i]['file']
    fig.suptitle(f'{run} / {frame}\nШаг {step["step_index"]}: {reason}',fontsize=11,y=.985)
    note=f'Overlap: {step.get("confirmed_overlap_n",0)} · кандидаты: {step.get("candidate_n",0)} · новые: {step.get("new_support_n",0)}'
    note+=f'\nΔyaw {step.get("delta_yaw",0):.3f}° · Δpitch {step.get("delta_pitch",0):.3f}° · confidence {conf if conf is not None else "нет принятого решения"}'
    fig.text(.5,.03,note,ha='center',fontsize=9);fig.subplots_adjust(top=.86,bottom=.17,left=.09,right=.97);fig.savefig(path,dpi=115);plt.close(fig)

def view3d(ax,xyz,pred,ref,step,upto=None):
    B0=np.asarray(pred['seed']['basis']);idx=pred['indices'];p=xyz[idx];mask=pred['point_step']<=(upto if upto is not None else 9999)
    pp=p[mask]@B0;ss=pred['point_step'][mask]
    if len(pp):ax.scatter(pp[:,0],pp[:,1],pp[:,2],c=ss,cmap='plasma',s=2,alpha=.8)
    scope=max(15,min(160,np.linalg.norm(p,axis=1).max()+10)) if len(p) else 20
    gg=ref['future'];gg=sample(gg[np.linalg.norm(gg,axis=1)<=scope],10000)@B0
    if len(gg):ax.scatter(gg[:,0],gg[:,1],gg[:,2],s=1,c=COL['future'],alpha=.22,label='Будущая разметка (оценка)')
    ax.scatter(0,0,0,c='black',s=40,marker='^');ax.text(0,0,.1,'LiDAR T',fontsize=8)
    for field,bfield,color in (('plane_origin','basis','#d75a31'),('previous_origin','previous_basis','#1e68aa')):
        if step.get(field) is None:continue
        C=np.asarray(step[field]);B=np.asarray(step[bfield]);rect=np.array([[0,-.22,-.20],[0,.22,-.20],[0,.22,.20],[0,-.22,.20]])
        rect=(C+rect@B.T)@B0;ax.add_collection3d(Poly3DCollection([rect],facecolors=color,edgecolors=color,alpha=.18))
        origin=C@B0
        for j in range(3):
            d=(B[:,j]*.7)@B0;ax.quiver(*origin,*d,color=color,linewidth=.9)
        if field=='plane_origin':
            other=rect+((B[:,0]*pred['config']['window'])@B0)
            for a,b in zip(rect,other):ax.plot(*np.array([a,b]).T,c=color,lw=.6,alpha=.6)
    ax.set(xlabel='u₀, м',ylabel='v₀, м',zlabel='w₀, м',xlim=(0,scope));ax.set_box_aspect((4,1,1),zoom=1.3);ax.view_init(elev=24,azim=-64)
    ax.yaxis.set_major_locator(MaxNLocator(3));ax.zaxis.set_major_locator(MaxNLocator(4));ax.tick_params(labelsize=8)
    ax.set_title('Реальные точки T; плоскости и базисы\nПоперечный масштаб увеличен',fontsize=10)

def terminal(task):
    run,i=task;folder=OUT/'heldout'/key(run,i);pred=load_prediction(folder)
    if pred['seed']['status']!='AVAILABLE':return key(run,i),'unavailable'
    out=OUT/'failures';step=pred['steps'][-1];a=out/(key(run,i)+'_without_gt.png');b=out/(key(run,i)+'_with_gt.png')
    if a.exists() and b.exists():return key(run,i),'cached'
    xyz=np.load(OUT/'cache'/key(run,i)/'xyz.npy');ref,_=reference(run,i,'heldout')
    slice_image(run,i,pred,xyz,ref,step,False,a);slice_image(run,i,pred,xyz,ref,step,True,b)
    return key(run,i),'rendered'

def choose_examples(rows):
    good=[r for r in rows if r.get('evaluation_eligible')];chosen={}
    def add(rr,tag,n):
        for r in rr[:n]:chosen.setdefault((r['run'],r['start_frame']),dict(row=r,tags=[]))['tags'].append(tag)
    ordered=sorted(good,key=lambda r:r['continuous_reach_5cm']);add(ordered[::-1],'best',5)
    mid=len(ordered)//2;add(ordered[max(0,mid-2):],'median',5)
    genuine=[r for r in ordered if r['terminal_evaluation_reason'] not in ('CR_GT_GAP','RUN_END','GT_UNAVAILABLE_AHEAD','CR_SIDE_CHANGE_SUSPECTED')]
    add(genuine,'early',10)
    add([r for r in ordered if r['terminal_evaluation_reason']=='ORIENTATION_FAILURE'],'orientation',5)
    add([r for r in ordered if r['terminal_evaluation_reason']=='SENSOR_SPARSITY'],'sparsity',5)
    add([r for r in rows if r['seed_available'] and (r.get('wrong_structure_suspected') or r.get('raw_wrong_structure_trigger'))],'wrong',len(rows))
    add(sorted([r for r in rows if r.get('seed_current_GT_conflict')],key=lambda r:-r['seed_current_GT_p95']),'gtconflict',5)
    add([r for r in ordered if r['terminal_evaluation_reason'] in ('CR_GT_GAP','CR_SIDE_CHANGE_SUSPECTED','RUN_END')],'gtgap',3)
    # Curvature/grade selection uses only evaluation columns after predictions.
    changes=[]
    for r in good:
        e=load(OUT/'heldout'/key(r['run'],r['start_frame'])/'evaluation.json')
        curve=max([s.get('gt_curvature_per_m',0) for s in e['steps']],default=0)
        grade=max([abs(s.get('gt_pitch_deg',0)) for s in e['steps']],default=0)
        changes.append((r,curve,grade))
    add([r[0] for r in sorted(changes,key=lambda x:-x[1])],'curve',2);add([r[0] for r in sorted(changes,key=lambda x:-x[2])],'grade',2)
    return list(chosen.values())

def representatives():
    examples=choose_examples(load(OUT/'analysis_rows.json'));save(OUT/'gallery/examples.json',examples)
    # At least 15 distinct exports; examples already contain >=20 by construction.
    for j,case in enumerate(examples):
        r=case['row'];run=r['run'];i=r['start_frame'];pred=load_prediction(OUT/'heldout'/key(run,i));xyz=np.load(OUT/'cache'/key(run,i)/'xyz.npy');ref,_=reference(run,i,'heldout')
        path=OUT/'gallery'/(key(run,i)+'_3d.png')
        reach='н/д' if r.get('continuous_reach_5cm') is None else f'{r["continuous_reach_5cm"]:.2f} м'
        fig=plt.figure(figsize=(14,7));ax=fig.add_subplot(111,projection='3d');view3d(ax,xyz,pred,ref,pred['steps'][-1]);fig.suptitle(f'{run} / {frames(run)[i]["file"]}\nКорректная непрерывная дальность ≤5 см: {reach}',fontsize=13);fig.tight_layout();fig.savefig(path,dpi=115);plt.close(fig)
        # Wide near view prevents a wrong-side annotation/seed conflict from
        # disappearing outside the narrow terminal tracker crop.
        B=np.array(pred['seed']['basis']);uv=xyz@B;gt=ref['current']@B
        scope=(uv[:,0]>=0)&(uv[:,0]<=8)&(abs(uv[:,1])<=2.5)&(uv[:,2]>-1.8)&(uv[:,2]<.2)
        q=sample(uv[scope],24000);fig,ax=plt.subplots(figsize=(13,5))
        ax.scatter(q[:,1],q[:,2],s=2,c='#b0bbc5',alpha=.4,linewidths=0,label='Облако T, 0–8 м')
        g=gt[(gt[:,0]>=0)&(gt[:,0]<=8)];ax.scatter(g[:,1],g[:,2],s=5,c=COL['current'],alpha=.5,linewidths=0,label='Исходный class2 T')
        seedids=np.array(pred['seed']['indices'],dtype=int);ax.scatter(uv[seedids,1],uv[seedids,2],s=5,c=COL['new'],alpha=.7,linewidths=0,label='Production seed без labels')
        ax.set(xlim=(-2.5,2.5),ylim=(-1.8,.2),xlabel='v₀, м',ylabel='w₀, м',title=f'{run} / {frames(run)[i]["file"]}: весь ближний срез, разметка только для оценки');ax.grid(alpha=.15);ax.legend();fig.tight_layout();fig.savefig(OUT/'gallery'/(key(run,i)+'_bootstrap.png'),dpi=125);plt.close(fig)
        if r.get('wrong_structure_suspected') or r.get('raw_wrong_structure_trigger'):
            ev=load(OUT/'heldout'/key(run,i)/'evaluation.json');bad=next((s for s in ev['steps'] if s['status']=='ACCEPTED' and s.get('point_p95') is not None and s['point_p95']>.20),None)
            if bad:
                step=next(s for s in pred['steps'] if s['step_index']==bad['step_index'])
                slice_image(run,i,pred,xyz,ref,step,True,OUT/'gallery'/(key(run,i)+'_first_error_with_gt.png'))
    # Distinct representatives covering requested categories; missing categories
    # are explicitly declared rather than fabricated.
    animation_cases=[]
    for tag in ('best','median','early','curve','grade','sparsity','wrong','gtgap','orientation'):
        case=next((e for e in examples if tag in e['tags'] and e not in animation_cases),None)
        if case:animation_cases.append(case)
    for case in examples:
        if len(animation_cases)>=8:break
        if case not in animation_cases:animation_cases.append(case)
    for case in animation_cases:
        r=case['row'];run=r['run'];i=r['start_frame'];pred=load_prediction(OUT/'heldout'/key(run,i));xyz=np.load(OUT/'cache'/key(run,i)/'xyz.npy');ref,_=reference(run,i,'heldout');images=[]
        for step in pred['steps']:
            fig=plt.figure(figsize=(13,5.8));ax=fig.add_subplot(121,projection='3d');view3d(ax,xyz,pred,ref,step,step['step_index']);ax=fig.add_subplot(122);slice_panel(ax,xyz,pred,step,ref,False)
            fig.suptitle(f'{run} / {frames(run)[i]["file"]} — шаг {step["step_index"]}: {step.get("failure_reason",step["status"])}',fontsize=10)
            fig.tight_layout(rect=(0,0,1,.94));buf=io.BytesIO();fig.savefig(buf,format='png',dpi=85);plt.close(fig);buf.seek(0);images.append(Image.open(buf).convert('RGB'))
        if images:images[0].save(OUT/'animations'/(key(run,i)+'.gif'),save_all=True,append_images=images[1:],duration=750,loop=0)
    save(OUT/'animations/index.json',[dict(run=e['row']['run'],frame=e['row']['start_frame'],tags=e['tags']) for e in animation_cases])
    print('REPRESENTATIVES',len(examples),'ANIMATIONS',len(animation_cases),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['terminal','representatives']);ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    if args.mode=='representatives':representatives();return
    tasks=[]
    for p in (OUT/'heldout').glob('*/prediction.json'):
        if not (p.parent/'evaluation.json').exists():continue
        r=load(p)
        if r['seed']['status']=='AVAILABLE':tasks.append((r['run'],r['start_frame']))
    t=time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(terminal,t) for t in tasks]
        for j,f in enumerate(as_completed(fs)):
            f.result()
            if j%50==0 or j==len(fs)-1:print('TERMINAL',j+1,len(fs),round(time.perf_counter()-t,1),flush=True)
if __name__=='__main__':main()
