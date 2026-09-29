"""Evaluation-only view of the shared bootstrap/reference disagreement."""
from render_long import *

def main():
    e=next(e for e in load(OUT/'gallery/examples.json') if e['variant']=='C4' and e['run']=='squareT_platform_squareT_switch' and e['start_frame']==830)
    p,c,ref,info=case_data(e);B=np.asarray(p['seed']['basis']);q=c['xyz']@B;gt=ref['current']@B;seed=q[p['indices'][p['point_step']==0]]
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    current=c['age_frames']==0
    for ax,kind in zip(axes,('top','profile')):
        upper=12 if kind=='top' else 8;m=current&(q[:,0]>=0)&(q[:,0]<=upper)&(abs(q[:,1])<=2.2)&(q[:,2]>=-1.7)&(q[:,2]<=.2);ids=subset(np.flatnonzero(m),30000);g=(gt[:,0]>=0)&(gt[:,0]<=upper)
        x,y=(0,1) if kind=='top' else (1,2)
        ax.scatter(q[ids,x],q[ids,y],s=1,c='#b1bcc2',alpha=.4,label='Текущие исходные точки')
        ax.scatter(gt[g,x],gt[g,y],s=14,c='#008a6a',alpha=.7,label='Текущий class2 GT')
        ax.scatter(seed[:,x],seed[:,y],s=3,c='#d54a46',label='Общий frozen bootstrap')
        ax.set(xlabel='u, м' if kind=='top' else 'v, м',ylabel='v, м' if kind=='top' else 'w, м',title='Вид сверху: оба борта' if kind=='top' else 'Поперечный срез 0–8 м')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[0].set(xlim=(0,12),ylim=(-2.2,2.2));axes[1].set(xlim=(-2.2,2.2),ylim=(-1.7,.2));axes[1].set_aspect('equal')
    fig.suptitle('squareT_platform_squareT_switch / frame_000831.las\nФлаг >1 м относится к общему seed; future GT не построен. Нужна ручная проверка стороны/разметки.',fontsize=11)
    fig.tight_layout(rect=(0,0,1,.90));folder=OUT/'gallery'/e['id'];fig.savefig(folder/'seed_audit.png',dpi=140);plt.close(fig)
    complete=load(folder/'COMPLETE.json');complete['files']=list(dict.fromkeys(complete['files']+['seed_audit.png']));save(folder/'COMPLETE.json',complete)
    save(OUT/'audit/shared_seed_disagreement.json',dict(run=e['run'],start_frame=830,reference=info,scope='Shared frozen seed and current reference disagree; future reference has no component. No GT correction and no inference change.'))

if __name__=='__main__':main()
