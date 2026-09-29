"""Visual decomposition of evaluated >20cm suspects; does not alter metrics."""
from fusion_common import *
sys.path.insert(0,str(ROOT/'results_rail2d/_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from assess import reference_readonly
from scipy.spatial import cKDTree

def main():
    flagged=[]
    for name in load(OUT/'research_lock.json')['heldout_variants']:
        for path in (OUT/'heldout'/name).glob('*/evaluation.json'):
            e=load(path)
            if e['summary'].get('wrong_structure_suspected'):flagged.append((path.parent,e))
    rows=[];cases=[]
    for folder,e in flagged:
        p=load_prediction(folder);run=p['run'];i=p['start_frame'];name=p['variant'];ref,info=reference_readonly(run,i)
        with np.load(folder/'provenance.npz') as z:xyz=z['xyz'];age=z['age_frames']
        with np.load(folder/'point_evaluation.npz') as z:valid=z['evaluable']
        for es in e['steps']:
            if es['status']!='ACCEPTED' or (es.get('point_p95') or 0)<=.2:continue
            st=next(s for s in p['steps'] if s['step_index']==es['step_index']);B=np.array(st['basis']);C=np.array(st['plane_origin']);slots=np.flatnonzero(p['point_step']==st['step_index']);pp=xyz[slots]
            distance,nearest=cKDTree(ref['future']).query(pp);diff=(pp-ref['future'][nearest])@B;q=(pp-C)@B;g=(ref['future']-C)@B
            for j,k in enumerate(slots):rows.append(dict(variant=name,run=run,start_frame=i,step_index=st['step_index'],source_age=int(age[k]),u=float(q[j,0]),distance=float(distance[j]),du=float(diff[j,0]),dv=float(diff[j,1]),dw=float(diff[j,2]),evaluated=bool(valid[k]),GT_max_u=float(g[:,0].max()),horizon=info['horizon_reason']))
            bad=(distance>.2)&valid[slots];transverse=np.linalg.norm(diff[:,1:],axis=1)
            axial_end=bool(bad.any() and info['horizon_reason']=='RUN_END' and np.all(q[bad,0]>g[:,0].max()) and np.all(transverse[bad]<.05))
            case=dict(variant=name,run=run,start_frame=i,step_index=st['step_index'],evaluated_bad_points=int(bad.sum()),end_of_reference_with_small_transverse_residual=axial_end,GT_max_u=float(g[:,0].max()),max_bad_transverse=float(transverse[bad].max()) if bad.any() else None,official_suspicion_unchanged=True);cases.append(case)
            fig,axes=plt.subplots(1,2,figsize=(12,4.5));ids=np.flatnonzero((g[:,0]>=3)&(g[:,0]<=8))
            if len(ids)>2500:ids=ids[np.linspace(0,len(ids)-1,2500).astype(int)]
            for j,ax in enumerate(axes,1):
                ax.scatter(g[ids,0],g[ids,j],s=3,c='#149776',alpha=.3,label='Future GT')
                for a in np.unique(age[slots]):
                    m=age[slots]==a;ax.scatter(q[m,0],q[m,j],s=35,label='T' if a==0 else f'T−{a}')
                ax.scatter(q[bad,0],q[bad,j],s=100,facecolors='none',edgecolors='#d52636',label='Оценённая ошибка >20 см')
                ax.axvline(g[:,0].max(),color='#333',ls='--',label='Конец GT');ax.set(xlim=(3,8),xlabel='u вдоль окна, м',ylabel=('v' if j==1 else 'w')+', м');ax.grid(alpha=.2)
            axes[1].legend(fontsize=8);fig.suptitle(f'{name} · {run} / {frames(run)[i]["file"]} · шаг {st["step_index"]}\nПродольный разрыв у конца reference; официальная метрика не меняется',fontsize=11);fig.tight_layout(rect=(0,0,1,.9))
            fig.savefig(OUT/'gallery'/(name+'__'+key(run,i)+'_gt_end_diagnostic.png'),dpi=140);plt.close(fig)
    csv_write(OUT/'audit/suspect_point_breakdown.csv',rows);save(OUT/'audit/suspect_review.json',cases);print('SUSPECT REVIEW',cases,flush=True)
if __name__=='__main__':main()
