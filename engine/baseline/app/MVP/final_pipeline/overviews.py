from . import ROOT
import sys
sys.path.insert(0,str(ROOT/'results_rail2d/_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import long_common as lc
import laspy

OUT=ROOT/'results_final_pipeline'
COLORS={1:'#168455',2:'#2265c6',3:'#e68a16'}


def main():
    selected=lc.load(OUT/'selected_las_runs.json')['runs'];images=[]
    for choice in selected:
        run,role=choice['run'],choice['role'];records=lc.frames(run)
        pose=np.asarray(records[0]['lidar_pose_in_folder']);origin=pose[:3,3]
        phase='deployable_benchmark_cache' if role=='straight_run' else 'deployable_curved_run_cache'
        first=lc.load(OUT/phase/lc.key(run,0)/'c4.json');B=np.array(first['seed']['basis']);direction=pose[:3,:3]@B[:,0]
        f=np.array([direction[0],direction[1],0.]);f/=np.linalg.norm(f);v=np.cross([0,0,1],f);axes=np.column_stack((f,v,[0,0,1]))
        with np.load(OUT/'visualizations'/(run+'_real_preview.npz')) as z:points=(z['world']-origin)@axes;stage=z['stage']
        las=laspy.read(OUT/'las'/role/(run+'_pipeline_geometry.las'))
        step=max(1,len(las.points)//60000);ids=np.arange(0,len(las.points),step)
        geometry=(np.column_stack((np.asarray(las.x)[ids],np.asarray(las.y)[ids],np.asarray(las.z)[ids]))-origin)@axes
        component=np.asarray(las.pipeline_component)[ids]
        fig=plt.figure(figsize=(18,13),layout='constrained');a=[fig.add_subplot(3,2,j+1,projection='3d' if j==4 else None) for j in range(6)]
        for ax in a:
            if ax.name!='3d':ax.grid(alpha=.2)
        for k,color in COLORS.items():
            take=stage==k;a[0].scatter(points[take,0],points[take,1],s=1,c=color,label=f'first stage {k}',rasterized=True)
            if k in (2,3):a[1].scatter(points[take,0],points[take,1],s=1,c=color,label='STEP2' if k==2 else 'C4',rasterized=True)
        a[0].set(title='A · First-found, полный run',xlabel='along initial horizontal direction, м',ylabel='cross direction, м');a[0].legend()
        a[1].set(title='B · Только contact rail: bootstrap / marching',xlabel='along initial direction, м',ylabel='cross direction, м');a[1].legend()
        for k,label,color in [(4,'CR reference','#172d46'),(5,'LEFT','#168455'),(6,'RIGHT','#e68a16'),(7,'center','#7650a1')]:
            take=component==k
            a[2].scatter(geometry[take,0],geometry[take,1],s=.5,c=color,label=label,rasterized=True)
            a[3].scatter(geometry[take,0],geometry[take,2],s=.5,c=color,label=label,rasterized=True)
            a[4].scatter(geometry[take,0],geometry[take,1],geometry[take,2],s=.4,c=color,label=label,rasterized=True)
        a[2].set(title='C · All final geometry, все прогнозы в общей системе',xlabel='along initial direction, м',ylabel='cross direction, м');a[2].legend()
        a[3].set(title='D · Elevation, estimated registration up',xlabel='along initial direction, м',ylabel='Δ world Z, м');a[3].legend()
        a[4].set(title='E · 3D: CR, два рельса, центр',xlabel='longitudinal, м',ylabel='lateral, м',zlabel='ΔZ, м');a[4].set_box_aspect((3,1,.8));a[4].view_init(24,-70)
        for index in np.linspace(0,len(records)-2,6,dtype=int):
            path=OUT/phase/lc.key(run,int(index))/'curve.npz'
            if not path.exists():continue
            with np.load(path) as z:C=z['C'];s=z['s']
            R=np.asarray(records[int(index)]['lidar_pose_in_folder'])[:3,:3];d=np.gradient(C@R.T,s,axis=0)
            heading=np.unwrap(np.arctan2(d[:,1],d[:,0]));a[5].plot(s,np.gradient(heading,s)*1000,label=f'T={index}')
        a[5].set(title='F · Curvature vs station: шесть распределённых стартов',xlabel='station in current forecast, м',ylabel='κ × 1000, 1/м');a[5].legend(fontsize=8)
        fig.suptitle(f'{role.upper()} · {run}\n{len(records)} кадров · реальные точки: deterministic 1/53 sample · geometry: overlapping per-frame forecasts, NOT new LiDAR returns',fontsize=14)
        path=OUT/'visualizations'/(role+'_overview.png');fig.savefig(path,dpi=150);plt.close(fig);images.append(path.relative_to(OUT).as_posix())
    # Compact engineering comparison figure.
    fig,a=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    names=['baseline','raw_cache','cache','preloaded'];med=[];p95=[]
    for name in names:
        rows=lc.load(OUT/'profiling_ablation'/f'{name}.json')['rows'];v=np.array([r['timing']['T_TOTAL']*1000 for r in rows]);med.append(np.median(v));p95.append(np.percentile(v,95))
    x=np.arange(4);a[0].bar(x-.18,med,.36,label='median');a[0].bar(x+.18,p95,.36,label='p95');a[0].set_xticks(x,names,rotation=15);a[0].set(ylabel='ms/frame',title='Serial 32-start benchmark, failures included');a[0].legend()
    q=lc.load(OUT/'deployable_benchmark_cache/QUALITY.json');new=[];old=[]
    for lo in (30,50,75):
        new.append(next(r['far_p95'] for r in q['metrics'] if r['lo']==lo and r['method']=='DEPLOYABLE_PIPELINE')*100)
        old.append(next(r['far_p95'] for r in q['metrics'] if r['lo']==lo and r['method']=='RESEARCH_REFERENCE')*100)
    x=np.arange(3);a[1].bar(x-.18,old,.36,label='research GT-seed');a[1].bar(x+.18,new,.36,label='deployable RAW seed');a[1].set_xticks(x,['30–50 m','50–75 m','75–100 m']);a[1].set(ylabel='far rail pooled p95, cm',title='Benchmark: matched common stations');a[1].legend()
    fig.savefig(OUT/'visualizations/performance_quality.png',dpi=160);plt.close(fig)
    path=OUT/'gallery.html';page=path.read_text(encoding='utf-8')
    block='<section><h2>Два полных diagnostic runs</h2>'+''.join(f'<a href="{p}"><img loading="lazy" src="{p}" alt="Full run overview"></a>' for p in images)+'</section>'
    page=page.replace('</main>',block+'</main>');path.write_text(page,encoding='utf-8')
    lc.save(OUT/'visualizations/overviews.json',images)


if __name__=='__main__':main()
