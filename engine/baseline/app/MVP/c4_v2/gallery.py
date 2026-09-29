"""Scientific diagnostic plots and offline HTML gallery, post-evaluation only."""
from . import ROOT,OUT
import sys
sys.path.insert(0,str(ROOT/'results_rail2d/_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import long_common as lc
from .evaluate import read
from .persistent import sensor,stations
from MVP.final_pipeline.frame_data import FrameData
import html


def main():
    records=lc.csv_read(OUT/'worst_cases.csv');available=[]
    for r in records:
        if r['far_variant_p95'] not in ('','None'):available.append(r)
    ranked=sorted(available,key=lambda r:float(r['far_variant_p95']))
    chosen=[];used=set()
    def add(category,items):
        for r in items:
            key=(r['group'],r['run'],r['frame'])
            if key not in used:chosen.append((category,r));used.add(key)
    add('Худшие регрессии',available[:8]);add('Лучшие',ranked[:2]);mid=len(ranked)//2;add('Средние',ranked[mid:mid+3])
    # Also include representative requested route shapes by actual recordings.
    for run in ('roundT_squareT_pressureGate_squareT','roundT_pressureGate_roundT','squareT_platform_squareT_switch','new_data_part_01a2','new_data_part_01d1'):
        candidates=[r for r in ranked if r['run']==run]
        if candidates:add('Представительный маршрут',[candidates[len(candidates)//2]])
    images=OUT/'gallery';images.mkdir(exist_ok=True);cards=[]
    for category,row in chosen:
        run=row['run'];i=int(row['frame']);root=OUT/('development/combined_r8_o8' if row['group']=='development' else 'benchmark/locked_r8_o8');folder=root/run/f'{i:06d}'
        a=read(folder/'reference.npz');b=read(folder/'prediction.npz');e=read(folder/'evaluation.npz');mp=read(folder/'persistent_map.npz')
        meta=lc.load(next((lc.dataset()/run).glob('*.frames.json')))['frames'];P=np.asarray(meta[i]['lidar_pose_in_folder']);B=a['B'][0];gt=e['gt'];rng=e['ranges'];valid=e['valid']
        transform=lambda x:np.asarray(x)@B
        ac=transform(a['C']);bc=transform(b['C']);ap=transform(a['pair']);bp=transform(b['pair']);g=transform(gt)
        mapc=sensor(mp['anchors'],P);mc=transform(mapc);ms=stations(mp['anchors']);timings=lc.load(root/run/'timings.json');tr=next(r for r in timings if r['frame']==i)
        boundary=ms[-1]-tr['extension_m']-tr['repair_m'];extension=ms[-1]-tr['extension_m'];colors=np.where(ms<boundary,'#247cc9',np.where(ms<extension,'#ee9322','#23a76a'))
        if tr['fallback'] or tr['status']=='COLD_FULL_BOOTSTRAP':colors[:]='#d64a44'
        fig,axes=plt.subplots(2,2,figsize=(16,10));ax=axes[0,0]
        raw=FrameData.read(lc.dataset()/run/meta[i]['file'],i);rawq=transform(sensor(raw.world,P))
        railq=ap.reshape(-1,3);take=(rawq[:,0]>=0)&(rawq[:,0]<=min(130,max(ac[-1,0],bc[-1,0])+3))&(rawq[:,1]>=railq[:,1].min()-.8)&(rawq[:,1]<=railq[:,1].max()+.8)&(rawq[:,2]>=railq[:,2].min()-.35)&(rawq[:,2]<=max(railq[:,2].max(),ac[:,2].max())+.35)
        rawq=rawq[take];rawq=rawq[::max(1,int(np.ceil(len(rawq)/12000)))]
        ax.scatter(rawq[:,0],rawq[:,1],s=.4,color='#aab5c0',alpha=.22,label='Точки текущего T, прорежены для рисунка')
        axes[0,1].scatter(rawq[:,0],rawq[:,2],s=.4,color='#aab5c0',alpha=.22)
        previous=sorted(p for p in (root/run).glob('*/persistent_map.npz') if int(p.parent.name)<i)
        if previous:
            previous_path=previous[-1];old_map=read(previous_path);prev=transform(sensor(old_map['anchors'],P))
            ax.plot(prev[:,0],prev[:,1],color='#a6a6a6',ls='--',lw=1,label=f'Предыдущий снимок карты T={int(previous_path.parent.name)}')
        ax.plot(ac[:,0],ac[:,1],color='#171f2c',lw=2,label='CR reference full solve');ax.plot(bc[:,0],bc[:,1],color='#bd39ad',lw=2,label='CR persistent V2')
        for rail in (0,1):
            ax.plot(ap[:,rail,0],ap[:,rail,1],color='#8090a0',lw=1.4,label='Ходовые reference' if rail==0 else None)
            ax.plot(bp[:,rail,0],bp[:,rail,1],color=('#0b9cba' if rail==0 else '#ac52e0'),lw=1.8,label=('Ближний V2' if rail==0 else 'Дальний V2'))
            ax.scatter(g[:,rail,0],g[:,rail,1],s=8,color='#248a4a',alpha=.5,label='GT после inference' if rail==0 else None)
        ax.scatter(mc[:,0],mc[:,1],c=colors,s=22,zorder=4,label='Карта: prefix/repair/extension/reset');ax.set(xlabel='Продольная координата, м',ylabel='Поперечная координата, м',title='План: контактный и два ходовых рельса');ax.legend(fontsize=8,ncol=2)
        ax=axes[0,1];ax.plot(ac[:,0],ac[:,2],color='#171f2c',label='CR reference');ax.plot(bc[:,0],bc[:,2],color='#bd39ad',label='CR V2')
        for rail in (0,1):
            ax.plot(ap[:,rail,0],ap[:,rail,2],color='#8090a0',alpha=.8)
            ax.plot(bp[:,rail,0],bp[:,rail,2],color=('#0b9cba' if rail==0 else '#ac52e0'),label=('Ближний V2' if rail==0 else 'Дальний V2'))
            ax.scatter(g[:,rail,0],g[:,rail,2],s=8,color='#248a4a',alpha=.5)
        ax.set(xlabel='Продольная координата, м',ylabel='Высота в исходном срезе, м',title='Высотный профиль');ax.legend(fontsize=8)
        ax=axes[1,0];ax.plot(rng[valid],e['reference'][valid,1]*100,color='#48586a',label='Reference far error');ax.plot(rng[valid],e['variant'][valid,1]*100,color='#d85c27',label='V2 far error');ax.axhline(20,color='#b63732',ls='--',label='20 см');ax.axhline(10,color='#bbb',ls=':');ax.set(xlabel='Дальность от текущего LiDAR, м',ylabel='Ошибка относительно GT, см',title='Те же плоскости, тот же GT');ax.legend(fontsize=8)
        ax=axes[1,1];ax.plot(rng,e['cr_change']*100,color='#bd39ad',label='Δ CR V2/reference')
        if 'variant_cr' in e:
            delta=(e['variant_pair'][:,1]-e['variant_cr'])-(a['pair'][:,1]-a['C']);normal=a['B'][:,:,0];delta-=np.einsum('ij,ij->i',delta,normal)[:,None]*normal
            ax.plot(rng,np.linalg.norm(delta,axis=1)*100,color='#0b9cba',label='Δ относительного положения far/CR')
        ax.set(xlabel='Дальность, м',ylabel='Расхождение прогнозов, см',title='Диагностика источника расхождения, не новый GT');ax.legend(fontsize=8)
        for ax in axes.flat:ax.grid(alpha=.2)
        fig.suptitle(f'{category}: {run} / frame_{i:06d}.las\n'+f"{tr['status']} · seed frame {tr['origin_frame']} · после seed {tr['meters_since_cold']:.1f} м · {int(valid.sum())} общих GT срезов · full update {tr['total_ms']:.0f} мс",fontsize=15)
        fig.tight_layout(rect=(0,0,1,.94));filename=f'{row["group"]}__{run}__{i:06d}.png';fig.savefig(images/filename,dpi=120);plt.close(fig)
        crdelta=float(row['cr_disagreement_p95']) if row['cr_disagreement_p95'] not in ('','None') else np.nan
        explanation=f"Far p95: {100*float(row['far_reference_p95']):.2f} → {100*float(row['far_variant_p95']):.2f} см. CR disagreement p95: {100*crdelta:.2f} см."
        cards.append(dict(category=category,run=run,frame=i,image='gallery/'+filename,explanation=explanation,status=tr['status']))
    lc.save(OUT/'gallery_cases.json',cards)
    style='body{font:16px/1.5 system-ui;background:#eef3f8;color:#203348;max-width:1600px;margin:auto;padding:25px}img{width:100%;background:white}article{background:white;padding:20px;margin:28px 0;border-radius:10px}a{color:#146da8}.note{padding:20px;background:#fff1dd}h1{font-size:30px}'
    body='<h1>C4 V2 — лучшие, средние и худшие случаи</h1><p><a href="REPORT_C4_V2_FINAL.html">Полный отчёт</a> · <a href="worst_cases.csv">Все ранжированные случаи CSV</a></p><div class="note">Persistent V2 не принят: quality FAIL. GT приближённый, из исходной разметки. Синий — сохранённый prefix, оранжевый — ремонт, зелёный — продолжение, красный — reset/full bootstrap. Линии прогноза — геометрия, не дополнительные реальные точки LAS. На графиках ошибки сравниваются одни и те же плоскости.</div>'
    for category in dict.fromkeys(c['category'] for c in cards):
        body+='<h2>'+html.escape(category)+'</h2>'
        for c in cards:
            if c['category']!=category:continue
            body+=f'<article><h3>{html.escape(c["run"])} / {c["frame"]:06d}</h3><p>{html.escape(c["explanation"])}</p><a href="{c["image"]}"><img loading="lazy" src="{c["image"]}"></a></article>'
    (OUT/'gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>C4 V2 галерея</title><style>'+style+'</style>'+body,encoding='utf-8')
    print('GALLERY',len(cards),flush=True)

if __name__=='__main__':main()
