"""Scientific PNG figures and local HTML, strictly after the frozen evaluation."""
from rr_common import *
sys.path.insert(0,str(ROOT/'results_rail2d/_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from concurrent.futures import ProcessPoolExecutor
from seed_input import read_input
import html,argparse

COLORS=['#df4938','#246cb4'];NAMES=['ближний','дальний'];Q95=5.991464547
CSS='body{font:16px/1.55 system-ui,sans-serif;background:#f4f6f8;color:#152938;margin:0}main{max-width:1450px;margin:auto;padding:28px}h1{font-size:30px}a{color:#126598}p{max-width:1100px}.note{padding:15px;background:#fff4d9;border-left:5px solid #d89b19}.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:18px}.card,section{background:white;border:1px solid #dce2e6;border-radius:9px;padding:16px;margin:14px 0}.card img,img.plot{width:100%;height:auto}.tags{font-size:13px;color:#566}.controls{position:sticky;top:0;background:#f4f6f8;padding:12px;z-index:2}select,input{padding:9px;margin:4px;border:1px solid #aaa;border-radius:5px}table{border-collapse:collapse}th,td{padding:7px;border:1px solid #ccd;text-align:right}th:first-child,td:first-child{text-align:left}.bad{color:#b52127}.muted{color:#627180}'
def esc(v):return html.escape(str(v))
def page(title,body):return '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+esc(title)+'</title><style>'+CSS+'</style><main>'+body+'</main></html>'
def label(item):return item['run']+' / '+item['file']
def polish(axes):
    for ax in np.asarray(axes).ravel():ax.grid(alpha=.18);ax.tick_params(labelsize=8)
def savefig(fig,path):fig.savefig(path,dpi=115);plt.close(fig)
def figure(item):
    folder=OUT/item['group']/item['method']/item['id'];rec=load(folder/'prediction.json');meta,a=read_input(OUT/rec['input_folder'])
    with np.load(folder/'prediction.npz') as z:p={k:z[k] for k in z.files}
    with np.load(folder/'evaluation.npz') as z:e={k:z[k] for k in z.files}
    with np.load(OUT/'oracle_decomposition'/item['group']/item['method']/item['id']/'curves.npz') as z:d={k:z[k] for k in z.files}
    dest=OUT/'gallery'/item['id'];dest.mkdir(parents=True,exist_ok=True);scale=load(OUT/'configs.json')[item['method']]['corridor_scale']/rec['config']['corridor_scale'];cov=p['rail_cov']*scale**2;s=p['s'];I=a['initial_basis'];C=p['C']@I;pred=p['pair']@I;gt=e['gt']@I;pts=a['cr_points'][::max(1,len(a['cr_points'])//3000)]@I
    fig,ax=plt.subplots(3,2,figsize=(16,13));polish(ax)
    for col,axis in enumerate((1,2)):
        ar=ax[0,col];ar.scatter(pts[:,0],pts[:,axis],s=2,c='#d294c1',alpha=.25,label='точки C4');ar.plot(C[:,0],C[:,axis],c='#8d258e',label='STEP6 spline по C4');anchors=a['anchor_xyz']@I;ar.plot(anchors[:,0],anchors[:,axis],'o--',ms=3,c='#666',lw=.9,label='исходные опоры C4')
        for j in (0,1):
            ar.plot(pred[:,j,0],pred[:,j,axis],c=COLORS[j],label='прогноз: '+NAMES[j]);ar.plot(gt[:,j,0],gt[:,j,axis],'.',c=COLORS[j],alpha=.4,label='GT: '+NAMES[j]);sigma=np.sqrt(np.einsum('i,nij,j->n',I[:,axis],cov[:,j],I[:,axis]));ar.fill_between(pred[:,j,0],pred[:,j,axis]-np.sqrt(Q95)*sigma,pred[:,j,axis]+np.sqrt(Q95)*sigma,color=COLORS[j],alpha=.1)
        ar.set(xlabel='u начального сечения, м',ylabel=('v' if axis==1 else 'w')+', м',title='План (u,v)' if axis==1 else 'Профиль (u,w); w — оценённая ось, не gravity truth',xlim=(max(-2,C[0,0]-2),C[-1,0]+3))
    ax[0,0].legend(fontsize=7,ncol=2)
    ar=ax[1,0];ar.plot(s,np.rad2deg(e['alpha_gt']),c='#222',label='α GT');ar.plot(s,np.rad2deg(e['alpha_pred_common']),c='#176b80',label='α прогноз');ar.fill_between(s,np.rad2deg(e['alpha_pred_common']-1.96*p['alpha_sigma']),np.rad2deg(e['alpha_pred_common']+1.96*p['alpha_sigma']),color='#176b80',alpha=.13,label='±1,96σ модели α');ar.set(xlabel='Общая CR station, м',ylabel='градусы',title='Поворот сечения');ar.legend(fontsize=8)
    ar=ax[1,1]
    for j in (0,1):ar.plot(s,e['error'][:,j]*100,c=COLORS[j],label=NAMES[j])
    ar.plot(s,e['center_error']*100,c='#687868',label='центр');ar.axhline(20,c='#b22',ls=':',label='20 см');ar.set(xlabel='Общая CR station, м',ylabel='ошибка, см',title='Поперечное расхождение с будущим GT');ar.legend(fontsize=8)
    ar=ax[2,0]
    for k,name,color in ((1,'d','#267e6b'),(2,'h near','#a57125'),(3,'g','#5768b4'),(4,'h far','#985895')):
        ar.plot(s,p['state'][:,k],color=color,label=name+' прогноз');ar.plot(s,d['oracle_state'][:,k],ls=':',color=color,label=name+' GT proxy')
    ar.set(xlabel='Общая CR station, м',ylabel='м',title='Расстояния от seed и диагностического эталона');ar.legend(fontsize=7,ncol=2)
    ar=ax[2,1]
    for j in (0,1):
        for axis,style in (('lateral','-'),('vertical','--')):ar.plot(s,2*np.sqrt(Q95)*p['sigma_'+axis][:,j]*scale*100,c=COLORS[j],ls=style,label=NAMES[j]+(' lateral' if axis=='lateral' else ' vertical'))
    ar.set(xlabel='Общая CR station, м',ylabel='полная ширина, см',title='Номинальный 95% коридор после development-калибровки');ar.legend(fontsize=8,ncol=2)
    for ar in ax.ravel():
        if ar not in ax[0]:
            for k in np.flatnonzero(p['cr_state']!=2):ar.axvspan(s[k]-.5,s[k]+.5,color='#d8ac2b',alpha=.035)
    fig.suptitle(label(item)+'\n'+item['method']+' · '+('development' if item['group']=='phase_b' else 'reused benchmark')+' · far p95 = %.1f см, max = %.1f см'%(item['far_p95']*100,item['far_max']*100),fontsize=13);fig.tight_layout(rect=(0,0,1,.94));savefig(fig,dest/'overview.png')
    # Fixed 10 m snapshot grid; absence is shown instead of extrapolation.
    fig,axes=plt.subplots(4,3,figsize=(16,14));polish(axes)
    for ar,station in zip(axes.ravel(),range(10,121,10)):
        if station>s[-1]+.01:ar.text(.5,.5,'C4 не достиг этой station\nПрогноз не продлевается',ha='center',va='center',transform=ar.transAxes,color='#788');ar.set_title(str(station)+' м');continue
        k=int(np.argmin(abs(s-station)));B=a['B'][k];T=B[:,1:];xy=(p['pair'][k]-p['C'][k])@T;truth=(e['gt'][k]-p['C'][k])@T;ar.scatter([0],[0],marker='D',c='#8d258e',s=45,label='CR')
        for j in (0,1):
            cc=T.T@cov[k,j]@T;val,vec=np.linalg.eigh(cc);theta=np.rad2deg(np.arctan2(vec[1,1],vec[0,1]));ell=Ellipse(xy[j],2*np.sqrt(Q95*max(val[1],0)),2*np.sqrt(Q95*max(val[0],0)),angle=theta,color=COLORS[j],alpha=.13);ar.add_patch(ell);ar.scatter(*xy[j],marker='x',s=55,c=COLORS[j],label=NAMES[j]);ar.scatter(*truth[j],marker='*',s=65,c='#111',label='GT' if j==0 else None)
        ar.set_title('%d м · α %.2f° · %s'%(station,np.rad2deg(e['alpha_pred_common'][k]),'GT есть' if e['valid'][k] else 'GT отсутствует'),fontsize=9);ar.set(xlabel='e1 относительно CR, м',ylabel='e2, м');ar.set_aspect('equal',adjustable='datalim');ar.autoscale_view();ar.margins(.18)
    handles,labels=axes[0,0].get_legend_handles_labels()
    if handles:fig.legend(handles,labels,loc='lower center',ncol=4)
    fig.suptitle(label(item)+'\nОбщие поперечные плоскости · эллипсы 95% · синтетические рельсы',fontsize=13);fig.tight_layout(rect=(.01,.025,.99,.95));savefig(fig,dest/'sections.png')
    variants=[('production_CR_production_alpha','C4 + прогноз α'),('oracle_CR_production_alpha','Oracle CR + прогноз α'),('production_CR_oracle_alpha','C4 + oracle α'),('all_oracle','All oracle: алгебраическая проверка')];fig,axes=plt.subplots(2,2,figsize=(15,8));polish(axes)
    for ar,(variant,title) in zip(axes.ravel(),variants):
        if variant not in d:ar.text(.5,.5,'Нет сопоставимого CR-эталона',transform=ar.transAxes,ha='center');continue
        delta=d[variant]-d['gt'];delta-=np.einsum('nri,ni->nr',delta,d['common_B'][:,:,0])[:,:,None]*d['common_B'][:,None,:,0];err=np.linalg.norm(delta,axis=2);err[~d['valid']]=np.nan
        for j in (0,1):ar.plot(s,err[:,j]*100,c=COLORS[j],label=NAMES[j])
        ar.axhline(20,c='#b22',ls=':');ar.set(xlabel='Общая CR station, м',ylabel='ошибка, см',title=title);ar.legend(fontsize=8)
    fig.suptitle(label(item)+'\nДиагностика на одинаковых сечениях; oracle использует будущую разметку и не является алгоритмом',fontsize=12);fig.tight_layout(rect=(0,0,1,.92));savefig(fig,dest/'oracles.png')
    cause=', '.join(item['cause']);body=f'<a href="../../gallery.html">← Все примеры</a> · <a href="../../REPORT_RUNNING_FROM_CR.html">Полный отчёт</a><h1>{esc(label(item))}</h1><p>{esc(item["method"])} · {esc(item["group"])} · {esc(", ".join(item["tags"]))}</p><p class="note">Far p95: {item["far_p95"]*100:.1f} см; максимум: {item["far_max"]*100:.1f} см на station {item["worst_station"]:.1f} м (range {item["worst_range"]:.1f} м). Ближний в этом сечении: {item["worst_near"]*100:.1f} см. Диагностические признаки: {esc(cause)}. Это расхождение с расчётным GT; оно само по себе не доказывает ошибку алгоритма.</p><p>p95 разброса future GT: {item["GT_spread_p95"]*100:.1f} см. Разметка и регистрация не являются геодезическим эталоном. Полосы рельсов используют зафиксированную development-калибровку; полоса α показывает исходную модельную дисперсию. Жёлтым отмечены tentative/gap stations.</p>'
    notes=load(STAGE/'MANUAL_REVIEW_RU.json')
    if item['id'] in notes:body+='<section><h2>Разбор после просмотра графиков</h2><p>'+esc(notes[item['id']])+'</p></section>'
    for f,title in [('overview','Геометрия, угол и неопределённость'),('sections','Срезы через 10 м'),('oracles','Что меняют известные CR и угол')]:body+=f'<section><h2>{title}</h2><a href="{f}.png"><img class="plot" src="{f}.png" loading="lazy" alt="{title}"></a></section>'
    body+='<p>Строгий прогноз после seed не использует ходовые рельсы. Oracle-ветви изолированы и предназначены только для объяснения ошибки. Отсутствующий дальний прогноз не дорисован.</p>'
    (dest/'index.html').write_text(page(label(item),body),encoding='utf-8');return item['id']

def index(items):
    names=dict(best='Лучшие 5',median='Средние 10',worst_far='Худшие 20',high_roll='Большая ошибка roll 10',curve='Повороты 10',grade='Уклоны 10',all_gt20cm='Все случаи >20 см');body='<h1>STEP 6 · Ходовые рельсы из C4</h1><p><a href="REPORT_RUNNING_FROM_CR.html">Полный отчёт</a> · <a href="research_freeze.json">Research freeze</a> · <a href="las_exports.csv">LAS exports</a></p><p class="note">C4 неизменён. Красный — ближний рельс, синий — дальний. Линии — синтетический геометрический прогноз; GT вычислен после прогноза. Лучшие и средние выбираются среди стартов с ≥30 м проверяемого горизонта. Все случаи с ошибкой дальнего рельса >20 см после seed включены отдельно.</p><div class="controls"><select id="filter"><option value="">Все категории</option>'+''.join(f'<option value="{k}">{v}</option>' for k,v in names.items())+'</select><input id="search" placeholder="Запись или кадр"><span id="count"></span></div><div class="cards">'
    for item in items:
        url='gallery/'+item['id']+'/index.html';body+=f'<article class="card" data-tags="{esc(" ".join(item["tags"]))}" data-search="{esc(label(item).lower())}"><a href="{url}"><img src="gallery/{item["id"]}/overview.png" loading="lazy" alt="{esc(label(item))}"></a><a href="{url}">{esc(label(item))}</a><p>far p95 <b>{item["far_p95"]*100:.1f} см</b> · max {item["far_max"]*100:.1f} см<br>Горизонт с GT {item["max_evaluated"]:.1f} м</p><p class="tags">{esc(", ".join(names[t] for t in item["tags"]))}</p></article>'
    body+='</div><script src="gallery/filter.js"></script>';(OUT/'gallery.html').write_text(page('STEP6 — галерея',body),encoding='utf-8')
    (OUT/'gallery/filter.js').write_text("'use strict';const f=document.getElementById('filter'),q=document.getElementById('search'),c=document.getElementById('count'),cards=[...document.querySelectorAll('.card')];function update(){let n=0;for(const a of cards){const show=(!f.value||a.dataset.tags.split(' ').includes(f.value))&&a.dataset.search.includes(q.value.toLowerCase());a.hidden=!show;if(show)n++;}c.textContent=n+' примеров';}f.addEventListener('change',update);q.addEventListener('input',update);if(location.hash)f.value=location.hash.slice(1);update();",encoding='utf-8')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);ap.add_argument('--limit',type=int);args=ap.parse_args();items=load(OUT/'gallery/cases.json');selected=items[:args.limit] if args.limit else items
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for j,result in enumerate(pool.map(figure,selected),1):
            if j%10==0 or j==len(selected):print('FIGURES',j,len(selected),result,flush=True)
    index(items);save(OUT/'gallery/RENDER_COMPLETE.json',dict(time_ns=time.time_ns(),rendered=len(selected),total=len(items),browser_tested=False,static_png_review=True))

if __name__=='__main__':main()
