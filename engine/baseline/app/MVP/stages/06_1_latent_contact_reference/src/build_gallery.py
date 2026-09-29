"""Static scientific figures and local HTML. Never used for inference/selection."""
from lc_common import *
sys.path.insert(0,str(ROOT/'results_rail2d/_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator
from concurrent.futures import ProcessPoolExecutor
import html

OLD='#d9543c';NEW='#087e70';GT='#313945';RAW='#a888b9';CSS='body{margin:0;background:#f3f5f7;color:#183044;font:16px/1.6 system-ui,sans-serif}main{max-width:1440px;padding:28px;margin:auto}h1{font-size:30px}a{color:#17689a}section,.card{background:white;border:1px solid #d6dfe5;border-radius:9px;padding:18px;margin:16px 0}.note{border-left:5px solid #c79222;background:#fff5dc;padding:16px}.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px}.card img,.plot{width:100%;height:auto}.controls{position:sticky;top:0;background:#f3f5f7;padding:12px;z-index:3}input,select{font:inherit;padding:8px;margin:4px}td,th{border:1px solid #d6dfe5;padding:7px;text-align:right}td:first-child,th:first-child{text-align:left}table{border-collapse:collapse}.muted{color:#627586}code{background:#f1f4f5;padding:2px 5px}'
TAGS={'overshoot':'Выброс сплайна','partial-profile':'Частичный профиль','vertical-wave':'Вертикальная волна','horizontal-kink':'Излом в плане','sparse':'Мало точек','curve':'Поворот','grade':'Уклон','gap':'Пропуск опор','best':'Лучшие','median':'Средние','worst':'Худшие','gt20':'Ошибка >20 см','old-failure':'Все старые ошибки >20 см'}
def esc(x):return html.escape(str(x))
def page(title,body):return '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+esc(title)+'</title><style>'+CSS+'</style><main>'+body+'</main></html>'
def figsave(fig,path):
    for ax in fig.axes:
        if hasattr(ax,'zaxis'):
            ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(3));ax.zaxis.set_major_locator(MaxNLocator(4));ax.tick_params(labelsize=7,pad=1)
    fig.savefig(path,dpi=115);plt.close(fig)
def polish(ax):
    for a in np.asarray(ax).ravel():a.grid(alpha=.17);a.tick_params(labelsize=8)
def ellipse(ax,mean,cov,color=NEW):
    if not np.isfinite(cov).all():return
    v,U=np.linalg.eigh(cov);ax.add_patch(Ellipse(mean,2*np.sqrt(5.99146*max(v[1],0)),2*np.sqrt(5.99146*max(v[0],0)),angle=np.rad2deg(np.arctan2(U[1,1],U[0,1])),facecolor=color,edgecolor=color,alpha=.16))

def figure(item):
    run,i=item['run'],item['frame'];folder=OUT/item['group']/item['method']/item['id'];dest=OUT/'gallery'/item['id'];dest.mkdir(parents=True,exist_ok=True);info=load(folder/'details.json');m,a,p0,pr=__import__('latent_inference').read_causal(run,i)
    def npz(path):
        with np.load(path) as z:return {k:z[k] for k in z.files}
    p=npz(folder/'prediction.npz');c=npz(folder/'curve.npz');e=npz(folder/'evaluation.npz');oldgroup='phase_b' if item['group']=='phase_d' else 'phase_e';op=npz(OLD_OUT/oldgroup/'B4_SPLINE_50__O1'/item['id']/'prediction.npz');oe=npz(OLD_OUT/oldgroup/'B4_SPLINE_50__O1'/item['id']/'evaluation.npz');I=a['initial_basis'];P=pr['xyz']@I;A=a['anchor_xyz']@I;OC=op['C']@I;NC=p['C']@I;pair=p['pair']@I;oldpair=op['pair']@I;gt=e['gt']@I;s=e['station'];profile=info.get('profiles',[]);worst=int(np.nanargmax(np.where(e['valid'],e['error'][:,1],np.nan))) if e['valid'].any() else len(s)//2;station=float(s[worst]);j=int(np.argmin(abs(p['s']-station)));B=p['B'][j];rel=(pr['xyz']-p['C'][j])@B;inside=abs(rel[:,0])<1
    fig=plt.figure(figsize=(16,11));ax=fig.add_subplot(221,projection='3d');step=max(1,len(P)//4000);ax.scatter(P[::step,0],P[::step,1],P[::step,2],s=1,c=RAW,alpha=.35,label='реальные C4 points');ax.plot(OC[:,0],OC[:,1],OC[:,2],c=OLD,label='старый cubic');ax.plot(NC[:,0],NC[:,1],NC[:,2],c=NEW,label='новая CR');ax.set(xlabel='u, м',ylabel='v, м',zlabel='w, м',title='A · Исходные измерения и CR в 3D');ax.set_box_aspect((4,1,1));ax.legend(fontsize=7)
    ax=fig.add_subplot(222);ax.scatter(rel[inside,1],rel[inside,2],s=8,c=RAW,alpha=.6,label='измерения в срезе');template=__import__('seed_input').canonical()*[m['q'],1];ax.scatter(template[:,0],template[:,1],s=4,c=NEW,alpha=.4,label='эмпирический профиль');ax.scatter(0,0,c=NEW,marker='+',s=100,label='новая reference');oldlocal=(np.array([np.interp(station,op['s'],op['C'][:,k]) for k in range(3)])-p['C'][j])@B;ax.scatter(oldlocal[1],oldlocal[2],c=OLD,marker='x',s=60,label='старый cubic');ellipse(ax,np.zeros(2),B[:,1:].T@c['covariance'][j]@B[:,1:]);ax.set(title=f'B · Профиль около {station:.1f} м; эллипс CR 95%',xlabel='e1, м',ylabel='e2, м');ax.axis('equal');ax.legend(fontsize=7);ax.grid(alpha=.17)
    ax=fig.add_subplot(223);ax.scatter(P[::step,0],P[::step,2],s=2,c=RAW,alpha=.2,label='raw');ax.plot(A[:,0],A[:,2],'o--',ms=3,c='#778',label='C4 anchors');ax.plot(OC[:,0],OC[:,2],c=OLD,label='старый cubic');ax.plot(NC[:,0],NC[:,2],c=NEW,label='новая CR');
    if profile:
        observed=[v for v in profile if v['n']>0]
        if observed:
            anchors=np.array([v['anchor_xyz'] for v in observed])@I;ax.scatter(anchors[:,0],anchors[:,2],s=12,c='#297bab',label='латентные опоры с точками')
    crgt=e['cr_gt']@I;ax.plot(crgt[:,0],crgt[:,2],ls=':',c='#b78a11',lw=1.5,label='future class2 proxy')
    ax.set(title='C · Вертикальная геометрия CR',xlabel='u, м',ylabel='w, м');ax.legend(fontsize=7);ax.grid(alpha=.17)
    ax=fig.add_subplot(224);ax.plot(op['s'],oe['error'][:,1]*100,c=OLD,label='дальний: старый');ax.plot(s,np.where(e['valid'],e['error'][:,1],np.nan)*100,c=NEW,label='дальний: новый');ax.plot(s,np.where(e['valid'],e['error'][:,0],np.nan)*100,c='#2878af',ls=':',label='ближний: новый');ax.axhline(20,c='#a33',ls='--',lw=1);ax.set(title='D · Ошибки ходовых рельсов на общих плоскостях',xlabel='station, м',ylabel='расхождение с GT, см');ax.legend(fontsize=7);ax.grid(alpha=.17)
    fig.suptitle(item['run']+' / '+item['file']+'\n'+item['method']+f' · far p95: {item["old_far_p95"]*100:.1f} → {item["far_p95"]*100:.1f} см',fontsize=13);fig.tight_layout(rect=(0,0,1,.95));figsave(fig,dest/'overview.png')
    fig,axes=plt.subplots(2,2,figsize=(16,9));polish(axes)
    for row,axis in enumerate((1,2)):
        ax=axes[row,0]
        for rail in (0,1):
            ax.plot(oldpair[:,rail,0],oldpair[:,rail,axis],c=OLD,alpha=.7,label='старые рельсы' if rail==0 else None);ax.plot(pair[:,rail,0],pair[:,rail,axis],c=NEW,label='новые рельсы' if rail==0 else None);ax.plot(gt[:,rail,0],gt[:,rail,axis],'.',ms=2,c=GT,label='future class1 proxy' if rail==0 else None)
        ax.set(xlabel='u, м',ylabel=('v' if axis==1 else 'w')+', м',title='Рельсы: план' if row==0 else 'Рельсы: высота');ax.legend(fontsize=8)
    axes[0,1].plot(c['s'],c['curvature_h'],c=NEW);axes[0,1].axhline(.01,c=OLD,ls='--');axes[0,1].axhline(-.01,c=OLD,ls='--');axes[0,1].set(title='Кривизна в плане и проектный предел',xlabel='station, м',ylabel='1/м')
    axes[1,1].plot(c['s'],c['grade'],c=NEW,label='grade');axes[1,1].plot(c['s'],c['curvature_v'],c='#75539b',label='производная grade');axes[1,1].set(title='Вертикальная геометрия; ось регистрации — оценка',xlabel='station, м');axes[1,1].legend(fontsize=8);fig.tight_layout();figsave(fig,dest/'rails_physics.png')
    fig,axes=plt.subplots(2,5,figsize=(20,8));polish(axes)
    for ax,station in zip(axes.ravel(),range(10,101,10)):
        if station>p['s'][-1]+.01:ax.text(.5,.5,'C4 не достиг\nэтой дальности',ha='center',va='center',transform=ax.transAxes);ax.set_title(str(station)+' м');continue
        j=int(np.argmin(abs(p['s']-station)));B=p['B'][j];rel=(pr['xyz']-p['C'][j])@B;inside=abs(rel[:,0])<=1;ax.scatter(rel[inside,1],rel[inside,2],s=5,c=RAW,alpha=.5);ax.scatter(template[:,0],template[:,1],s=3,c=NEW,alpha=.3);ax.scatter(0,0,marker='+',c=NEW,s=70);ellipse(ax,np.zeros(2),B[:,1:].T@c['covariance'][j]@B[:,1:]);ax.set_title(f'{station} м · n={sum(inside)}');ax.set(xlabel='e1, м',ylabel='e2, м');ax.axis('equal')
    fig.suptitle('Реальные C4 points, новая reference и CR uncertainty; отсутствующий горизонт не дорисован',fontsize=12);fig.tight_layout(rect=(0,0,1,.95));figsave(fig,dest/'sections.png')
    body='<a href="../../gallery.html">← Галерея</a> · <a href="../../REPORT_LATENT_CONTACT_REFERENCE.html">Полный отчёт</a><h1>'+esc(item['run']+' / '+item['file'])+'</h1><p>'+esc(item['group']+' · '+item['method'])+'</p><p class="note">'+esc(item['analysis'])+'</p><p>Красный — старый STEP6, зелёный — новый CR и восстановленные рельсы; чёрный — расчётный эталон. Это сравнение с несовершенной разметкой и оценённой регистрацией. Физические флаги вычислены до GT. Отсутствующие сечения явно исключены с учётом coverage.</p>'
    for name,title in [('overview','Измерения, профиль, кривые и ошибки'),('rails_physics','Ходовые рельсы и физическая проверка'),('sections','Срезы 10–100 м')]:body+='<section><h2>'+title+'</h2><a href="'+name+'.png"><img class="plot" src="'+name+'.png" loading="lazy" alt="'+title+'"></a></section>'
    manual=STAGE/'MANUAL_REVIEW_RU.json'
    if manual.exists() and item['id'] in load(manual):body+='<section><h2>Разбор после просмотра</h2><p>'+esc(load(manual)[item['id']])+'</p></section>'
    if (OUT/'las'/item['id']/'EXPORT.json').exists():body+='<p><a href="../../las/'+item['id']+'/EXPORT.json">LAS: состав и provenance</a></p>'
    (dest/'index.html').write_text(page(item['file'],body),encoding='utf-8');return item['id']

def main():
    items=load(OUT/'gallery/cases.json')
    with ProcessPoolExecutor(max_workers=6) as pool:
        for j,v in enumerate(pool.map(figure,items),1):print('GALLERY',j,len(items),v,flush=True)
    body='<h1>STEP 6.1 · Latent contact-rail reference</h1><p><a href="REPORT_LATENT_CONTACT_REFERENCE.html">Полный отчёт</a> · <a href="research_freeze.json">Research freeze</a> · <a href="las_exports.csv">LAS</a></p><p class="note">Включены все 11 старых тяжёлых случаев, новые ошибки >20 см, лучшие, средние и худшие примеры. Фильтры обозначают диагностические признаки, а не автоматически доказанные причины.</p><div class="controls"><select id="filter"><option value="">Все</option>'+''.join('<option value="'+k+'">'+v+'</option>' for k,v in TAGS.items())+'</select><input id="search" placeholder="Запись или кадр"><span id="count"></span></div><div class="cards">'
    for item in items:
        url='gallery/'+item['id']+'/index.html';body+='<article class="card" data-tags="'+esc(' '.join(item['tags']))+'" data-search="'+esc((item['run']+' '+item['file']).lower())+'"><a href="'+url+'"><img src="gallery/'+item['id']+'/overview.png" alt="'+esc(item['file'])+'" loading="lazy"></a><a href="'+url+'">'+esc(item['run']+' / '+item['file'])+'</a><p>Far p95: '+f'{item["old_far_p95"]*100:.1f} → {item["far_p95"]*100:.1f} см'+'</p><p>'+esc(', '.join(TAGS.get(t,t) for t in item['tags']))+'</p></article>'
    (OUT/'gallery.html').write_text(page('STEP6.1 — галерея',body+'</div><script src="gallery/filter.js"></script>'),encoding='utf-8');(OUT/'gallery/filter.js').write_text("'use strict';const f=document.getElementById('filter'),q=document.getElementById('search'),c=document.getElementById('count'),cards=[...document.querySelectorAll('.card')];function update(){let n=0;for(const a of cards){const show=(!f.value||a.dataset.tags.split(' ').includes(f.value))&&a.dataset.search.includes(q.value.toLowerCase());a.hidden=!show;if(show)n++;}c.textContent=n+' примеров';}f.addEventListener('change',update);q.addEventListener('input',update);if(location.hash)f.value=location.hash.slice(1);update();",encoding='utf-8');save(OUT/'gallery/RENDER_COMPLETE.json',dict(time_ns=time.time_ns(),cases=len(items),figures=3*len(items),browser_tested=False))

if __name__=='__main__':main()
