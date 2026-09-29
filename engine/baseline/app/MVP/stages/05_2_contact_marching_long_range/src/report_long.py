"""Full Russian research report and local searchable gallery, from saved results."""
from long_common import *
sys.path.insert(0,str(ROOT/'results_rail2d/_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from select_long import summarize
import html,re
from collections import Counter

STYLE='body{font:16px/1.55 system-ui;color:#183447;background:#edf2f5;margin:0}main{max-width:1240px;margin:auto;padding:28px}h1,h2,h3{line-height:1.25;overflow-wrap:anywhere}h2{margin-top:36px}a{color:#00677f}img{max-width:100%}code{overflow-wrap:anywhere}.table{overflow:auto;margin:18px 0}table{border-collapse:collapse;background:#fff;font-size:14px;width:100%}td,th{border:1px solid #d4e0e6;padding:8px;text-align:left}th{background:#dfeaf0;white-space:nowrap}tr:nth-child(even){background:#f5f8fa}.card{background:white;padding:18px;margin:18px 0;border-radius:10px}.controls{position:sticky;top:0;background:#edf2f5;padding:12px;z-index:2;display:flex;gap:8px;flex-wrap:wrap}input,select,button{padding:9px;font:inherit;border:1px solid #bccbd3;border-radius:5px}summary{cursor:pointer}.tag{background:#dcebed;border-radius:4px;padding:3px 6px;font-size:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:850px){.grid{grid-template-columns:1fr}}'

def f(x,n=2):return 'н/д' if x is None or x=='' else f'{float(x):.{n}f}'
def pct(x):return 'н/д' if x is None else f'{100*float(x):.1f}%'
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def md_html(md):
    def inline(s):
        s=html.escape(s);s=re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',r'<img src="\2" alt="\1">',s);s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)
        s=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',s);return re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
    blocks=[]
    for block in md.strip().split('\n\n'):
        lines=block.splitlines()
        if lines[0].startswith('| '):
            cells=lambda x:x.strip().strip('|').split('|')
            head='<tr>'+''.join('<th>'+inline(c.strip())+'</th>' for c in cells(lines[0]))+'</tr>'
            body=''.join('<tr>'+''.join('<td>'+inline(c.strip())+'</td>' for c in cells(line))+'</tr>' for line in lines[2:]);blocks.append('<div class="table"><table>'+head+body+'</table></div>')
        elif lines[0].startswith('#'):
            n=len(lines[0])-len(lines[0].lstrip('#'));blocks.append(f'<h{n}>'+inline(lines[0][n:].strip())+f'</h{n}>')
        else:blocks.append('<p>'+inline(block).replace('\n','<br>')+'</p>')
    return ''.join(blocks)

def metric_table(rows):
    return table(['Метод','N','Median, м','p90, м','p95, м','Max, м','≥50','≥75','≥100','Wins >4','Losses >4','Starts >20 см'],[[r['variant'],r['n'],f(r['median']),f(r['p90']),f(r['p95']),f(r['max']),pct(r['p50']),pct(r['p75']),pct(r['p100']),r['wins4'],r['losses4'],r['catastrophic_starts']] for r in rows])

def figures(rows,names,obs):
    fig,axs=plt.subplots(1,2,figsize=(14,5.4))
    for name in names:
        rr=[r for r in rows if r['variant']==name and r.get('evaluation_eligible')];values=np.array([r['continuous_reach_5cm'] for r in rr]);x=np.arange(0,121)
        axs[0].plot(x,[np.mean(values>=v) for v in x],label=name,lw=2 if name=='B0' else 1.4)
        vals=np.array([r.get('max_continuous_correct_range_5cm') or 0 for r in rr]);axs[1].plot(x,[np.mean(vals>=v) for v in x],label=name,lw=2 if name=='B0' else 1.4)
    for ax,title in zip(axs,['Исходный exact evaluator STEP5','Дополнительно остановка непрерывности на GAP / TENTATIVE']):
        ax.set(xlabel='Дальность R, м',ylabel='P(correct confirmed reach ≥ R)',xlim=(0,120),ylim=(0,1.02),title=title);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'gallery/survival.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(14,5.4));hs=(1,2,4,8,12,16)
    for h in hs:
        rr=[r for r in obs if r['history']==h];x=np.arange(20,131)
        axs[0].plot(x,[np.mean([r['max_observable_range_5cm']>=v for r in rr]) for v in x],label='T' if h==1 else 'F'+str(h))
    budgets=csv_read(OUT/'point_budget_100m.csv');allowed={(r['run'],r['start_frame']) for r in obs};centers=(60,75,90,100,110,120)
    for h in hs:
        rr=[r for r in budgets if int(r['history'])==h and (r['run'],int(r['start_frame'])) in allowed]
        values=[np.median([int(r['unique_xyz']) for r in rr if int(r['center_m'])==v]) for v in centers]
        axs[1].plot(centers,values,'o-',label='T' if h==1 else 'F'+str(h))
    axs[0].set(xlabel='Дальность реальных возвратов рядом с GT, м',ylabel='Доля стартов с хотя бы одним дальним возвратом',ylim=(0,1),title='Наблюдаемость ≠ обнаружение')
    axs[1].set(xlabel='Центр 4-метрового диапазона, м',ylabel='Медиана уникальных XYZ рядом с GT',yscale='symlog',title='Количество информации на дальности')
    for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'gallery/observability.png',dpi=140);plt.close(fig)

def case_text(e):
    p=OUT/'failures'/e['variant']/(key(e['run'],e['start_frame'])+'.json');packet=load(p) if p.exists() else {};d=packet.get('diagnosis',{})
    parts=[f"Непрерывная дальность по исходному GT: {f(e.get('continuous_reach_5cm'))} м; изменение к B0: {f(e.get('delta_reach'))} м.",
      f"Подтверждённые реальные наблюдения: {f(e.get('max_confirmed_observed_range'))} м; tentative: {f(e.get('max_tentative_range'))} м; поиск: {f(e.get('max_search_hypothesis_range'))} м."]
    if d:parts.append(f"Причина по диагностике: {d.get('category')}. В последних 4 м окна есть {d.get('terminal_gt_near_unique')} уникальных XYZ около GT в доступной истории до F16; ошибка направления относительно GT {f(d.get('terminal_tangent_error_deg'))}°. Остаток перекрытия: {f(None if d.get('terminal_overlap_p90') is None else 100*d['terminal_overlap_p90'])} см.")
    if e.get('confirmed_transverse_segments_gt20cm'):parts.append('Есть подтверждённые блоки с поперечным p95 >20 см. Это флаг для проверки структуры, а не автоматическое исправление GT.')
    if not e.get('evaluation_eligible'):parts.append('Этот старт не входит в основной сопоставимый набор из-за недостаточного эталона; его точки и флаги сохранены.')
    return ' '.join(parts)

def gallery(examples):
    tags=sorted(set(t for e in examples for t in e['tags']));methods=sorted(set(e['variant'] for e in examples));cards=[]
    for e in examples:
        target=OUT/'gallery'/e['id'];complete=load(target/'COMPLETE.json');file=frames(e['run'])[e['start_frame']]['file'];base='gallery/'+e['id']+'/'
        title=e['variant']+' · '+e['run']+' / '+file;links=[f'<a href="{base}terminal.png">Срез</a>',f'<a href="{base}track.png">Весь участок</a>',f'<a href="{e["folder"]}/prediction.json">Шаги и кандидаты</a>']
        if (OUT/'las'/e['id']/'EXPORT.json').exists():links += [f'<a href="las/{e["id"]}/overlay.las">LAS overlay</a>',f'<a href="las/{e["id"]}/prediction_confirmed.las">LAS confirmed</a>',f'<a href="las/{e["id"]}/prediction_tentative.las">LAS tentative</a>']
        if complete.get('animation'):links.append(f'<a href="{complete["animation"]}">Анимация шагов</a>')
        extra=f'<img loading="lazy" src="{base}sections_10m.png" alt="Срезы каждые 10 м">' if 'sections_10m.png' in complete['files'] else ''
        if 'first_failure.png' in complete['files']:
            extra=f'<h3>Первый разрыв корректности / непрерывности по оценке</h3><img loading="lazy" src="{base}first_failure.png" alt="Первый неуспешный участок">'+extra
        if 'seed_audit.png' in complete['files']:
            extra=f'<h3>Общий bootstrap и текущая разметка</h3><img loading="lazy" src="{base}seed_audit.png" alt="Несогласие исходного seed и GT">'+extra
        hay=html.escape((title+' '+' '.join(e['tags'])).lower(),quote=True)
        cards.append(f'<article class="card" id="{e["id"]}" data-search="{hay}" data-method="{html.escape(e["variant"])}" data-tags="{html.escape(" ".join(e["tags"]))}"><h2>{html.escape(title)}</h2><p>'+''.join('<span class="tag">'+html.escape(t)+'</span> ' for t in e['tags'])+f'</p><p>{html.escape(case_text(e))}</p><p>'+ ' · '.join(links)+f'</p><img loading="lazy" src="{base}terminal.png" alt="Одинаковый срез текущего и накопленного облака"><details><summary>Весь участок, вид сверху и сбоку</summary><img loading="lazy" src="{base}track.png" alt="Полный найденный участок">{extra}</details></article>')
    script="""const cards=[...document.querySelectorAll('.card')];function filter(){const q=document.getElementById('search').value.toLowerCase(),m=document.getElementById('method').value,t=document.getElementById('tag').value;let n=0;for(const c of cards){const ok=(!q||c.dataset.search.includes(q))&&(!m||c.dataset.method===m)&&(!t||c.dataset.tags.split(' ').includes(t));c.hidden=!ok;if(ok)n++;}document.getElementById('count').textContent=n+' / '+cards.length;}document.getElementById('search').addEventListener('input',filter);for(const id of ['method','tag'])document.getElementById(id).addEventListener('change',filter);filter();"""
    controls='<div class="controls"><input id="search" placeholder="Поиск записи, кадра, метода"><select id="method"><option value="">Все методы</option>'+''.join('<option>'+html.escape(m)+'</option>' for m in methods)+'</select><select id="tag"><option value="">Все категории</option>'+''.join('<option>'+html.escape(t)+'</option>' for t in tags)+'</select><span id="count"></span></div>'
    page='<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>STEP5.2 — галерея</title><style>'+STYLE+'</style><main><h1>Contact rail: дальнее продолжение</h1><p><a href="report.html">Полный отчёт</a> · <a href="REPORT_LONG_RANGE.md">Markdown</a></p><p>Красный — подтверждено алгоритмом; оранжевый — tentative; зелёный — GT только для оценки. Слева показаны решения исходного B0. Границы среза общие и при необходимости расширены для видимости GT. Данные источников и их классы сохранены.</p>'+controls+''.join(cards)+'</main><script>'+script+'</script></html>'
    (OUT/'gallery.html').write_text(page,encoding='utf-8');(OUT/'audit/gallery_script.js').write_text(script,encoding='utf-8')

def questions(a,mechanisms,final,obs,failure_counts):
    aa={r['variant']:r for r in a};mm={r['variant']:r for r in mechanisms}
    longcases=[r for r in load(OUT/'gallery/examples.json') if (r.get('max_confirmed_observed_range') or 0)>=100]
    valid10=[r for r in longcases if (r.get('max_continuous_correct_range_10cm') or 0)>=100 and not r.get('confirmed_transverse_segments_gt20cm')]
    detail100=' '.join(f"{r['variant']} / {r['run']} / {frames(r['run'])[r['start_frame']]['file']}: strict10={f(r.get('max_continuous_correct_range_10cm'))} м, transverse5={f(r.get('transverse_continuous_reach_5cm'))} м, exact5={f(r.get('continuous_reach_5cm'))} м." for r in valid10)
    def stats(names):
        rr=[aa[n] for n in names if n in aa]
        return ' '.join(f"{r['variant']}: median {f(r['median'])} м, max5cm {f(r['max'])} м, wins/losses>4 м {r['wins4']}/{r['losses4']}, флаги>20 см {r['catastrophic_starts']}/{r['n']}." for r in rr)
    def count(n,k):return mm.get(n,{}).get(k,0)
    qa=[
      ('Помогает ли asymmetric fusion?',stats(['B0','AF2','AF4','AF8'])+' Это сравнение на фиксированной Phase A. Положительный эффект на отдельных стартах не означает улучшения каждого старта.'),
      ('Always-on или fallback?',stats(['AF4','SF24','SF248','SF48'])+' В SF сначала полностью проверяется T. При успешном T история не запрашивается. Спасение разрешено после недостатка/слабости новых наблюдений; жёсткий конфликт не маскируется накоплением.'),
      ('Какая глубина истории полезна?',stats(['AF2','AF4','AF8','AF12','AF16','AF24'])+' Глубина истории не заменяет качество локального направления; отдельная таблица наблюдаемости показывает, добавились ли реальные дальние возвраты.'),
      ('Можно ли отделить старую геометрию от новых past points?', 'Да. В режиме A PCA, транспорт и обе проверки old overlap используют только подтверждённые точки T. В режиме B новые принятые past points участвуют в следующем PCA, но не в old overlap. '+stats(['AF8','AF8_B'])),
      ('Помогает ли tentative state?',stats(['A0','LOOK2','TENTATIVE2'])+f" В LOOK2 было {count('LOOK2','promoted')} повышений статуса; неразрешённые точки остались tentative. CONFIRMED — решение алгоритма, не гарантия GT."),
      ('Можно ли безопасно уменьшить минимум с 3 до 2?',f"В TENTATIVE2 на Phase A реально подтверждены {count('TENTATIVE2','support2_confirmed')} двухпозиционных блока; {count('TENTATIVE2','support2_tentative')} остались tentative. Нужны разные позиции и независимые ring/azimuth samples, затем пространственное подтверждение. "+stats(['TENTATIVE2'])+' Ограниченное число таких событий не доказывает общую безопасность.'),
      ('Когда полезен один point tentative?',f"В диагностическом TENTATIVE1: {count('TENTATIVE1','support1_tentative')} одиночных блоков остались tentative, {count('TENTATIVE1','support1_confirmed')} получили статус позже, через совместное подтверждение несколькими реальными позициями. Немедленное подтверждение одного point запрещено. Этот вариант исключён из выбора финальных конфигураций. "+stats(['TENTATIVE1'])),
      ('Помогает ли track-age prior?',stats(['TRACKAGE','RANGE_RELAX'])+' Возраст измеряется длиной уже подтверждённой собственной цепочки, а не будущей траекторией. Ослабление включается после 24 м и меняется на 40/60/80 м; на молодых цепочках оно не действует.'),
      ('Можно ли ослаблять требования после длинной истории?',stats(['NEW_ONLY_RELAX','TRACKAGE'])+' NEW_ONLY_RELAX оставляет обе проверки старого T overlap на 3 см, ослабляя только поиск новых наблюдений. Его отдельно сравнили с ослаблением общей проверки.'),
      ('Слишком ли строг hard 3cm overlap?',stats(['A0','OVERLAP_3.5','OVERLAP_4','OVERLAP_5','OVERLAP_6'])+' Да, он отсекает часть пригодных продолжений; однако увеличение допуска также может пропустить ошибочное направление. Это проверяется по новым точкам и поперечной ошибке, а не по одному residual.'),
      ('Чем его заменить безопаснее?',stats(['OVERLAP_soft','OVERLAP_q80','OVERLAP_trimmed','OVERLAP_fraction','OVERLAP_regularized'])+' Сравнены ограниченная robust loss, квантили, доля inliers и регуляризация смещения anchor. Hysteresis допускает только явно заданную умеренную область; более сильный конфликт не становится tentative.'),
      ('Помогает ли one-gap?',stats(['GAP1'])+' GAP имеет длину 4 м и не содержит предсказанных точек. Успешный проход отмечается только после последующего совместимого наблюдаемого участка.'),
      ('Помогают ли two-gap?',stats(['GAP2'])+' Допускается максимум 8 м подряд без наблюдений. Неудачный поиск после пропуска не увеличивает confirmed observed reach.'),
      ('Помогает ли two-step spatial confirmation?',stats(['LOOK2','LOOK3'])+' Следующие пространственные участки берутся из того же причинного облака. Временных кадров T+1/T+2 в этом поиске нет.'),
      ('Помогает ли beam search?',stats(['BEAM2','BEAM3','BEAM5','GRAPH'])+' Ветви хранят отдельные frames, anchors, состояния, score и provenance. GRAPH — пространственный track graph с beam=5, 3-step pending и двумя GAP; это не глобально оптимальное решение всех возможных кривых. Дополнительно проверены веса 0.75/1.25 и штраф изменения кривизны.'),
      ('Помогает ли dynamic window?',stats(['DYNAMIC_A4','DYNAMIC_A6','DYNAMIC_A8','ADAPTIVE_W'])+' Изменение окна зависит от собственной дальности plane или слабости support, без GT. Увеличение окна даёт больше точек, но повышает чувствительность к кривизне.'),
      ('Помогает ли piecewise dewarp?',stats(['W16','DEWARP16','W24','DEWARP24'])+' Для проекции используются 4-метровые подокна и краткосрочный prior кривизны. Коррекция применяется только к координатам сопоставления профиля; исходные XYZ не перемещаются.'),
      ('Нужно ли partial-template matching?',stats(['PARTIAL_vertical','PARTIAL_horizontal','PARTIAL_corner','PARTIAL_subset'])+' Исходный matcher уже использует одностороннее расстояние к шаблону: полнота профиля даёт bonus, но не является обязательным gate. Поэтому subset здесь добавляет отложенное подтверждение, а не восстанавливает невидимые части L. Отдельные грани неоднозначны без continuity.'),
      ('Помогает ли current-frame negative evidence?',stats(['AF8','VETO'])+' Отсутствие T returns не запрещает историю. Veto требует хорошо заполненной текущей области, отсутствия поддержки кандидата и сильного расхождения точек с ним. Это эвристика наблюдаемости, не физическая модель окклюзии.'),
      ('Помогает ли registration gating?',stats(['REG2','REG3','REG4','REG5'])+' Исторический кадр оценивается на уже подтверждённом overlap: учитываются residual и смещение относительно T. Без достаточного overlap строгий gate отвергает источник.'),
      ('Помогает ли keyframe selection?',stats(['KF_DISP','KF_UNIFORM','KF_COVERAGE'])+' Проверены заданные смещения 0/.5/1/2/3/4/6/8 м, равномерные смещения и прирост новых canonical bins 1 см. Coverage не использует классы и прекращает добавление при малом приросте.'),
      ('Нужен ли local dv/dw alignment?',stats(['AF8','ALIGN8_1','ALIGN8_2','ALIGN8_3','ALIGN16_1'])+' Это ограниченный сдвиг evidence по old track, без yaw/pitch ICP. Он не исправляет физическую регистрацию всего облака и не меняет LAS XYZ.'),
      ('Лучше ли короткий constant-curvature prior?',stats(['A0','P1','P2','P3'])+' P1/P2/P3 служат только для следующего поиска. Ни spline, ни local polynomial не создают наблюдений. Отдельно проверены robust tube, refit последних anchors и backward reconnect.'),
      ('Где заканчивается физическая наблюдаемость?', 'Она различается по старту. Таблицы observability_range и point_budget_100m содержат T/F2/F4/F8/F12/F16, числа уникальных XYZ, источников, колец и template regions. Максимальный единичный возврат и заполненный интервал — разные величины; 5-см tube также зависит от погрешности GT/регистрации.'),
      ('Есть ли реальные returns на 100 м?', 'Да, у части стартов; точные числители и знаменатели приведены в разделе наблюдаемости. В профилях отдельно окрашены существующий class2 и только геометрически близкие к GT точки. Последние нельзя автоматически считать доказанным CR.'),
      ('В скольких starts fusion наблюдает CR ≥100?', 'См. таблицу по глубине истории: в ней приведено количество стартов с реальными возвратами в 5-см tube, а также чувствительность к 10 см. Условная доля успехов рассчитывается на том же фиксированном сопоставимом наборе.'),
      ('Почему detector не достигает 100, когда observable ≥100?', 'Одного дальнего возврата недостаточно для связанной цепочки. В failure gallery показан первый отказ, ориентация, overlap и ближайшие реальные точки. Основные категории: '+', '.join(f'{k}: {v}' for k,v in failure_counts.most_common(5))+'.'),
      ('Получены ли реальные confirmed 100m tracks?', f'Есть {len(longcases)} development method/start случая с реальными подтверждёнными observations ≥100 м. Из них {len(valid10)} сохраняют непрерывную корректность при допуске 10 см без confirmed >20 см. Итог основного 5-см критерия и финального benchmark показан отдельно. '+detail100),
      ('Какие configs это сделали?', detail100+' Этот C5 не попал в финальную тройку: на development он принимал ошибочную структуру в других стартах. Полный audit всех дальних случаев сохранён в 100m_cases; после просмотра benchmark конфигурация не менялась.'),
      ('Есть ли >100m без catastrophic switch?', 'Да, один development-пример C5 до 119,41 м: confirmed блоков >10/20 см нет, поперечная 5-см метрика принятых окон сохраняется. Исходная 3D 5-см метрика обрывается на 91,76 м; 3D 10-см доходит до 119,41 м. Опора разрежена: один пустой фиксированный 4-метровый диапазон, максимальное расстояние по дальности между наблюдениями 5,39 м. Поэтому это соединение реальных наблюдений через пропуск, не полностью наблюдаемая полоса и не надёжность 100 м между прогонами.'),
      ('Каков следующий bottleneck?', 'Первый отказ и разрыв observable−confirmed приведены для каждого старта. Ключевое различие: слабая sampling плотность, неверный локальный frame и принятие соседней структуры требуют разных решений. Простое увеличение N history или ослабление residual не решает все три причины одновременно.')
    ]
    return '\n\n'.join(f'### {i}. {q}\n\n{answer}' for i,(q,answer) in enumerate(qa,1))

def oracle_sections(rows):
    base={(r['run'],r['start_frame']):r for r in rows if r['variant']=='B0' and r.get('evaluation_eligible')}
    b8={(r['run'],r['start_frame']):r for r in rows if r['variant']=='B8' and r.get('evaluation_eligible')}
    switch=csv_read(OUT/'fusion_switch_oracle.csv');threshold=csv_read(OUT/'fusion/oracle_threshold.csv')
    threshold=[r for r in threshold if (r['run'],int(r['start_frame'])) in base and r.get('continuous_reach_5cm') not in ('',None)]
    def summary(label,values,deltas):
        v=np.array(values);d=np.array(deltas)
        return [label,len(v),f(np.median(v)) if len(v) else 'н/д',f(max(v)) if len(v) else 'н/д',pct(np.mean(v>=75)) if len(v) else 'н/д',pct(np.mean(v>=100)) if len(v) else 'н/д',int(np.sum(d>4)),int(np.sum(d<-4))]
    table_rows=[summary('B0', [r['continuous_reach_5cm'] for r in base.values()],[0]*len(base)),
      summary('Oracle: лучший B0/B2/B4/B8 для целого старта',[float(r['oracle_reach']) for r in switch],[float(r['gain']) for r in switch]),
      summary('Oracle: мягкий 3–6 см overlap + проверка NEW по GT',[float(r['continuous_reach_5cm']) for r in threshold],[float(r['continuous_reach_5cm'])-base[r['run'],int(r['start_frame'])]['continuous_reach_5cm'] for r in threshold])]
    improved=sum(float(r['continuous_reach_5cm'])>b8[r['run'],int(r['start_frame'])]['continuous_reach_5cm']+4 for r in threshold if (r['run'],int(r['start_frame'])) in b8)
    conditions=load(OUT/'fusion/development_conditions.json')['rows'];rr=[r for r in conditions if r['feature']=='support_group' and r['variant'] in ('B8','AF8','SF248')]
    return ['## Offline oracle: запас в данных и потери от gates',
      'Это отдельные диагностические вычисления после inference, использующие GT. Они не участвуют в выборе конфигураций. Первый выбирает лучшее целое предсказание B0/B2/B4/B8. Второй повторяет frozen F8, разрешая только умеренный overlap 3–6 см, если реальные NEW points удовлетворяют будущему GT: минимум три позиции, ≥95% в пределах 5 см. Остальные gates сохранены. Это контролируемая проверка одного препятствия, не абсолютный предел возможного алгоритма.',
      table(['Диагностика','N','Median, м','Max, м','≥75','≥100','Wins >4 vs B0','Losses >4 vs B0'],table_rows),
      f'GT-controlled overlap oracle выигрывает более 4 м у B8 в {improved} стартах. Число разрешённых gates и все случаи: [oracle_threshold.csv](fusion/oracle_threshold.csv). Отдельная [независимая огибающая 4-метровых slabs](fusion/oracle_slab_envelope.csv) оптимистична: совместимость выбранных из разных методов участков не гарантируется.',
      '## Когда накопление помогает: диагностические условия',
      'На фиксированных 36 development-стартах сгруппированы доступные самому T-only детектору признаки перед первым terminal rescue: support и old-overlap. Результат ниже — описание сохранённых решений, а не обученная политика переключения. Пороги не подбирались на финальном benchmark; малые группы не доказывают переносимость. Сравнение полной дальности также включает последующие решения метода.',
      table(['Метод','T support','N','Прогонов','Wins >4','Losses >4','Median Δ, м'],[[r['variant'],r['value'],r['n'],r['runs'],r['wins4'],r['losses4'],f(r['median_delta'])] for r in rr]),
      '[Все условия overlap/support/причина остановки](fusion/development_conditions.csv) · [Данные каждого старта](fusion/development_conditions_samples.csv). Простого универсального causal переключателя эта таблица не доказывает: нехватка текущих точек допускает rescue, но плохо согласованная история способна ухудшить дальнейшую геометрию.']

def feature_sections():
    features=load(OUT/'audit/development_feature_summary.json');vis=csv_read(OUT/'partial_template/development_class2_visibility.csv');vrows=[]
    for lo in (0,20,40,60,80,100):
        rr=[r for r in vis if int(r['history'])==16 and int(r['lo'])==lo and int(r['class2_returns'])>0]
        vrows.append([f'{lo}–{lo+20 if lo<100 else 130}',len(rr),f(np.median([float(r['template_fraction']) for r in rr]) if rr else None,3),sum(str(r['both_faces_visible']).lower()=='true' for r in rr),sum(int(r['class2_returns']) for r in rr)])
    return ['## Признаки LAS и видимость профиля',
      'На development проверены существующие intensity/intensity_raw, ring, return_number и number_of_returns. AUC ниже — вероятность, что значение у class2 выше, чем у соседнего clutter в той же области и диапазоне. AUC≈0,5 означает отсутствие одномерного разделения; AUC<0,5 означает обратное направление. Это диагностика по несовершенным labels, не качество классификатора на независимых данных.',
      table(['Признак','Диапазон от, м','Стартов','Median AUC','Class2 points','Clutter points'],[[r['field'],r['lo'],r['starts'],f(r['median_auc'],3),r['positive_n'],r['negative_n']] for r in features if r['field']!='intensity_raw']),
      'Интенсивность вдали близка к слабому разделению; поля returns в этих данных не дают разделения. Ring заметно связан с положением поверхности, поэтому проверены только малые добавки к score (RING_BEAM05/10). Они не стали финальными победителями. Номер канала отражает также геометрию сканирования и не является универсальным признаком материала рельса.',
      table(['Дальность, м','Стартов с class2 returns','Median доля занятых template nodes','Видны обе грани','Всего class2 records'],vrows),
      'Таблица относится к causal F16 на development и только исходному class2 в 5-см tube эталона. На дальности не требуется автоматически полный L-профиль: разреженные точки одной грани нуждаются в проверке продолжения. Обе грани где-либо в 20-метровом диапазоне не означают полный профиль в каждом отдельном срезе.',
      '[Схема и значения number_of_returns](audit/development_schema.json) · [Условные квантили class2/clutter](audit/development_conditional_distributions.csv) · [AUC](audit/development_feature_summary.csv) · [Видимость только class2](partial_template/development_class2_visibility.csv)']

def main():
    lock=load(OUT/'research_freeze.json');configs=load(OUT/'configs.json');rows=load(OUT/'phase_e_benchmark/analysis_rows.json');names=['B0','B2','B4','B8']+lock['top3']
    data={n:[r for r in rows if r['variant']==n] for n in names};summary=summarize(data);by={r['variant']:r for r in summary};base=by['B0'];best=by[lock['best_development']]
    cohorts={(r['run'],r['start_frame']) for r in data['B0'] if r.get('evaluation_eligible')}
    obs=[r for p in (OUT/'observability').glob('*/ranges.json') for r in load(p) if (r['run'],r['start_frame']) in cohorts]
    ob100={h:{(r['run'],r['start_frame']) for r in obs if r['history']==h and r['max_observable_range_5cm']>=100} for h in (1,2,4,8,12,16)}
    gt100set={(r['run'],r['start_frame']) for r in data['B0'] if (r['run'],r['start_frame']) in cohorts and (r.get('max_GT_available_range') or 0)>=100}
    rr=[r for r in data[best['variant']] if (r['run'],r['start_frame']) in cohorts];cond=[r for r in rr if (r['run'],r['start_frame']) in ob100[16]]
    success100=[r for r in rows if r['variant'] in lock['top3'] and (r.get('max_continuous_correct_range_5cm') or 0)>=100 and not r.get('confirmed_transverse_segments_gt20cm')]
    success120=[r for r in success100 if r['max_continuous_correct_range_5cm']>=120]
    success100_10=[r for r in rows if r['variant'] in lock['top3'] and (r.get('max_continuous_correct_range_10cm') or 0)>=100 and not r.get('confirmed_transverse_segments_gt20cm')]
    failures=csv_read(OUT/'failure_reasons.csv')
    failed100={(r['run'],r['start_frame']) for r in data[best['variant']] if r.get('evaluation_eligible') and (r.get('max_GT_available_range') or 0)>=100 and (r.get('continuous_reach_5cm') or 0)<100}
    fc=Counter(r['category'] for r in failures if r['variant']==best['variant'] and (r['run'],int(r['start_frame'])) in failed100)
    fc_all=Counter(r['category'] for r in failures if r['variant']==best['variant'])
    phasea=load(OUT/'phase_a_screen/summary.json');phaseb=load(OUT/'phase_b_full_dev/summary.json');phased=load(OUT/'phase_d_combinations/summary.json');mechanisms=load(OUT/'audit/ablation_mechanisms.json');runtime=load(OUT/'runtime/summary.json');roi_runtime=load(OUT/'runtime/roi_summary.json')
    examples=load(OUT/'gallery/examples.json');exports=load(OUT/'las/index.json');profiles=load(OUT/'observability/profile_examples.json');protocol=load(OUT/'protocol.json')
    development100=[r for r in examples if (r.get('max_confirmed_observed_range') or 0)>=100 and r.get('group')!='phase_e_benchmark']
    figures(rows,names,obs);gallery(examples)
    flat=[{k:v for k,v in by[n].items() if k!='run_metrics'} for n in names];csv_write(OUT/'final_benchmark.csv',flat)
    md=['# STEP 5.2 — Contact rail: исследование дальнего продолжения',
      f"**BEST DEVELOPMENT CONFIG: {best['variant']}**\nFINAL BENCHMARK CONFIGS: "+', '.join(lock['top3'])+'.',
      '**100M ACHIEVED: '+('YES' if success100 else 'NO')+'**\n**120M ACHIEVED: '+('YES' if success120 else 'NO')+'**\nКритерий: непрерывная корректная подтверждённая дальность ≤5 см, без подтверждённых блоков с поперечным p95 >20 см. GAP не считается наблюдаемым участком.',
      metric_table([base,best]),
      f"P≥100 при GT available ≥100: {pct(best['p100_conditional_gt'])}, знаменатель {best['gt100_n']}.\nP≥100 при наличии causal F16 возвратов ≥100: {pct(np.mean([r['continuous_reach_5cm']>=100 for r in cond]) if cond else None)}, знаменатель {len(cond)}.\nWins >4/8/20 м: {best['wins4']}/{best['wins8']}/{best['wins20']}. Losses >4/8/20 м: {best['losses4']}/{best['losses8']}/{best['losses20']}.\nОсновные оставшиеся причины: "+', '.join(f'{k} ({v})' for k,v in fc.most_common(3))+'.',
      '**reused benchmark, not fresh external validation.** Два прежних тестовых прогона уже анализировались в прошлых исследованиях. Эти результаты служат исследовательской проверкой; production freeze не создавался. Лучший метод выбран на development до нового benchmark, а не переименован по его итогам.',
      '[Галерея лучших, средних и худших случаев](gallery.html) · [Результаты по каждому старту](per_start.csv) · [LAS exports](las_exports.csv) · [Research freeze и SHA](research_freeze.json)',
      '## Что именно проверено',
      f"Отдельный этап: MVP/stages/05_2_contact_marching_long_range. Результаты: results_contact_marching_long_range. Код и результаты STEP5 / STEP5.1 не изменялись. Широкий экран: {len(phasea)} конфигураций × {protocol['screen_n']} стартов. Полная development-проверка: {len(load(OUT/'phase_b_selection.json')['variants'])} вариантов × {protocol['development_n']} стартов. Комбинации и дополнительные score-ablation: {len(configs)-len(phasea)} вариантов × {protocol['development_n']}. Финальная проверка: {len(names)} методов × {protocol['benchmark_n']} стартов.",
      f"Основной сопоставимый benchmark: {base['n']} стартов с пригодным эталоном. Остальные случаи не исчезают: unavailable seed и недостаточный GT остаются в per_start.csv. Это протокол выбранных стартов, а не заявление, что выполнена независимая проверка каждого из 6551 исходных кадров.",
      '## Финальный benchmark',metric_table([by[n] for n in names]),
      table(['Метод']+[f'P≥{h} м' for h in (30,40,50,60,75,80,100,120)],[[n]+[pct(by[n][f'p{h}']) for h in (30,40,50,60,75,80,100,120)] for n in names]),
      '![Survival: исходная и строгая метрики](gallery/survival.png)',
      '## Наблюдения, поиск и непрерывность',
      'В исходном exact evaluator STEP5 используются те же frozen reference и правила: ≥3 оценённых point records, ≥80% охвата эталоном и ≥95% точек в допуске на каждом принятом блоке. Он сохранён без изменений. Дополнительная строгая метрика прерывается также на GAP и неразрешённом TENTATIVE. Поперечная метрика убирает компонент вдоль локального GT tangent; продольная ошибка конца reference хранится отдельно. Дальняя одиночная точка не означает непрерывные 100 м.',
      table(['Метод','Max search, м','Max tentative, м','Max confirmed observed, м','Max continuous exact5, м','Max strict5, м','Max transverse5, м'],[[n]+[f(max((r.get(k) or 0 for r in data[n]),default=0)) for k in ('max_search_hypothesis_range','max_tentative_range','max_confirmed_observed_range','continuous_reach_5cm','max_continuous_correct_range_5cm','transverse_continuous_reach_5cm')] for n in names]),
      '## Безопасность и ошибки структуры',
      'Считаются подтверждённые блоки с достаточным GT, у которых поперечный p95 превышает заданный порог. Это подозрение на ошибочную структуру, а не доказательство безошибочности самой разметки. Таблица включает все старты с оценёнными блоками, в том числе исключённые из основной метрики дальности. Ошибки tentative сохранены отдельно в per_start.csv.',
      table(['Метод','Confirmed >10 см','>20 см','>50 см','>1 м','Tentative >20 см'],[[n]+[sum(r.get(k) or 0 for r in data[n]) for k in ('confirmed_transverse_segments_gt10cm','confirmed_transverse_segments_gt20cm','confirmed_transverse_segments_gt50cm','confirmed_transverse_segments_gt100cm','tentative_transverse_segments_gt20cm')] for n in names]),
      'Старый 3D point-to-surface error может включать несовпадение продольных концов GT. Поэтому он не заменяется новым, а показывается рядом с поперечным. Большой объём правильных near points способен скрыть ошибку нескольких дальних блоков в общем point p95; проверка каждого блока обязательна.',
      '## Есть ли информация на 100 м?',
      'Все значения ниже вычислены после завершения inference. Использован тот же причинный набор T и прошлого, но точки проверены на близость к будущему GT. Это оценка потенциальной наблюдаемости: близость к GT может включать соседние отражения, а ошибка регистрации может увести настоящий CR за 5-см tube. Поэтому сохранена также чувствительность к 10 см и существующий исходный class2.',
      table(['История','Измеренных стартов','Есть ≥100, 5 см','Есть ≥120, 5 см','Есть ≥100, 10 см','Median max возврат, м','Max возврат, м'],[[('T' if h==1 else f'F{h}'),len([r for r in obs if r['history']==h]),len(ob100[h]),sum(r['max_observable_range_5cm']>=120 for r in obs if r['history']==h),sum(r['max_observable_range_10cm']>=100 for r in obs if r['history']==h),f(np.median([r['max_observable_range_5cm'] for r in obs if r['history']==h])),f(max([r['max_observable_range_5cm'] for r in obs if r['history']==h],default=0))] for h in (1,2,4,8,12,16)]),
      '![Наблюдаемость и point budget](gallery/observability.png)',
      '[Полный point budget на 60/75/90/100/110/120 м](point_budget_100m.csv) · [Наблюдаемость по каждому старту](observability_range.csv) · [Какие части профиля видны](partial_template/profile_visibility.csv)',
      'Максимум наблюдаемости ограничен также протяжённостью самого reference. Отсутствие подходящей точки после конца GT не доказывает физическую ненаблюдаемость. Поэтому ниже приведён отдельный знаменатель: только старты с GT≥100 м.',
      table(['История','Старты GT≥100','С реальными возвратами ≥100 в 5см tube','Доля'],[[('T' if h==1 else f'F{h}'),len(gt100set),len(ob100[h]&gt100set),pct(len(ob100[h]&gt100set)/len(gt100set) if gt100set else None)] for h in (1,2,4,8,12,16)]),
      '## Development: отбор и устойчивость между прогонами',
      'Phase A фиксировалась до новых результатов и включала короткие/длинные baseline tracks, повороты и уклоны. Основной отбор использует равный вес прогонов: 2×P50 + 3×P75 + 5×P100 + mean(min(reach,120))/120 − 10×доля стартов с confirmed >20 см − 2×доля потерь >4 м. Эквивалентные наборы результатов не заполняют весь shortlist. Для каждого исключённого прогона заново строится shortlist без его данных; Phase B охватывает объединение этих shortlist.',
      metric_table(phaseb[:15]),
      'Число соседних кадров не является числом независимых испытаний. Часть development-папок происходит из одной исходной записи; leave-one-run-out по папкам — проверка устойчивости между сохранёнными прогонами, но не независимая внешняя валидация маршрутов.',
      table(['Исключённый прогон','Выбран без этого прогона','Train utility','Test N','Test P50','Test P75','Test cat20'],[[r['left_out'],r['selected_without_this_run'],f(r['training_score'],4),(r.get('held_run_metrics') or {}).get('n','н/д'),pct((r.get('held_run_metrics') or {}).get('p50')),pct((r.get('held_run_metrics') or {}).get('p75')),(r.get('held_run_metrics') or {}).get('catastrophic','н/д')] for r in lock['leave_one_run_out']]),
      '[Top8 по каждому прогону](cross_run.csv) · [Полный Phase A](phase_a.csv) · [Полный Phase B](phase_b.csv)',
      '## Комбинации и дополнительные проверки',metric_table(phased),
      'C1: AF4 + tentative + soft overlap. C2: adaptive history + 2-step + one-gap. C3: AF8 + current veto + track-age relaxation. C4: F12 ROI + beam3 + partial subset. C5: adaptive history + dynamic window + beam3 + tentative; принятые past points допускаются в будущий PCA. RING_BEAM05/10 добавляют малый score по относительному номеру канала после 20 м. WEIGHTS075/125 меняют нормированные веса tangent/curvature; CURVATURE_DELTA / _F8 штрафуют изменение кривизны относительно последних подтверждённых anchors.',
      '## Ответы на 31 вопрос',questions(phasea,mechanisms,summary,obs,fc),
      '## Сравнение скорости и памяти',
      'Последовательные свежие процессы, одинаковые 12 стартов для финальных методов. Полный start включает чтение, преобразования, свежий STEP1/STEP2 bootstrap и marching; запуск интерпретатора исключён. Файловый кеш ОС тёплый/неконтролируемый. Cached march — два повторения с готовыми данными/seed. Полный пространственный step (все beam-ветви) измеряется в отдельном cached проходе с лёгкой трассировкой границ цикла; start/cached timers от неё не зависят. Отношение instrumented/cached времени сохранено для контроля накладных расходов. Параллельные исследовательские задания во время этих замеров не выполнялись. Peak RSS включает интерпретатор и проверки совпадения результата.',
      table(['Метод','Start median, мс','Start p95, мс','Full step median, мс','Full step p95, мс','Cached median, мс','Peak RSS max, MiB','Profile/cached'],[[n,f(runtime[n]['start_ms'].get('p50'),1),f(runtime[n]['start_ms'].get('p95'),1),f(runtime[n]['step_ms'].get('p50'),1),f(runtime[n]['step_ms'].get('p95'),1),f(runtime[n]['cached_march_ms'].get('p50'),1),f(runtime[n]['peak_rss_bytes'].get('p100',0)/2**20,1),f(runtime[n]['instrumentation_ratio'].get('p50'),2)] for n in names]),
      'Full step включает накладные расходы инструментирования; коэффициент Profile/cached позволяет видеть их величину. Для практической скорости главным является неинструментированный полный start. Исходные таймеры одного родительского шага выбранного пути также сохранены в [selected_path_steps.csv](runtime/selected_path_steps.csv), но они не включают остальные beam-ветви.',
      'Отдельно whole-cloud F8 и ROI F8/F16 сравнены на одинаковых 12 development-стартах, чтобы не расширять список финальных benchmark-конфигураций после фиксации. ROI ограничивает evidence, подаваемый matcher; прототип хранит кеш отдельных исходных кадров для повторных пространственных запросов. Поэтому узкий ROI не означает автоматически минимальную RAM.',
      table(['Метод','Start median, мс','Start p95, мс','Peak RSS max, MiB','Материализовано точек, median'],[[n,f(roi_runtime[n]['start_ms'].get('p50'),1),f(roi_runtime[n]['start_ms'].get('p95'),1),f(roi_runtime[n]['peak_rss_bytes'].get('p100',0)/2**20,1),f(roi_runtime[n]['points'].get('p50'),0)] for n in ('B8','AF8','AF16')]),
      '[Подробная скорость](runtime.csv) · [Память](memory.csv) · [Whole-cloud / ROI на одинаковых starts](runtime/roi_comparison.csv)',
      '## Ограничения и смысл результата',
      'Параметры являются исследовательскими инженерными гипотезами, а не выученными на benchmark значениями. Все варианты, пороги и диапазоны сохранены в configs.json. Будущие clouds, labels, CR detections и trajectory недоступны источнику inference. Только очищенная pose T+1 используется неизменённым bootstrap для начального forward. История останавливается на разрыве времени/регистрации. Временного сглаживания через такие разрывы нет.',
      'Оценённые регистрации и ручной class2 не являются геодезическим GT. Поперечная метрика уменьшает ошибку продольного края, но не исправляет неверную разметку, неточность регистрации или неверно выбранный связный компонент. Маска собственных отражений поезда не добавлялась: пользователь отложил размеры/экстристику. Никакие исходные LAS или классы не переписаны.',
      'Search prior, spline и GAP существуют только как геометрия поиска. В point outputs и LAS находятся только исходные измерения. state=0 — не выбрано, 1 — tentative, 2 — confirmed; step=65535 — не выбрано. Исходный classification сохранён и отделён от state. confidence/continuity/template scores — эвристики, не вероятности правильности.',
      '## Артефакты и воспроизводимость',
      f"Галерея: {len(examples)} случаев. LAS: {len(exports)} наборов по пять файлов: input_T, fused_evidence, prediction_confirmed, prediction_tentative, overlay. Во всех проверены исходные point fields и новые Extra Bytes после обратного чтения. Полные JSON/NPZ содержат шаги, конкурентов, состояния, actual source frames и (source_frame, source_row, source_point_index). Все confirmed observations ≥75 м включены, даже при плохой метрике; все ≥100 м дополнительно имеют полный audit и срезы каждые 10 м.",
      '[Манифест результатов](MANIFEST.json) · [Итоговая проверка](VERIFICATION.json) · [Стартовые выборки](audit/cohort.json) · [Протокол](protocol.json) · [Конфигурации](configs.json)',
      'Предварительные результаты с обнаруженными ошибками прототипа сохранены в audit и исключены из итоговых таблиц. После исправления порядка source rows контроль A0 совпал с B0 на 36/36 стартах. После исправления hysteresis пересчитаны все затронутые tentative-методы. Замороженные baselines воспроизводятся отдельным неизменённым кодом. Фиксация research_freeze.json сделана до финального benchmark; после неё inference code/config не менялись.',
      '## Полная таблица широкого экрана',metric_table(phasea)]
    bestall=data[best['variant']]
    md.insert(3,f"На финальном benchmark при дополнительном допуске 10 см: непрерывных ≥100 м без confirmed >20 см — {len(success100_10)} method/start случаев. Этот результат показан отдельно от основного 5-см критерия и не меняет его задним числом.")
    md.insert(4,'**Единичный development-пример: C5 достиг 119,41 м при 3D-допуске 10 см и поперечном допуске 5 см по принятым окнам, без подтверждённых блоков >10 см.** По исходной 3D 5-см метрике здесь 91,76 м. Опора разрежена: есть пустой 4-метровый диапазон и интервал 5,39 м между наблюдениями; это не полностью наблюдаемые 119 м. C5 не выбран для финального benchmark из-за ошибок структуры в других development-стартах. Надёжного метода на 100 м между прогонами не получено.')
    md.insert(4,'Confirmed catastrophic blocks (поперечный p95): '+', '.join(f'>{cm} см — {sum(r.get(f"confirmed_transverse_segments_gt{cm}cm") or 0 for r in bestall)}' for cm in (10,20,50,100))+'. Это число блоков по всем стартам с оцениваемыми наблюдениями; число стартов основной выборки с >20 см указано в таблице.')
    at=md.index('## Development: отбор и устойчивость между прогонами');md[at:at]=oracle_sections(rows)+feature_sections()
    at=md.index('## Безопасность и ошибки структуры')
    ten=[]
    for n in names:
        rr10=[r for r in data[n] if (r['run'],r['start_frame']) in cohorts];v=np.array([r.get('continuous_reach_10cm') or 0 for r in rr10])
        ten.append([n,len(v),f(np.median(v)),f(max(v)),pct(np.mean(v>=50)),pct(np.mean(v>=75)),pct(np.mean(v>=100)),pct(np.mean(v>=120)),f(max(r.get('max_continuous_correct_range_10cm') or 0 for r in rr10))])
    md[at:at]=['### Чувствительность к допуску 10 см',table(['Метод','N','Median 10см, м','Max 10см, м','≥50','≥75','≥100','≥120','Max strict 10см, м'],ten),
      'Оба допуска были заданы до финального benchmark. 10 см уменьшает влияние погрешности GT/регистрации, но одновременно допускает более крупное смещение. Изолированные удачные точки и реальные переходы на соседнюю структуру по-прежнему не считаются непрерывным успехом.']
    at=md.index('## Безопасность и ошибки структуры')
    md[at:at]=['### Причины отказов и пределы такой диагностики',
      'На первой странице причины посчитаны только для оцениваемых стартов лучшего development-метода, где GT продолжается ≥100 м, а непрерывная дальность не достигла 100 м. Они определены по terminal-срезу; отдельный first_failure snapshot показывает более ранний разрыв метрики, если алгоритм затем продолжал цепочку.',
      table(['Причина','GT≥100, reach<100','Все старты с доступной диагностикой'],[[k,fc[k],fc_all[k]] for k in sorted(set(fc)|set(fc_all))]),
      'Категории вычислены по наблюдениям causal F16 рядом с GT, а не по скрытому решению детектора. NO_POINTS означает отсутствие геометрически близких возвратов в проверяемой новой области; WEAK_POINTS — менее трёх уникальных XYZ. FRAME использует угол к локальному эталону >1°. Это исследовательское объяснение, чувствительное к GT и регистрации, а не автоматическая истина о физической причине.']
    run_table=[]
    for n in names:
        for run in SPLIT['validation']:
            rr_run=[r for r in data[n] if r['run']==run and (r['run'],r['start_frame']) in cohorts];v=np.array([r.get('continuous_reach_5cm') or 0 for r in rr_run])
            run_table.append([n,run,len(v),f(np.median(v)),f(np.quantile(v,.95)),pct(np.mean(v>=50)),pct(np.mean(v>=75)),sum((r.get('confirmed_transverse_segments_gt20cm') or 0)>0 for r in rr_run)])
    at=md.index('## Наблюдения, поиск и непрерывность');md[at:at]=['### Каждый повторно использованный тестовый прогон отдельно',table(['Метод','Прогон','N','Median, м','p95, м','≥50','≥75','Starts >20см'],run_table)]
    devbest=next(r for r in lock['development_summary'] if r['variant']==best['variant'])
    at=md.index('## Комбинации и дополнительные проверки')
    md[at:at]=['### Проверка инженерных критериев продвижения',
      f"У {best['variant']} на development P≥50={pct(devbest['p50'])}, P≥75={pct(devbest['p75'])}, P≥100={pct(devbest['p100'])}; ≥50 достигнуты в {sum(r['p50']>0 for r in devbest['run_metrics'])}/{len(devbest['run_metrics'])} прогонов, ≥75 — в {sum(r['p75']>0 for r in devbest['run_metrics'])}/{len(devbest['run_metrics'])}. Worst-run P≥50={pct(devbest['worst_run_p50'])}. Потери >4 м: {devbest['losses4']}/{devbest['n']}; confirmed >20 см: {devbest['catastrophic_starts']}/{devbest['n']}. Это перспективный исследовательский вариант для отдельных дальних случаев, но устойчивость 75–100 м между прогонами не доказана."]
    # Add actual distant-profile and representative case links rather than empty headings.
    insert=['## Как выглядят измерения на 80–120 м']
    for r in profiles[:6]:insert.append('!['+r['run']+' / '+frames(r['run'])[r['start_frame']]['file']+']('+r['image']+')')
    md[md.index('## Development: отбор и устойчивость между прогонами'):md.index('## Development: отбор и устойчивость между прогонами')]=insert
    representative=[]
    for tag in ('best_reach','best_gains','median','worst_losses','negative_100m_detail','catastrophic_review'):
        cases=[e for e in examples if tag in e['tags']][:3]
        if cases:
            representative.append('### '+tag)
            for e in cases:representative.append('['+e['variant']+' / '+e['run']+' / '+frames(e['run'])[e['start_frame']]['file']+'](gallery.html#'+e['id']+') — '+case_text(e))
    at=md.index('## Ограничения и смысл результата');md[at:at]=['## Лучшие, средние и худшие случаи']+representative+[
      '### Ручной просмотр development-примеров',
      '[C4, new_data_part_01a2 / frame_001286.las](gallery.html#C4__new_data_part_01a2__000285): корректная по исходному GT цепочка до 78,71 м. В terminal-срезе T содержит только две точки в показанной области; история добавляет точки, но новых устойчивых наблюдений для дальнейшего подтверждения недостаточно. Увеличение числа кадров помогло пройти дальше, не обеспечив 100 м.',
      '[C5, new_data_part_01a1 / frame_000581.las](gallery.html#C5__new_data_part_01a1__000580): реальные подтверждённые точки до 98,11 м, но непрерывная корректность только 61,85 м. На дальнем конце видны принятые точки примерно на 30 см ниже продолжающегося CR-профиля. Это пример, почему максимальная дальность найденных точек не равна успеху.',
      '[C4, new_data_part_01c2 / frame_005856.las](gallery.html#C4__new_data_part_01c2__000855): метрика обрывается уже на 8,09 м при найденных точках до 46,90 м. GT содержит дополнительные вертикальные структуры и заканчивается раньше предсказания. Такой случай требует проверки исходной разметки/регистрации; по одному низкому reach нельзя объявлять всю найденную цепочку ошибочной. GT вручную не исправлялся.',
      '[C4, roundT_pressureGate_roundT / frame_000121.las](gallery.html#C4__roundT_pressureGate_roundT__000120): средний development-пример, 30,33 м по прежней метрике. История заметно уплотняет срез, но вторичная проверка old overlap останавливает дальнейшее продолжение. В срезе также виден разброс будущего эталона относительно локального шаблона.',
      '### Два разных дальних исхода на финальном benchmark',
      '[C4, roundT_squareT_pressureGate_squareT / frame_000036.las](gallery.html#C4__roundT_squareT_pressureGate_squareT__000035): подтверждённые наблюдения до 95,89 м; 5-см метрика прерывается на 43,10 м, а 10-см остаётся корректной до 95,89 м. На общем виде цепочка следует CR, подтверждённых блоков >10 см нет. Причина разрыва строгой метрики — небольшие локальные расхождения около 5–8 см; только картинка не позволяет разделить ошибку детектора и эталона.',
      '[C4, roundT_squareT_pressureGate_squareT / frame_000052.las](gallery.html#C4__roundT_squareT_pressureGate_squareT__000051): после 66,56 м цепочка принимает нижние точки. Три подтверждённых блока имеют поперечный p95 23,8 / 27,3 / 29,4 см. Вид сверху выглядит правдоподобно, а вид сбоку раскрывает ошибку высоты. Это реальная причина сохранять оба вида и проверять дальние блоки отдельно от общего point p95.',
      '[squareT_platform_squareT_switch / frame_000831.las](gallery.html#C4__squareT_platform_squareT_switch__000830): дополнительный флаг >1 м в общей safety-таблице относится к исходному seed, общему для всех методов. Текущая разметка и bootstrap расходятся по стороне; seed p95≈3,40 м в 3D, поперечный p95≈2,82 м. Future reference не построен: component_points=0, seed_gap=3,42 м. По frozen правилам старт не входит в 783 оцениваемых, но сохранён в общей safety-таблице и отдельном рисунке обоих бортов. Это не новое доказательство ошибки C4 marching; требуется ручная проверка bootstrap и разметки. GT не исправлялся.']
    at=md.index('## Артефакты и воспроизводимость');md[at:at]=[
      '### Следующая проверяемая гипотеза',
      'Наибольший смысл имеет совместная оценка позиции профиля и её неопределённости по последовательности пространственных участков: отдельно проверять горизонтальное и вертикальное продолжение и сохранять конкурирующие интерпретации одной видимой грани. Ошибка frame_000052 показывает, что гладкого вида сверху недостаточно. Потребуется штраф за несовместимое изменение высоты профиля, проверенный на development с явным контролем потерь, а не только увеличение history.',
      'Вторая задача — разделить источник неопределённости: неполный профиль, ошибку совмещения прошлого кадра и неверный локальный tangent. Допуск supporting points должен зависеть от измеренной согласованности источников и давности подтверждённой геометрии. Простое увеличение всех допусков не рекомендуется по результатам агрессивных вариантов. Эти предложения не внесены в финальные конфигурации и требуют следующего отдельного исследования.',
      '### Границы реализованных исследовательских вариантов',
      'DEWARP — упрощённая локальная компенсация: для каждого 4-метрового подокна вычитается поперечное смещение второго порядка по recent curvature. Полный поворот каждого подокна в отдельный Bishop frame не реализован, поэтому отрицательный результат этого варианта не исключает пользу более точного dewarp. BACKWARD — проверка обратного геометрического соединения по двум дальним anchors, а не отдельный повторный обратный detector. TUBE использует robust PCA и локальный fit, без глобальной оптимизации всей кривой.',
      'SHAPE 2–6 см меняет gate среднего residual кандидата. Само назначение supporting observations во всех вариантах сохраняет исходную полосу 2 см от fitted template. Поэтому это не полный sweep ширины support tube: увеличение shape gate не обязано добавлять дальние точки. Контролируемая модель неопределённости support с дальностью остаётся отдельной задачей; простое расширение полосы способно добавить нижние и соседние структуры.',
      'В вариантах с advance 6/8 м поля window_start_s и s_from_seed сохраняют условную координату legacy шага 4×k; это не точная длина дуги. Дальности и метрики рассчитаны по реальным XYZ, а положение каждого среза задано plane_origin/basis. Финальные три метода используют advance=4 м. Отдельное отсутствие надёжных 100 м у проверенных вариантов не является доказательством физической невозможности задачи.']
    at=md.index('## Комбинации и дополнительные проверки')
    md[at:at]=['## Все реальные наблюдаемые цепочки ≥100 м на development',
      table(['Метод / прогон / кадр','Наблюдения, м','Exact5, м','Strict10, м','Transverse5, м','GT available, м','Блоки >20 см'],[[f"[{r['variant']} / {r['run']} / {frames(r['run'])[r['start_frame']]['file']}](gallery.html#{r['id']})",f(r.get('max_confirmed_observed_range')),f(r.get('continuous_reach_5cm')),f(r.get('max_continuous_correct_range_10cm')),f(r.get('transverse_continuous_reach_5cm')),f(r.get('max_GT_available_range')),r.get('confirmed_transverse_segments_gt20cm')] for r in development100]),
      'C5 / roundT_doubleT / frame_000023.las — отдельный случай >100 м, удовлетворяющий метрикам принятых окон при допуске 10 см и поперечных 5 см. В doubleT_platform / frame_000064.las после 91,68 м появляются расхождения около 11–12 см; это не удовлетворяет 10-см непрерывности. В roundT_pressureGate_roundT / frame_000073.las дальняя цепочка уходит на другую структуру, до ~78 см поперечной ошибки. Все три сохранены полностью, включая срезы каждые 10 м и LAS.',
      'Длинное adaptive окно может содержать внутренний пропуск, не представленный отдельным состоянием GAP. Дополнительный [аудит реальных интервалов между наблюдениями](100m_cases/sampling_summary.json) показывает для этих трёх случаев максимальные расстояния по дальности 5,18 / 5,39 / 14,18 м. У 119-метрового случая один пустой фиксированный 4-метровый диапазон и три диапазона с менее чем тремя уникальными XYZ. Поэтому старые window-level метрики не следует трактовать как доказательство плотного непрерывного наблюдения каждого метра. Это ограничение C5 сохранено в отчёте, метрики задним числом не заменялись.',
      'Среди 76 оцениваемых development-стартов C5 имеет 12 стартов с confirmed >20 см. Поэтому один 119-метровый успех не послужил основанием продвинуть его в финальный benchmark или production. Этот отрицательный результат устойчивости столь же важен, как дальний удачный пример.']
    text='\n\n'.join(md)+'\n';(OUT/'REPORT_LONG_RANGE.md').write_text(text,encoding='utf-8')
    (OUT/'report.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>STEP5.2 — полный отчёт</title><style>'+STYLE+'</style><main>'+md_html(text)+'</main></html>',encoding='utf-8')
    save(OUT/'REPORT_FACTS.json',dict(best_development=best['variant'],final=flat,success100=[dict(run=r['run'],i=r['start_frame'],variant=r['variant'],reach=r['max_continuous_correct_range_5cm']) for r in success100],success100_10cm=[dict(run=r['run'],i=r['start_frame'],variant=r['variant'],reach=r['max_continuous_correct_range_10cm']) for r in success100_10],success120=len(success120),cohort_n=base['n'],gt100_n=best['gt100_n'],observable100_F16_n=len(ob100[16]),examples=len(examples),las_sets=len(exports),failure_counts=fc,
      success_scope='Final reused benchmark; development exceptions are listed separately.',
      development_observed_100m=[dict(id=r['id'],observed_range=r.get('max_confirmed_observed_range'),exact5=r.get('continuous_reach_5cm'),strict10=r.get('max_continuous_correct_range_10cm'),transverse5=r.get('transverse_continuous_reach_5cm'),confirmed_blocks_gt20cm=r.get('confirmed_transverse_segments_gt20cm')) for r in development100],
      development_sampling_audit=load(OUT/'100m_cases/sampling_summary.json')))
    print('REPORT',best['variant'],'100m',bool(success100),'120m',bool(success120),len(examples),'examples')

if __name__=='__main__':main()
