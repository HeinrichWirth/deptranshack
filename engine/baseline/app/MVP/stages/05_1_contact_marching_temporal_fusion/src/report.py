"""Russian scientific report, overview figures and local searchable gallery."""
from fusion_common import *
sys.path.insert(0,str(ROOT/'results_rail2d/_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import html
import re

REPORT_STYLE='body{font:16px/1.55 system-ui;color:#183447;background:#edf2f5;margin:0}main{max-width:1240px;margin:auto;padding:28px}h1,h2,h3{line-height:1.25;overflow-wrap:anywhere}h2{margin-top:36px}a{color:#00677f}img{max-width:100%}code{overflow-wrap:anywhere}.table{overflow:auto;margin:18px 0}table{border-collapse:collapse;background:#fff;font-size:14px;width:100%}td,th{border:1px solid #d4e0e6;padding:8px;text-align:left}th{background:#dfeaf0;white-space:nowrap}tr:nth-child(even){background:#f5f8fa}'

def report_html(md):
    """Render only the controlled Markdown subset emitted by this report."""
    def inline(s):
        s=html.escape(s)
        s=re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',r'<img src="\2" alt="\1">',s)
        s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s)
        s=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',s)
        return re.sub(r'`([^`]+)`',r'<code>\1</code>',s)
    blocks=[]
    for block in md.strip().split('\n\n'):
        lines=block.splitlines()
        if lines[0].startswith('| '):
            cells=lambda row:row.strip().strip('|').split('|')
            head='<tr>'+''.join('<th>'+inline(x.strip())+'</th>' for x in cells(lines[0]))+'</tr>'
            body=''.join('<tr>'+''.join('<td>'+inline(x.strip())+'</td>' for x in cells(line))+'</tr>' for line in lines[2:])
            blocks.append('<div class="table"><table>'+head+body+'</table></div>')
        elif lines[0].startswith('#'):
            h=len(lines[0])-len(lines[0].lstrip('#'));blocks.append(f'<h{h}>'+inline(lines[0][h:].strip())+f'</h{h}>')
        else:blocks.append('<p>'+inline(block).replace('\n','<br>')+'</p>')
    return ''.join(blocks)

def f(x,n=2):return 'н/д' if x is None or x=='' else f'{float(x):.{n}f}'
def pct(x):return 'н/д' if x is None else f'{100*float(x):.1f}%'
def cm(x):return f(None if x is None else float(x)*100)
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join(['---']*len(h))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def median(rows,key):return percentiles([float(r[key]) for r in rows if r.get(key) not in ('',None)]).get('p50')

def figures(s,rows):
    names=[v['variant'] for v in s['variants'] if v['variant'] not in ('FUSED_BOOTSTRAP','M2_F0','M2_BEST')]
    fig,ax=plt.subplots(1,2,figsize=(13,5));data=[]
    for name in names:
        rr=[r for r in rows if r['variant']==name and r.get('evaluation_eligible')];x=np.sort([r['continuous_reach_5cm'] for r in rr]);ax[0].step(x,1-np.arange(len(x))/len(x),where='post',label=name,lw=2 if name in ('F0',s['selected']) else 1)
        if name!='F0':data.append((name,[r.get('delta_reach',0) for r in rr]))
    ax[0].set(xlabel='Непрерывная дальность ≤5 см относительно GT, м',ylabel='Доля стартов, reach ≥ R',ylim=(0,1.02));ax[0].legend(ncol=2,fontsize=9);ax[0].grid(alpha=.2)
    ax[1].boxplot([v for _,v in data],tick_labels=[n for n,_ in data],showfliers=True);ax[1].axhline(0,c='#555',lw=1);ax[1].axhline(4,c='#379980',ls='--');ax[1].axhline(-4,c='#c45d51',ls='--');ax[1].set(ylabel='Парная разница дальности, м');ax[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'gallery/reach_comparison.png',dpi=135);plt.close(fig)
    blur=csv_read(OUT/'registration_blur.csv');density=csv_read(OUT/'fusion_density.csv');fig,ax=plt.subplots(1,2,figsize=(13,5))
    for name in ('F0','F2','F4','F8'):
        dd=[r for r in density if r['variant']==name and int(r.get('reference_points_in_bin',0))>=3];bins=sorted(set(float(r['lo']) for r in dd));m=[median([r for r in dd if float(r['lo'])==b],'unique_xyz') for b in bins];ax[1].plot(bins,m,'o-',label=name)
    rr=[r for r in blur if r['variant']=='F8' and int(r['age_frames'])>=0];ages=sorted(set(int(r['age_frames']) for r in rr))
    for field,label in [('residual_median','Median'),('residual_p90','p90'),('residual_p95','p95 (включает clutter)')]:
        med=[median([r for r in rr if int(r['age_frames'])==a],field) for a in ages];ax[0].plot(ages,np.array(med)*100,'o-',label=label)
    ax[0].set(xlabel='Возраст кадра в F8',ylabel='Медиана per-start остатка, см',ylim=(0,9),title='Один подтверждённый overlap T · широкий ROI');ax[1].set(xlabel='Начало диапазона, м',ylabel='Медиана уникальных XYZ около future GT',yscale='symlog',title='Плотность реальных наблюдений')
    for a in ax:a.legend();a.grid(alpha=.2)
    fig.tight_layout();fig.savefig(OUT/'gallery/blur_density.png',dpi=135);plt.close(fig)

def case_text(r,packet):
    delta=r.get('delta_reach',0);last=packet['failed_step'];parts=[]
    if not r.get('evaluation_eligible'):parts.append('Старт исключён из сопоставимой оценки дальности из-за недостаточного reference. Его подозрительные принятые точки сохранены отдельно и не исчезают из галереи из-за исключения из основной метрики.')
    elif delta>4:parts.append('Непрерывная подтверждённая дальность выросла на '+f(delta)+' м.')
    elif delta< -4:parts.append('Накопление ухудшило непрерывную дальность на '+f(-delta)+' м.')
    else:parts.append('Изменение находится в пределах ±4 м: '+f(delta)+' м.')
    if r.get('true_range_extension'):parts.append('За концом всех принятых baseline-точек есть '+str(r.get('confirmed_past_extension_points',0))+' прошлых измерений, подтверждённых future GT: выполнен строгий критерий реального увеличения дальности.')
    elif delta>4:parts.append('Улучшение continuous reach не означает доказанное расширение за весь принятый baseline-участок; критерий historical support за его концом не выполнен.')
    parts.append('В следующей половине baseline-среза: T '+str(r.get('baseline_terminal_returns'))+' возвратов около GT, накопление '+str(r.get('baseline_terminal_fused_returns'))+'; уникальных XYZ '+str(r.get('baseline_terminal_fused_unique'))+'.')
    if r['stop_reason']=='OVERLAP_INCONSISTENT':parts.append('Остановка связана с согласованием overlap; записанный p90 первой подгонки '+cm(last.get('overlap_residual_p90'))+' см. Даже если он меньше 3 см, последующая новая подгонка также обязана объяснять старый overlap.')
    detail=OUT/'diagnostics/overlap_detail'/r['variant']/(key(r['run'],r['start_frame'])+'.json')
    if detail.exists():
        d=load(detail)
        if d.get('second_fit_p90') is not None:parts.append('Восстановленный p90 старого overlap после переноса шаблона в новый anchor — '+cm(d['second_fit_p90'])+' см при пороге 3 см; новых уникальных кандидатов '+str(d.get('new_support_unique'))+'. Это количественная причина отказа, даже при высокой плотности.')
    if r.get('terminal_evaluation_reason')=='ORIENTATION_FAILURE':parts.append('Контрфактический GT frame даёт '+f(r.get('oracle_frame_reach'))+' м; замена направления помогает, но не доказывает физически идеальный frame.')
    if r.get('wrong_structure_review'):parts.append('Первичный большой error связан с недостаточным охватом GT у конца reference; это не подтверждённый переход на другой объект.')
    elif r.get('wrong_structure_suspected'):parts.append('Есть оценённые новые блоки с p95 >20 см. Случай оставлен как подозрение и показан с происхождением точек; несовершенная разметка не позволяет автоматически объявить его доказанной ошибкой структуры.')
    if r.get('past_only_terminal_roi'):parts.append('В terminal ROI '+str(r['past_only_terminal_roi'])+' past-точек дальше 3 см от текущих возвратов. Это могут быть новые углы наблюдения, разреженность, registration ghost или подвижная структура; по одному этому признаку причина не устанавливается.')
    reviewed=OUT/'audit/visual_review.json'
    if reviewed.exists():
        note=load(reviewed).get(r['variant']+'__'+key(r['run'],r['start_frame']))
        if note:parts.append('Визуальный разбор: '+note)
    return ' '.join(parts)

def main():
    s=load(OUT/'SUMMARY.json');rows=load(OUT/'analysis_rows.json');lock=load(OUT/'research_lock.json');configs=load(OUT/'configs.json');bench=load(OUT/'runtime/summary.json');examples=load(OUT/'gallery/examples.json');exports=load(OUT/'las/index.json');v={r['variant']:r for r in s['variants']};b=v['F0'];best=v[s['selected']]
    figures(s,rows);dev=csv_read(OUT/'method_ablation.csv');rescue=csv_read(OUT/'failure_rescue.csv');quality=csv_read(OUT/'common_range_quality.csv');blur=csv_read(OUT/'registration_blur.csv');sources=csv_read(OUT/'source_contribution.csv');ranges=csv_read(OUT/'range_bins_comparison.csv');orient=csv_read(OUT/'orientation_comparison.csv');sparse=csv_read(OUT/'sparsity_rescue.csv')
    selected=s['selected'];db=best['reach']['p50']-b['reach']['p50'];p40=best['survival']['40']-b['survival']['40'];old_sparse=sum(b['stops'].get(x,0) for x in ('TOO_FEW_SUPPORT','NO_NEW_POINTS'));new_sparse=sum(best['stops'].get(x,0) for x in ('TOO_FEW_SUPPORT','NO_NEW_POINTS'));reduction=1-new_sparse/old_sparse
    qq=[r for r in quality if r['variant']==selected and r.get('baseline_matched5') and r.get('fusion_matched5')];quality_delta=float(np.median([float(r['fusion_matched5'])-float(r['baseline_matched5']) for r in qq])) if qq else None
    helpful=(db>=3 or p40>=.05 or reduction>=.20) and best['wrong']<=b['wrong'] and (quality_delta is None or quality_delta>=-.01)
    verdict='HELPFUL' if helpful else 'HARMFUL' if db<=-3 or best['wrong']>b['wrong']+2 else 'MARGINAL'
    empirical=max([x for x in s['variants'] if x['variant'] not in ('F0','FUSED_BOOTSTRAP','M2_F0','M2_BEST')],key=lambda x:x['reach']['p50'])
    lines=[];add=lines.append
    add('# STEP 5.1 — Causal temporal point cloud fusion')
    add(f'**VERDICT: {verdict}.** Основной кандидат, выбранный до held-out: **{selected}** — история в пределах 1 м displacement, RAW_CONCAT. Это исследование, без production freeze.')
    add(table(['Показатель','BASELINE T-only','BEST CAUSAL FUSION — development selection'],[
        ['Вариант','F0 = F1 = exact STEP5 M1',selected],['Median continuous reach ≤5 см, м',f(b['reach']['p50']),f(best['reach']['p50'])+f' (Δ {f(db)})'],['p90, м',f(b['reach']['p90']),f(best['reach']['p90'])+f' (Δ {f(best["reach"]["p90"]-b["reach"]["p90"])})'],
        *[[f'P(reach ≥{d} м)',pct(b['survival'][str(d)]),pct(best['survival'][str(d)])] for d in (30,40,50)],
        *[[reason,b['stops'].get(reason,0),best['stops'].get(reason,0)] for reason in ('TOO_FEW_SUPPORT','NO_NEW_POINTS','OVERLAP_INCONSISTENT')],
        ['ORIENTATION_FAILURE',b['diagnoses'].get('ORIENTATION_FAILURE',0),best['diagnoses'].get('ORIENTATION_FAILURE',0)],['Wrong-structure suspicion after GT coverage review',b['wrong'],best['wrong']],
        ['Последовательный полный старт, median / p95, мс',f(bench['F0']['uncached_start_ms']['p50'])+' / '+f(bench['F0']['uncached_start_ms']['p95']),f(bench[selected]['uncached_start_ms']['p50'])+' / '+f(bench[selected]['uncached_start_ms']['p95'])]]))
    add(f'Лидер контрольной таблицы по медиане — **{empirical["variant"]}: {f(empirical["reach"]["p50"])} м**, p90 {f(empirical["reach"]["p90"])} м. Это описательное ранжирование уже увиденных контрольных результатов. Основной вариант не переназначался по held-out; все варианты показаны ниже.')
    effects=[]
    for x in s['variants']:
        if x['variant'] in ('F0','FUSED_BOOTSTRAP','M2_F0','M2_BEST'):continue
        rr=[r for r in quality if r['variant']==x['variant'] and r.get('baseline_matched5') and r.get('fusion_matched5')]
        dq=float(np.median([float(r['fusion_matched5'])-float(r['baseline_matched5']) for r in rr])) if rr else None
        dm=x['reach']['p50']-b['reach']['p50'];dp=x['survival']['40']-b['survival']['40'];dr=1-sum(x['stops'].get(k,0) for k in ('TOO_FEW_SUPPORT','NO_NEW_POINTS'))/old_sparse
        good=(dm>=3 or dp>=.05 or dr>=.2) and x['wrong']<=b['wrong'] and (dq is None or dq>=-.01)
        label='HELPFUL' if good else 'HARMFUL' if dm<=-3 or x['wrong']>b['wrong']+2 else 'MARGINAL'
        effects.append([x['variant'],f(dm),f(dp*100)+' п.п.',pct(dr),pct(dq) if dq is not None else 'н/д',x['wrong'],label])
    add(table(['Variant','Δmedian, м','ΔP≥40','Снижение sparse STOP','Median Δ matched на общем участке','Wrong suspicion','Описание эффекта'],effects))
    add('Вывод о temporal fusion в целом следует читать по всей таблице. Небольшая медианная прибавка может сочетаться с заметным ростом доли дальних успешных стартов и уменьшением sparse STOP. Консервативный выбор H1 до held-out ограничивает объём истории и не обязан быть лидером контрольного набора. Превращать описательного лидера в новую замороженную production-конфигурацию по этим же данным нельзя считать независимым выбором.')
    add('Критерий практической пользы: median +3 м, либо заметное увеличение P≥40 м (для сводного verdict используется +5 процентных пунктов), либо уменьшение TOO_FEW_SUPPORT+NO_NEW_POINTS на 20%. Дополнительно проверяются подозрения на другую структуру и качество на общем участке; разница менее 1 процентного пункта в медианной доле ≤5 см не считается доказанным ухудшением. Это описательный вывод, а не порог настройки detector. Все точные изменения опубликованы, чтобы не зависеть от одного слова verdict.')
    add('## 1. Данные, изоляция и координаты')
    add(f'Источник: `{dataset()}`. Те же 9 development и 2 held-out заезда, что в STEP5. Development расширен с предварительных 36 до всех 108 исходных development-стартов; 76 имеют пригодный дальний GT. Held-out: 1 422 старта. Основной current-only bootstrap доступен в 869 случаях; основной сопоставимый GT — 783. Недоступный seed не считается ошибкой marching.')
    add('M1 неизменён: W=8 м, advance=4 м, recent tail=8 м, gate ±3 см, overlap p90 ≤3 см, local PCA и Bishop. Bootstrap STEP1+FINAL STEP2 получает только T. После него меняется входное облако, но не gates или template. FRAME_BALANCED — отдельный development challenger с изменёнными весами PCA/template statistics; геометрические acceptance thresholds остаются прежними.')
    add('LAS уже зарегистрированы в общей системе заезда: `p_sensorT = (p_map−t_T) @ R_T`. Pose_k не применяется к координатам повторно. Его inverse используется лишь для source-distance и source-azimuth. Это сверено с docs/CLASSIFIED_LAS_FRAMES.md и tools/export_registered_frames.py и проверено на реальном файле.')
    add('Causal loader оставляет из общего JSON только past…T и отдельную санитизированную pose T+1 для bootstrap direction. Чтение LAS с индексом >T отвергается до открытия файла. Ни labels, ни будущие detections/trajectory не входят в marching API. Future reference открывается только после завершённого prediction и проверки его SHA. Тест с изменением class labels сохраняет геометрию побитно.')
    add('Сохранены source_frame, source row, original point_index, возраст по кадрам и секундам, displacement возраста, расстояния до сенсора source и T, ring и azimuth. В raw fusion ни centroid, ни синтетические точки не добавляются. VOXEL_UNIQUE выбирает реальную наиболее свежую запись ячейки; весь исходный seed сохраняется отдельно от дедупликации.')
    add('## 2. История и development selection')
    add('F2/F3/F4 означают последние 2/3/4 кадра, включая T. H0.5/H1/H2/H4 идут назад до первого превышения Euclidean displacement 0.5/1/2/4 м от T. Это displacement, не интеграл пройденного пути. При начале заезда окно сокращается без дублирования T; фактическое число источников записано. Разрывы времени, недостоверная или future-based pose прекращают историю.')
    add('На всех индексах около остановки history до 4 м может включать 404 кадра. Однако среди стартов с разрешённым bootstrap максимум H0.5/H1/H2/H4 = 2/3/5/9 кадров. Почти неподвижные старты в основном исключены прежним требованием ≥50 см только между T и T+1. Поэтому исследование не доказывает пользу накопления для работы на остановке.')
    add(table(['Development variant','N','Median, м','Score','Wins >4 м','Losses >4 м','Mean frames'],[[r['variant'],r['n'],f(r['median']),f(r['score']),r['wins'],r['losses'],f(r['mean_history'])] for r in dev]))
    add('Score — среднее reach с одинаковым весом заездов минус 100×доля исходных триггеров другой структуры. Из вариантов в пределах 5% от лучшего score выбран меньший фактический объём истории; при равенстве предпочтение RAW_CONCAT. Поэтому H1 выбран консервативно, хотя H2/F2 дали более высокий score. Recency weighting не введён в основной поиск. F6/F8 допущены только после положительного среднего парного Δ и большего числа wins, чем losses у F4 на development; это слабое положительное свидетельство, не сильный выигрыш F4.')
    add('F6/F8 с самого начала считались дополнительными diagnostic variants и не входили в основной selection pool. Поэтому их более высокие development scores в общей таблице не использовались для переопределения H1. Формулировка «лучший score» правила выбора относится к основному pool; alignment и fused-bootstrap challengers также показаны отдельно от основного выбора.')
    add('Проверены voxel 5/10/20 мм; FRAME_BALANCED даёт равный суммарный вес каждому источнику в локальных PCA/translation-fit statistics, сохраняя реальные уникальные support counts и прежний overlap gate. Exponential recency использует displacement tau 0.5/1/2 м. Tiny alignment ±2/5 см использует только уже подтверждённый seed overlap 4–8 м, без class2; это development diagnostic, в основной held-out input не включено. Маска корпуса отложена по прямому ответу пользователя; физические extrinsics отсутствуют, условная маска не выдумывалась.')
    add('## 3. Полная контрольная таблица')
    add(table(['Variant','Seed / starts','N GT','p10','p25','median','p75','p90','p95','max','Median anchor p95, см'],[[x['variant'],f'{x["available"]}/{x["starts"]}',x['evaluable'],*[f(x['reach'].get('p'+str(p))) for p in (10,25,50,75,90,95,100)],cm(x['anchor_p95'].get('p50'))] for x in s['variants']]))
    add(table(['Variant','P≥20','P≥30','P≥40','P≥50','P≥60','Median Δ','p10 / p90 Δ','Wins / ties / losses'],[[x['variant'],*[pct(x['survival'][str(d)]) for d in (20,30,40,50,60)],f(x['paired_delta'].get('p50')),f(x['paired_delta'].get('p10'))+' / '+f(x['paired_delta'].get('p90')),f'{x["wins"]}/{x["ties"]}/{x["losses"]}'] for x in s['variants']]))
    add(table(['Variant','Улучшение >4 м','Ухудшение >4 м'],[[x['variant'],pct(x['wins']/x['evaluable']) if x['evaluable'] else 'н/д',pct(x['losses']/x['evaluable']) if x['evaluable'] else 'н/д'] for x in s['variants']]))
    perrun=csv_read(OUT/'per_run_summary.csv')
    add(table(['Variant','Held-out run','N','Median / p90, м'],[[r['variant'],r['run'],r['n'],f(r.get('p50'))+' / '+f(r.get('p90'))] for r in perrun if r['variant'] in ('F0',selected,'F2','F4','H2','F8')]))
    add('![Дальности и парные изменения](gallery/reach_comparison.png)')
    add('Основные метрики используют EXACT STEP5 evaluator и reference: seed сравнивается с class2 T, новые шаги — только с future T+2…; прежние bins и continuous rules неизменны. Для непрерывности нужны ≥3 оценённых точки, ≥80% support с доступным GT и ≥95% точек в допуске; первый плохой или неоцениваемый блок заканчивает reach. F0 массивы и решения шагов воспроизведены побитно, метрики — точно; времена не обязаны совпадать.')
    add('**FUSED_BOOTSTRAP — отдельная диагностика.** Его seed содержит past-точки, тогда как неизменный STEP5 near-reference содержит только class2 T. Разреженность текущей разметки может обнулить continuous reach такого seed. Эти значения сохранены для прозрачности, но не являются честным доказательством ошибки fused bootstrap. Для основного сравнения bootstrap одинаков; отдельный вывод этого опыта — доступность seed. При фактической истории из одного кадра fresh bootstrap сравнивается с baseline отдельно.')
    add('## 4. Какие отказы удалось продолжить')
    reasons=['OVERLAP_INCONSISTENT','TOO_FEW_SUPPORT','NO_NEW_POINTS','ORIENTATION_JUMP','AMBIGUOUS_CANDIDATE','LOCAL_FRAME_DEGENERATE']
    add(table(['Variant',*reasons],[[x['variant'],*[x['stops'].get(r,0) for r in reasons]] for x in s['variants']]))
    diagnoses=sorted(set(k for x in s['variants'] for k in x['diagnoses']))
    add(table(['Variant',*diagnoses],[[x['variant'],*[x['diagnoses'].get(k,0) for k in diagnoses]] for x in s['variants']]))
    add(table(['Variant','Baseline category','N','Rescued ≥4 / ≥8 / ≥12 м','Median Δ'],[[r['variant'],r['baseline_category'],r['n'],r['rescued_4m']+' / '+r['rescued_8m']+' / '+r['rescued_12m'],f(r['median_delta'])] for r in rescue if r['variant'] in (selected,'F2','F4','H2')]))
    add('Offline причины не равны STOP detector: ORIENTATION_FAILURE требует gain >4 м от отдельного GT-frame контрфактического прогона на том же fused input; далее отделяются отсутствие реальных возвратов и observation/template limit. Для F0 сохранена исходная STEP5 классификация, включая 9 seed-related случаев и 7 suspected side changes. У fused диагнозов эти неоднозначности могут входить в более широкие группы. Поэтому изменение названия причины не само по себе rescue; главным остаётся парный Δreach.')
    common_diagnoses=csv_read(OUT/'diagnosis_comparison_common_rules.csv');cd={r['variant']:r for r in common_diagnoses}
    dc=['ORIENTATION_FAILURE','OBSERVATION_TEMPLATE_LIMIT','SENSOR_SPARSITY','CR_GT_GAP','RUN_END','WRONG_STRUCTURE_SUSPECTED']
    add('Дополнительное сопоставление причин по одинаковым широким правилам: для F0 здесь также отключено отдельное выделение seed/side-change, как у fused. Исходная таблица выше и все метрики STEP5 сохранены.')
    add(table(['Variant',*dc],[[r['variant'],*[r.get(k) or 0 for k in dc]] for r in common_diagnoses if r['variant']!='FUSED_BOOTSTRAP']))
    details=csv_read(OUT/'overlap_failure_detail.csv');gates=[]
    for name in ('F0',selected,'F2','F4','F8'):
        rr=[r for r in details if r['variant']==name]
        gates.append([name,len(rr),sum(r['failure_gate']=='CONFIRMED_OVERLAP_FIT' for r in rr),sum(r['failure_gate']=='NEW_ANCHOR_RECHECK' for r in rr),sum(r['failure_gate']=='OVERLAP_TEMPLATE_NO_FIT' for r in rr),cm(median([r for r in rr if r['failure_gate']=='NEW_ANCHOR_RECHECK'],'second_fit_p90'))])
    add(table(['Variant','OVERLAP STOP','Первая подгонка >3см','Новый anchor нарушил старый overlap','Нет overlap fit','Median p90 второй проверки, см'],gates))
    add('В STEP5 лог хранит p90 первой подгонки. Для этого отчёта вторая проверка восстановлена из неизменённых сохранённых accepted points и записанного нового anchor. Она не запускает detector повторно и не меняет его решения. Поэтому остановка OVERLAP_INCONSISTENT при видимом первом p90 <3 см не является противоречием.')
    add('## 5. Разреженность, дальность и вклад источников')
    add('Таблица sparsity использует только случаи, где в следующей половине baseline-среза есть минимум 3 future-GT точки. Отсутствующая разметка не считается доказательством отсутствия возвратов; график плотности также требует доступного GT в соответствующем диапазоне.')
    add(table(['Variant','T returns','N','Median T / fused / unique','Тот же номер шага принят','Rescued ≥4 м'],[[r['variant'],r['current_returns_bin'],r['starts'],f(r['current_returns_median'])+' / '+f(r['fused_returns_median'])+' / '+f(r['fused_unique_median']),r['baseline_failed_step_ordinal_accepted'],r['rescued4']] for r in sparse if r['variant'] in (selected,'F2','F4')]))
    add('Принятие следующего шага проверяется по номеру шага, на котором baseline остановился. У fused собственный локальный frame, поэтому это сопоставимый шаг продвижения, а не побитно одинаковая плоскость; число возвратов измеряется строго в baseline-плоскости.')
    add(table(['Variant','Range','Active starts','Support','Attempts / accepted','Matched ≤5см','Template coverage','Anchor p95, см'],[[r['variant'],r['lo']+'–'+r['hi'],r['active_starts'],r['support_count'],r['attempts']+' / '+r['accepted'],pct(float(r['point_matched5'])) if r['point_matched5'] else 'н/д',f(r['median_template_region_coverage']),f(float(r['anchor_p95'])*100) if r['anchor_p95'] else 'н/д'] for r in ranges if r['variant'] in ('F0',selected,'F2','F4')]))
    add('Range-таблица включает все старты с доступным seed; matched считается только по точкам с reference. Это отличается от фиксированной когорты 783 стартов для continuous reach. Последний bin заканчивается на 75 м по протоколу, хотя единичные новые принятые участки лежат дальше.')
    add(table(['Variant','True range extension starts'],[[x['variant'],x['true_range_extensions']] for x in s['variants']]))
    add('True extension требует gain >4 м по continuous reach и реальных past-source support points дальше ВСЕХ принятых baseline-точек, подтверждённых будущим GT. Увеличение числа точек или исправление ошибки внутри прежней принятой дальности отдельно не считается доказанным расширением диапазона. Source contribution содержит доли каждого возраста, уникальные 1-см позиции, отсутствующие среди current accepted support, и дополнительные nearest canonical-template samples. Азимуты считаются и как уникальные углы, и как пары frame/angle; последние включают повторные наблюдения, поэтому не выдаются за новые направления лучей.')
    source_summary=[]
    for name in ('F2','F3','F4',selected,'H2','F6','F8'):
        unique={}
        for r in sources:
            if r['variant']==name:unique.setdefault((r['run'],r['start_frame'],r['step_index']),r)
        rr=list(unique.values());past=sum(int(r['past_only_template_samples'])>0 for r in rr)
        source_summary.append([name,len(rr),past,pct(past/len(rr)) if rr else 'н/д',f(median(rr,'past_only_template_samples')),f(median(rr,'past_only_voxels_1cm'))])
    add(table(['Variant','Принятых новых шагов','Шагов с новыми template samples от past','Доля','Median новых template samples','Median новых 1-см позиций'],source_summary))
    add('## 6. Registration blur и возможные ложные структуры')
    br=[]
    for name in ('F0','F2','F4','F8'):
        for age in sorted(set(int(r['age_frames']) for r in blur if r['variant']==name)):
            rr=[r for r in blur if r['variant']==name and int(r['age_frames'])==age];br.append([name,'all fused' if age==-1 else age,len(rr),cm(median(rr,'residual_median')),cm(median(rr,'residual_p90')),cm(median(rr,'residual_p95')),cm(median(rr,'dv')),cm(median(rr,'dw')),cm(median(rr,'profile_v_q95q05'))+' / '+cm(median(rr,'profile_w_q95q05'))])
    add(table(['Variant','Age','N','Median residual, см','Median p90, см','Median p95, см','Median dv, см','Median dw, см','Width v/w q95−q05, см'],br));add('![Blur и плотность](gallery/blur_density.png)')
    add('Остатки каждого возраста измерены в ОДНОМ baseline overlap, который current-only уже подтвердил. Это локальная согласованность с шаблоном, а не независимая оценка точности регистрации. Больший остаток может включать другую видимую часть профиля и соседний clutter. Терминальные окна двух алгоритмов обычно различаются — их нельзя сравнивать как один и тот же участок. Tiny alignment проверен только на development; held-out shifts не использовались для подстройки.')
    add('В широком ROI медиана остатка растёт примерно с 0.84 см для T до 2.11 см для T−7; у объединения F8 она 1.56 см. Но p95 даже текущего T около 8 см: эта статистика включает соседний clutter и полный профиль. Поэтому её нельзя назвать физическим размытием рельса до 8 см. Полная ширина v/w включает собственный размер L-профиля; изменение ширины нужно сравнивать с T, а не с нулём.')
    add('Ghost candidates — past-точки в CR terminal ROI без current-соседа ближе 3 см. Такой признак также возникает из-за разреженных лучей и нового угла обзора. Автоматически называть эти точки корпусом поезда или движущимся объектом нельзя. Для верхних кандидатов сохранены срезы и source age. Маска собственных отражений не применялась по решению пользователя.')
    add(table(['Variant','Raw >20см triggers','После проверки охвата GT'],[[x['variant'],x['raw_wrong'],x['wrong']] for x in s['variants']]))
    add('Все большие расхождения остаются в примерах. Если все плохие блоки имеют менее 80% доступного GT, сохраняется отдельная отметка конца reference; такой случай не объявляется подтверждённой другой структурой. Для остальных подозрений нужны визуальный разбор и качество ручной разметки; confidence detector не равен вероятности правильности.')
    reviewed=load(OUT/'audit/suspect_review.json');ends=[r for r in reviewed if r['end_of_reference_with_small_transverse_residual']]
    add(f'Дополнительный геометрический разбор после неизменной автоматической оценки: {len(ends)} из {len(reviewed)} оставшихся подозрительных принятых блоков имеют все оценённые точки с ошибкой >20 см за продольным концом GT, при поперечном остатке <5 см. Здесь F3 и F6 ссылаются на один frame_000852.las в squareT_platform_squareT_switch: ближайшая ошибочная оценённая точка лежит примерно на 28.9 см дальше конца GT, поперечно отклоняется примерно на 2.8 см. Поэтому доказательства перехода на другую структуру в этих двух флагах нет. Автоматические счётчики и continuous reach не исправлялись задним числом; таблица audit/suspect_point_breakdown.csv сохраняет все расстояния и provenance.')
    add('Ещё два флага F2/H2 относятся к одному frame_000543.las у конца roundT_squareT_pressureGate_squareT. Этот старт не входит в 783 сопоставимых GT-старта. В нём максимальное поперечное расхождение плохих оценённых точек около 10.2 см; его нельзя объяснить исключительно малым поперечным остатком у продольного конца. Поэтому он оставлен как нерешённое расхождение с коротким reference и показан отдельно, без включения в основную сопоставимую статистику.')
    add('## 7. Ориентация, M1/M2 и качество на общем участке')
    add(table(['Variant','Subset','Metric','N','Median','p95'],[[r['variant'],r['subset'],r['metric'],r['n'],f(r.get('p50')),f(r.get('p95'))] for r in orient if r['metric']=='tangent_angle_error_deg']))
    paired_angles=csv_read(OUT/'orientation_paired.csv');pa=[]
    for name in ('F2','F4',selected,'H2','M2_F0','M2_BEST'):
        rr=[r for r in paired_angles if r['variant']==name and r['metric']=='tangent_angle_error_deg'];pa.append([name,len(rr),f(median(rr,'baseline_abs'),3),f(median(rr,'fusion_abs'),3),f(median(rr,'delta_abs'),3)])
    add(table(['Variant','Общих принятых номеров шага','Median baseline angle, °','Median fusion angle, °','Median парного Δ absolute angle, °'],pa))
    qr=[]
    for name in v:
        rr=[r for r in quality if r['variant']==name and r.get('baseline_matched5') and r.get('fusion_matched5')]
        delta=[float(r['fusion_matched5'])-float(r['baseline_matched5']) for r in rr];qr.append([name,len(rr),pct(median(rr,'baseline_matched5')),pct(median(rr,'fusion_matched5')),pct(float(np.median(delta))) if delta else 'н/д',pct(float(np.percentile(delta,10))) if delta else 'н/д',pct(float(np.mean(np.asarray(delta)<-.01))) if delta else 'н/д'])
    add(table(['Variant','Общих участков','Baseline median ≤5см','Fusion median ≤5см','Median парного Δ','p10 парного Δ','Доля ухудшений >1п.п.'],qr))
    anchors=csv_read(OUT/'common_anchor_quality.csv');aq=[]
    for name in v:
        rr=[r for r in anchors if r['variant']==name and r.get('delta_p95')];aq.append([name,len(rr),cm(median(rr,'baseline_p95')),cm(median(rr,'fusion_p95')),cm(median(rr,'delta_p95'))])
    add(table(['Variant','Общих участков anchors','Baseline median per-start p95, см','Fusion median per-start p95, см','Median парного Δ, см'],aq))
    add('Anchor comparison берёт только целые принятые шаги, чья дальняя support-точка лежит внутри общей дальности. Нулевой median Δ качества не означает, что каждый старт сохранил качество: p10 и доля ухудшений опубликованы рядом. Причины дальних расхождений проверяются с учётом ограничений GT.')
    add('Common-range quality использует только marched points и future GT до меньшей из двух принятых дальностей. Это отделяет качество на прежнем участке от добавления новых сложных дальних точек. Статистика по всем accepted steps может иметь разный набор дальностей; это ограничение отмечено отдельно. M1 current/M1 fused/M2 current/M2 fused имеют одинаковые gates, window, tail и template; меняется только метод tangent и выбранная история.')
    add('## 8. Скорость и память')
    add(table(['Variant','Полный старт median / p95, мс','Cached march median / p95, мс','Шаг median / p95, мс','Peak RSS median / max, МиБ'],[[name,f(x['uncached_start_ms']['p50'])+' / '+f(x['uncached_start_ms']['p95']),f(x['cached_march_ms']['p50'])+' / '+f(x['cached_march_ms']['p95']),f(x['per_step']['total_ms']['p50'])+' / '+f(x['per_step']['total_ms']['p95']),f(x['peak_process_rss_bytes']['p50']/2**20)+' / '+f(x['peak_process_rss_bytes']['p100']/2**20)] for name,x in bench.items()]))
    add(table(['Variant','Read historical','Transform all','Fusion/dedup','PCA / projection / template per-step median, мс'],[[name,f(x['read_history_ms']['p50']),f(x['transform_ms']['p50']),f(x['fusion_ms']['p50']),' / '.join(f(x['per_step'][q]['p50']) for q in ('pca_ms','projection_ms','template_ms'))] for name,x in bench.items()]))
    add(table(['Variant','Точек median / max'],[[name,f(x['points']['p50'],0)+' / '+f(x['points']['p100'],0)] for name,x in bench.items()]))
    add('Fusion/dedup в отдельном benchmark включает сборку массивов и provenance; это измеренный assembly overhead после вычитания чтения и преобразований. В сырых массовых prediction metadata поле fusion_ms имеет более узкую границу таймера, поэтому для скорости используются только runtime.csv. Component timers не полностью покрывают отклонённые шаги; total step включает их целиком. Запуск интерпретатора не входит в latency алгоритма.')
    add('Замер выполнен последовательно, по 12 одинаковым стартам для каждого варианта, с новым процессом на каждый старт, без оценки GT и rendering. Машина имеет 32 логических CPU; BLAS/OMP/MKL ограничены одним потоком в процессе. Uncached означает обход application cloud cache, но кэш файлов ОС не сбрасывался. Он включает чтение T и истории, provenance, transform, fresh bootstrap и marching. Cached измеряет только marching на уже собранном облаке с готовым seed. Peak RSS — пик всего процесса, включая интерпретатор, проверку равенства результата и повторные cached прогоны; это не только размер XYZ. Одновременно тяжёлые задачи этого исследования во время замера не запускались. F8 измерен отдельным последующим блоком на тех же 12 стартах; повторяемость между днями не проверялась.')
    add('## 9. Неопределённость GT и границы вывода')
    add('Future reference и его ошибочные/неоднозначные классы не исправлялись. В STEP5 медиана per-start p95 согласованности current class2 с future surface была 6.43 см, а p95 этих значений — 21.24 см при ограничении соответствий 30 см. Это не геодезическая истина. Различия 2.8 против 3.0 см нельзя интерпретировать как доказанную физическую точность; здесь важнее метры reach, rescue и сохранение настоящего support.')
    add('Соседние кадры зависимы, оба контрольных заезда уже использовались в предыдущих исследованиях. Выбор STEP5.1 выполнен только на development, но эти данные не являются внешней независимой трассой. Жёсткое правило T+1 ограничивает вывод на медленные/остановочные участки. History truncation и число источников открыто записаны. Нет extrapolation, future density или автоматического production freeze.')
    add('## 10. Примеры и причины')
    for e in examples:
        r=e['row'];stem=r['variant']+'__'+key(r['run'],r['start_frame']);packet=load(OUT/'failures'/r['variant']/(key(r['run'],r['start_frame'])+'.json'))
        add(f"### {r['variant']} · {r['run']} / {frames(r['run'])[r['start_frame']]['file']}")
        add('Категории: '+', '.join(e['tags'])+'. '+case_text(r,packet))
        add(f'[Срезы с GT и без GT](gallery/{stem}_slices.png) · [3D T / fusion](gallery/{stem}_3d.png) · [Diagnostic packet](failures/{r["variant"]}/{key(r["run"],r["start_frame"])}.json)')
        if (OUT/'gallery'/(stem+'_suspect_slices.png')).exists():add(f'[Принятый шаг с наибольшим расхождением >20 см](gallery/{stem}_suspect_slices.png)')
        if (OUT/'gallery'/(stem+'_gt_end_diagnostic.png')).exists():add(f'[Продольная граница GT и подозрительные точки](gallery/{stem}_gt_end_diagnostic.png)')
        if (OUT/'gallery'/(stem+'_blur_overlap.png')).exists():add(f'[Размытие в одном подтверждённом baseline overlap](gallery/{stem}_blur_overlap.png)')
    add('## 11. Ответы на 20 вопросов')
    answers=[
      f'Накопление помогает прежде всего дальнему продолжению, но не каждому старту. F8: P≥40 {pct(v["F8"]["survival"]["40"])}, true extension в {v["F8"]["true_range_extensions"]} стартах, wins/losses {v["F8"]["wins"]}/{v["F8"]["losses"]}. У заранее выбранного {selected} Δmedian лишь {f(db)} м: {verdict}.',
      'Единого оптимума не доказано. F2 — минимальный вариант с заметным снижением sparse STOP; F4/F8 сильнее поднимают дальний хвост, но дороже и иногда резко сокращают reach. Консервативное правило development выбрало H1, чей эффект на контроле оказался небольшим.',
      'На 607 сопоставимых стартах с валидной прошлой скоростью ≥10 м/с H1 дал 0 wins и 0 losses: в этом диапазоне он фактически T-only. В диапазоне 5–10 м/с было 173 старта; F2 дал 41/23 wins/losses, H2 — 54/24. Ни одного оценимого старта ниже 5 м/с в этой таблице нет. Преимущество distance history на остановке не установлено.',
      'Voxel deduplication: 5/10/20 мм проверены на тех же 108 development стартах; ни один voxel вариант не выбран. Это не утверждение о любой другой плотности/регистрации.',
      'Да, локальная несогласованность возрастает: median residual T 0.84 см, объединение F8 1.56 см. Это широкая область около профиля, содержащая clutter; независимой точностью регистрации эти числа не являются.',
      'Возрастные медианы остатка для T−1…T−7: примерно 1.30, 1.41, 1.50, 1.65, 1.86, 1.84, 2.11 см. Рост общий, но не строго монотонный.',
      'Да, прошлые измерения занимают новые позиции и участки шаблона: для F8 '+str(next(r[2] for r in source_summary if r[0]=='F8'))+' из '+str(next(r[1] for r in source_summary if r[0]=='F8'))+' принятых новых шагов имеют canonical-template samples, не занятые текущим принятым support. Полная статистика отделяет новые позиции от простого роста N.',
      'TOO_FEW_SUPPORT rescue основного варианта: '+str(next(r['rescued_4m'] for r in rescue if r['variant']==selected and r['baseline_category']=='TOO_FEW_SUPPORT'))+' стартов на ≥4 м; ≥8/12 м в таблице раздела 4.',
      'NO_NEW_POINTS rescue основного варианта: '+str(next(r['rescued_4m'] for r in rescue if r['variant']==selected and r['baseline_category']=='NO_NEW_POINTS'))+' стартов на ≥4 м.',
      'У основных M1-fusion вариантов медиана парного изменения tangent error на общих принятых номерах шага равна 0°. Общего медианного улучшения направления не доказано. Отдельные ошибки ориентации спасаются; полные углы и условный subset опубликованы.',
      'Baseline ORIENTATION_FAILURE: '+str(next(r['rescued_4m'] for r in rescue if r['variant']==selected and r['baseline_category']=='ORIENTATION_FAILURE'))+' rescued ≥4 м в основном варианте; условные angular metrics выделены отдельно.',
      'По одинаковым широким правилам OBSERVATION_TEMPLATE_LIMIT: baseline '+str(cd['F0']['OBSERVATION_TEMPLATE_LIMIT'])+', H1 '+str(cd['H1']['OBSERVATION_TEMPLATE_LIMIT'])+', F4 '+str(cd['F4']['OBSERVATION_TEMPLATE_LIMIT'])+', F8 '+str(cd['F8']['OBSERVATION_TEMPLATE_LIMIT'])+'. Этот предел в целом не исчез; числа исходной STEP5 классификации показаны отдельно.',
      'OVERLAP_INCONSISTENT: '+str(b['stops'].get('OVERLAP_INCONSISTENT',0))+' → '+str(best['stops'].get('OVERLAP_INCONSISTENT',0))+'. Порог 3 см не ослаблялся; worse case не объявляется blur без дополнительных признаков.',
      'Wrong structures: '+str(b['wrong'])+' → '+str(best['wrong'])+' подозрений после проверки доступности GT; сырые триггеры и все такие примеры сохранены.',
      'Range dependence: см. общую таблицу 0–75 м; основной интерес 30–50 м раскрыт через active starts, returns, occupied template support и rescue, а не одну precision.',
      f'Основной median/p90/max: {f(best["reach"]["p50"])}/{f(best["reach"]["p90"])}/{f(best["reach"]["p100"])} м.',
      f'P≥40: {pct(b["survival"]["40"])} → {pct(best["survival"]["40"])}; P≥50: {pct(b["survival"]["50"])} → {pct(best["survival"]["50"])}.',
      'После fusion наиболее частый STOP: '+max(best['stops'],key=best['stops'].get)+'. Отдельный oracle-frame опыт разделяет frame-estimation sensitivity и observation/template ограничения.',
      'Для выбранной истории H1 существенной пользы M2 нет: median M1 current/fused = '+f(v['F0']['reach']['p50'])+'/'+f(v[selected]['reach']['p50'])+' м, M2 current/fused = '+f(v['M2_F0']['reach']['p50'])+'/'+f(v['M2_BEST']['reach']['p50'])+' м. M2 с F8 отдельно не проверялся; вывод на него не переносится.',
      f'Накопление стоит оставить кандидатом следующего алгоритма, начиная с простого F2 и отдельной проверки новых данных. Автоматически фиксировать primary по этому исследованию не следует: заранее выбранный H1 имеет итог {verdict}, а у длинных окон остаются существенные индивидуальные ухудшения и стоимость вычислений.'
    ];add('\n\n'.join(f'{i+1}. {x}' for i,x in enumerate(answers)))
    add('## 12. Файлы и воспроизведение')
    add(f'Код: `{STAGE}`. Сохранены prediction JSON/NPZ, точное point provenance, failure packets, {len(examples)} разобранных примеров, {len(exports)} LAS-наборов и 8 GIF. В LAS исходный classification и все первоначальные поля сохранены; алгоритм отмечен Extra Bytes pred_cr/pred_step/pred_confidence. source_frame_delta=0 для T, −1 для T−1 и т.д. Все XYZ LAS находятся в общей системе карты своего заезда.')
    add('Обязательные таблицы: per_start_variant.csv, per_step_variant.csv, source_contribution.csv, fusion_density.csv, registration_blur.csv, failure_rescue.csv, orientation_comparison.csv, range_bins_comparison.csv, runtime.csv, memory.csv, method_ablation.csv, las_exports.csv. Дополнительно: paired_comparison.csv, common_range_quality.csv, overlap_comparison.csv, sparsity_rescue.csv, speed_strata.csv. research_lock.json — фиксация исследовательского сравнения до held-out, не production freeze.')
    report='\n\n'.join(lines)+'\n';(OUT/'REPORT_TEMPORAL_FUSION.md').write_text(report,encoding='utf-8');save(OUT/'REPORT_FACTS.json',dict(verdict=verdict,selected=selected,empirical_median_leader=empirical['variant'],delta_median=db,delta_P40=p40,sparse_stop_reduction=reduction,common_quality_median_delta=quality_delta,examples=len(examples),las_sets=len(exports)))
    gallery(examples,exports,report,s)
    (OUT/'report.html').write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>STEP 5.1 — полный отчёт</title><style>'+REPORT_STYLE+'</style><main><p><a href="gallery.html">Галерея и примеры</a></p>'+report_html(report)+'</main></html>',encoding='utf-8')
    print('REPORT',verdict,selected,'median',best['reach']['p50'],'examples',len(examples),flush=True)

def gallery(examples,exports,report,s):
    exported={(r['variant'],r['run'],r['start_frame']):r for r in exports};cards=[]
    for e in examples:
        r=e['row'];name=r['variant'];stem=name+'__'+key(r['run'],r['start_frame']);links=[];x=exported.get((name,r['run'],r['start_frame']))
        if x:
            for field,label in (('fused_input','LAS вход'),('prediction_overlay','LAS результат'),('current_only_comparison','LAS T-only')):links.append(f'<a href="{x[field]}">{label}</a>')
        if (OUT/'animations'/(stem+'.gif')).exists():links.append(f'<a href="animations/{stem}.gif">Анимация</a>')
        packet=load(OUT/'failures'/name/(key(r['run'],r['start_frame'])+'.json'));links.append(f'<a href="failures/{name}/{key(r["run"],r["start_frame"])}.json">Диагностика</a>')
        if (OUT/'gallery'/(stem+'_suspect_slices.png')).exists():links.append(f'<a href="gallery/{stem}_suspect_slices.png">Принятый шаг с расхождением >20 см</a>')
        if (OUT/'gallery'/(stem+'_gt_end_diagnostic.png')).exists():links.append(f'<a href="gallery/{stem}_gt_end_diagnostic.png">Граница GT вдоль рельса</a>')
        if (OUT/'gallery'/(stem+'_blur_overlap.png')).exists():links.append(f'<a href="gallery/{stem}_blur_overlap.png">Blur в подтверждённом overlap</a>')
        cards.append(f'<article class="card" data-tags="{" ".join(e["tags"])}" data-search="{name} {r["run"]} {frames(r["run"])[r["start_frame"]]["file"]}"><h3>{name} · {r["run"]} / {frames(r["run"])[r["start_frame"]]["file"]}</h3><p><b>≤5 см: {f(r.get("continuous_reach_5cm"))} м · Δ {f(r.get("delta_reach"))} м</b> · {r["stop_reason"]}</p><p>{html.escape(case_text(r,packet))}</p><a href="gallery/{stem}_slices.png"><img loading="lazy" src="gallery/{stem}_slices.png" alt="Одинаковый срез T и fusion с GT и без GT"></a><details><summary>3D сравнение</summary><img loading="lazy" src="gallery/{stem}_3d.png" alt="Только T и накопление"></details><p>{" · ".join(links)}</p></article>')
    labels={'all':'Все примеры','rescue':'Улучшения','farthest':'Самые дальние','median':'Средние','median_improvement':'Типичная прибавка среди улучшений','no_effect':'Без эффекта','orientation_rescue':'Исправление ориентации','too_few_rescue':'Спасённые TOO_FEW','no_new_rescue':'Спасённые NO_NEW','sparsity_rescue':'Разреженность','worse':'Стало хуже','blur':'Registration blur','ghost_diagnostic':'Возможные ghosts','wrong_structure_review':'Большие расхождения'}
    options=''.join(f'<option value="{k}">{v}</option>' for k,v in labels.items())
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>STEP 5.1 · Temporal fusion</title><style>body{margin:0;background:#edf2f5;color:#183447;font:16px/1.55 system-ui}main{max-width:1180px;margin:auto;padding:28px}h1{font-size:32px;line-height:1.2}a{color:#00677f}.intro,.card{background:white;border:1px solid #d3e0e7;border-radius:12px;padding:22px;margin:20px 0}img{width:100%;height:auto}.controls{position:sticky;top:0;z-index:5;background:#e8f0f5ee;padding:12px;display:flex;gap:12px;flex-wrap:wrap}select,input{padding:9px;font:inherit;max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.55 system-ui}.hidden{display:none}h3{overflow-wrap:anywhere;font-size:18px}</style><main><h1>STEP 5.1 · Накопление прошлых облаков<br>для продолжения контактного рельса</h1>'''
    page+=f'<section class="intro"><p>Полный контроль: 1 422 старта. Основной вариант выбран до оценки: <b>{s["selected"]}</b>. Начальный поиск одинаков; будущие облака используются только для проверки результата.</p><p><a href="REPORT_TEMPORAL_FUSION.md">Полный отчёт</a> · <a href="per_start_variant.csv">Все старты и варианты</a> · <a href="failure_rescue.csv">Какие отказы спасены</a> · <a href="las_exports.csv">LAS</a> · <a href="runtime.csv">Скорость</a></p></section><img src="gallery/reach_comparison.png" alt="Дальность и парные изменения"><img src="gallery/blur_density.png" alt="Blur и плотность"><details><summary>Полный текст отчёта</summary><pre>'+html.escape(report)+'</pre></details>'
    facts=load(OUT/'REPORT_FACTS.json');baseline=next(x for x in s['variants'] if x['variant']=='F0');chosen=next(x for x in s['variants'] if x['variant']==s['selected'])
    headline=f'<section class="intro"><h2>{facts["verdict"]} · {s["selected"]}</h2><p>Медиана: {f(baseline["reach"]["p50"])} → {f(chosen["reach"]["p50"])} м. P≥40 м: {pct(baseline["survival"]["40"])} → {pct(chosen["survival"]["40"])}. Парные улучшения / ухудшения более 4 м: {chosen["wins"]} / {chosen["losses"]}.</p><p><a href="report.html">Полный отчёт с таблицами и ответами на 20 вопросов</a></p></section>'
    f8=next(x for x in s['variants'] if x['variant']=='F8')
    headline=headline.replace('<p><a href="report.html">',f'<p>Это итог заранее выбранной истории H1. Накопление нескольких кадров заметнее улучшило дальние случаи: у F8 P≥40 м = {pct(f8["survival"]["40"])}, максимум = {f(f8["reach"]["p100"])} м, при {f8["losses"]} индивидуальных ухудшениях более 4 м. Сопоставимый GT есть для 783 стартов.</p><p><a href="report.html">')
    page=page.replace('<section class="intro">',headline+'<section class="intro">',1)
    page+='<div class="controls"><select id="filter">'+options+'</select><input id="search" placeholder="Вариант, заезд или LAS"><span id="count"></span></div>'+''.join(cards)
    page+='''<script>const f=document.getElementById('filter'),q=document.getElementById('search'),cards=[...document.querySelectorAll('.card')];function update(){let n=0;for(const c of cards){const ok=(f.value==='all'||c.dataset.tags.split(' ').includes(f.value))&&c.dataset.search.toLowerCase().includes(q.value.toLowerCase());c.classList.toggle('hidden',!ok);n+=ok}document.getElementById('count').textContent=n+' примеров'}f.onchange=update;q.oninput=update;update()</script></main></html>'''
    (OUT/'gallery.html').write_text(page,encoding='utf-8')
if __name__=='__main__':main()
