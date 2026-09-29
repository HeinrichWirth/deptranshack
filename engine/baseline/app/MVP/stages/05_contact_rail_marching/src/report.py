"""Full Russian scientific report, overview figures and filterable local gallery."""
from common import *
sys.path.insert(0,str(ROOT/'results_rail2d/_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import html
from collections import Counter

def num(x,d=2):return 'н/д' if x is None or x=='' else f'{float(x):.{d}f}'
def pc(x):return 'н/д' if x is None or x=='' else f'{100*float(x):.1f}%'
def table(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def htable(head,rows):return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</tbody></table>'

def explain_case(r,packet):
    reason=r['terminal_evaluation_reason'];last=packet['failed_step'];of=packet.get('oracle_frame') or {};osd=packet.get('oracle_seed') or {}
    pieces=[]
    if reason=='ORIENTATION_FAILURE':
        pieces.append('Эталонная ориентация увеличивает непрерывную дальность с '+num(r.get('continuous_reach_5cm'))+' до '+num(of.get('continuous_reach_5cm'))+' м. Это свидетельствует о чувствительности переноса локального frame; абсолютную ошибку ориентации нужно читать вместе с неопределённостью GT.')
    elif reason=='SEED_OR_INITIAL_DRIFT':
        pieces.append('Замена только начального участка разметкой T даёт '+num(osd.get('continuous_reach_5cm'))+' м вместо '+num(r.get('continuous_reach_5cm'))+' м. Основная гипотеза здесь — состав production seed или раннее смещение, а не отсутствие всех дальних отражений.')
    elif reason=='SENSOR_SPARSITY':
        pieces.append(f'В новой половине есть {r.get("terminal_future_points",0)} future-GT точек, но лишь {r.get("terminal_observable_returns",0)} возвратов текущего T в 5 см от этой поверхности. Будущий кадр видит объект после приближения; текущий кадр не даёт достаточной опоры для подтверждения.')
    elif reason in ('CR_GT_GAP','RUN_END','GT_UNAVAILABLE_AHEAD','CR_SIDE_CHANGE_SUSPECTED'):
        pieces.append('Дальний эталон здесь заканчивается или не связан с исходным участком. Это ограничение оценки; нельзя на основании отсутствующего GT назвать всё дальнейшее продолжение ошибкой. Горизонт reference: '+str(packet['reference'].get('horizon_reason','н/д'))+', максимум '+num(packet['reference'].get('max_GT_available_range'))+' м.')
    else:
        pieces.append('Текущие возвраты около будущей размеченной поверхности присутствуют, но замена ориентации не даёт прироста более 4 м. Ограничение относится к форме, локальному gate или согласованию перекрытия; по этому тесту оно не сводится только к PCA.')
    if r['stop_reason']=='OVERLAP_INCONSISTENT':
        ov=last.get('overlap_residual_p90');pieces.append('Поиск остановлен защитой старого перекрытия. P90 остатка исходной overlap-подгонки — '+num(ov*100 if ov is not None else None)+' см; отдельная новая подгонка также обязана объяснять эти старые точки. Уже подтверждённые точки не перемещались.')
    elif r['stop_reason'] in ('NO_NEW_POINTS','TOO_FEW_SUPPORT'):
        pieces.append('В пределах текущего tracking gate найдено '+str(last.get('candidate_n',0))+' уникальных кандидатов; для принятия требуется минимум 3 объяснённых шаблоном позиции. Расширять окно автоматически после такого отказа протокол не разрешает.')
    elif r['stop_reason']=='LOCAL_FRAME_DEGENERATE':pieces.append('Подтверждённого recent tail недостаточно для устойчивого локального направления; произвольное направление не подставлялось.')
    if r.get('seed_current_GT_conflict'):pieces.append('Дополнительно bootstrap конфликтует с текущей разметкой: p95 '+num(r['seed_current_GT_p95']*100)+' см. Полный ближний срез сохранён отдельно, чтобы видеть обе стороны и не скрыть конфликт узким crop.')
    if r.get('raw_wrong_structure_trigger'):pieces.append('Первичный триггер >20 см разобран отдельно: большой вектор до GT преимущественно продольный и возникает на конце доступной разметки; доказанной смены структуры по этим данным нет.')
    return ' '.join(pieces)

def figures(rows,summary):
    good=[r for r in rows if r.get('evaluation_eligible')];plt.rcParams.update({'font.size':10})
    fig,ax=plt.subplots(1,2,figsize=(13,5))
    for tol,color in ((2,'#e39633'),(5,'#087b84'),(10,'#7254a0')):
        v=np.sort([r[f'continuous_reach_{tol}cm'] for r in good]);ax[0].step(v,1-np.arange(len(v))/len(v),where='post',label=f'≤{tol} см',color=color)
    ax[0].set(xlabel='Непрерывная корректная дальность от LiDAR, м',ylabel='Доля стартов, reach ≥ R',ylim=(0,1.02));ax[0].legend();ax[0].grid(alpha=.2)
    runs=sorted(set(r['run'] for r in good));vv=[[r['continuous_reach_5cm'] for r in good if r['run']==run] for run in runs]
    ax[1].boxplot(vv,tick_labels=['Гермозатвор','Платформа / стрелка'],showfliers=True);ax[1].set(ylabel='Дальность ≤5 см, м',title='Два контрольных заезда');ax[1].grid(axis='y',alpha=.2)
    fig.suptitle('Основной результат: строгая непрерывность, доступная будущая разметка');fig.tight_layout();fig.savefig(OUT/'gallery/reach.png',dpi=135);plt.close(fig)
    steps=csv_read(OUT/'orientation_metrics.csv');fig,axes=plt.subplots(2,2,figsize=(13,9))
    for ax,field,label in zip(axes.flat[:2],('GT_anchor_error','tangent_angle_error_deg'),('Ошибка опоры, см','Ошибка касательной, °')):
        rr=[r for r in steps if r.get(field) and r.get('range_far')];factor=100 if field=='GT_anchor_error' else 1
        ax.scatter([float(r['range_far']) for r in rr],[float(r[field])*factor for r in rr],s=5,alpha=.2,c='#197896');ax.set(xlabel='Дальность от LiDAR, м',ylabel=label);ax.grid(alpha=.2)
    rr=[r for r in steps if r.get('transverse_spread') and r.get('tangent_angle_error_deg')]
    axes[1,0].scatter([float(r['transverse_spread']) for r in rr],[float(r['tangent_angle_error_deg']) for r in rr],s=5,alpha=.2,c='#a16534');axes[1,0].set(xlabel='Робастный transverse scatter, м²',ylabel='Ошибка касательной, °');axes[1,0].grid(alpha=.2)
    stops=summary['stops'];names=sorted(stops,key=stops.get,reverse=True);axes[1,1].barh(names[::-1],[stops[n] for n in names[::-1]],color='#3a8292');axes[1,1].tick_params(axis='y',labelsize=7);axes[1,1].set(xlabel='Число доступных стартов',title='Первичная причина остановки детектора')
    fig.tight_layout();fig.savefig(OUT/'gallery/drift_and_stops.png',dpi=135);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(13,11))
    for line,(field,label,factor) in enumerate((('GT_anchor_error','Ошибка опоры, см',100),('yaw_angle_error_deg','Ошибка yaw, °',1),('pitch_angle_error_deg','Ошибка pitch, °',1))):
        rr=[r for r in steps if r.get(field) and r.get('range_far')]
        y=np.array([float(r[field])*factor for r in rr])
        for column,(xfield,xlabel) in enumerate((('step_index','Номер шага'),('range_far','Дальность от LiDAR, м'))):
            x=np.array([float(r[xfield]) for r in rr]);ax=axes[line,column]
            ax.scatter(x,y,s=4,alpha=.16,c='#197896',rasterized=True)
            bins=x if column==0 else np.floor(x/4)*4
            values=np.unique(bins);ax.plot(values,[np.median(y[bins==v]) for v in values],c='#c8712d',lw=1.5,label='Медиана по шагу / 4 м')
            ax.set(xlabel=xlabel,ylabel=label);ax.set_yscale('symlog',linthresh=1 if factor==100 else .1);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Drift: все доступные оценки; симметричная логарифмическая шкала ошибок')
    fig.tight_layout();fig.savefig(OUT/'gallery/error_progression.png',dpi=135);plt.close(fig)
    obs=csv_read(OUT/'observability.csv');bins=sorted(set((float(r['lo']),float(r['hi'])) for r in obs));fig,ax=plt.subplots(figsize=(12,5))
    data=[[float(r['unique_xyz']) for r in obs if float(r['lo'])==lo] for lo,hi in bins]
    ax.boxplot(data,tick_labels=[f'{lo:g}–{hi:g}' for lo,hi in bins],showfliers=False);ax.set(yscale='symlog',xlabel='Дальность, м',ylabel='Уникальные точки T рядом с будущей CR разметкой',title='Наблюдаемость в текущем облаке; GT используется только после поиска');ax.grid(alpha=.2);fig.tight_layout();fig.savefig(OUT/'gallery/observability.png',dpi=135);plt.close(fig)

def main():
    s=load(OUT/'SUMMARY.json');rows=load(OUT/'analysis_rows.json');cfg=load(OUT/'config.json');examples=load(OUT/'gallery/examples.json');bench=load(OUT/'runtime/sequential_summary.json')
    paired=csv_read(OUT/'calibration/paired_selected_summary.csv');sweep=csv_read(OUT/'method_ablation.csv');oracles=csv_read(OUT/'oracle_comparison.csv');las=load(OUT/'las/index.json')
    good=[r for r in rows if r.get('evaluation_eligible')];diagnoses=s['diagnoses'];n=len(good);reach=s['reach']['5'];most=max(s['stops'],key=s['stops'].get)
    frac=lambda name:diagnoses.get(name,0)/n if n else 0
    figures(rows,s)
    lines=[];add=lines.append
    add('# STEP 5 — Contact rail marching / self-aligned track frame')
    add(f'**PROPOSED METHOD:** {cfg["method"]}, локальный PCA + Bishop; W={cfg["window"]:g} м, advance={cfg["advance"]:g} м, recent tail={cfg["tail"]:g} м, tracking gate ±{cfg["gate"]*100:g} см. Без численного уточнения ориентации и roll. Это исследовательская конфигурация, не production release.')
    add(table(['Показатель','Результат'],[
        ['Held-out starts',s['total_starts']],['Seed available',f'{s["seed_available"]} / {s["total_starts"]} = {pc(s["seed_availability"])}'],['С пригодной будущей разметкой',n],
        ['Median continuous reach ≤5 cm',num(reach.get('p50'))+' м'],['p90 / p95 reach',num(reach.get('p90'))+' / '+num(reach.get('p95'))+' м'],['Maximum',num(reach.get('p100'))+' м'],
        *[[f'P(reach ≥{d} м)',pc(s['survival'][str(d)])] for d in (20,30,50,75,100)],
        ['Median per-start anchor p95',num(s['median_anchor_p95']*100 if s['median_anchor_p95'] is not None else None)+' см'],['Most common detector STOP',most],
        ['Orientation failure fraction (offline diagnostic)',pc(frac('ORIENTATION_FAILURE'))],['Sparse/no-return fraction',pc(frac('SENSOR_SPARSITY'))],['Wrong-structure suspicion',pc(frac('WRONG_STRUCTURE_SUSPECTED'))],
        ['Marching step, sequential median / p95',num(bench['per_step']['total_ms'].get('p50'))+' / '+num(bench['per_step']['total_ms'].get('p95'))+' мс'],
        ['Полный старт с чтением LAS, median / p95',num(bench['per_start']['end_to_end_ms'].get('p50'))+' / '+num(bench['per_start']['end_to_end_ms'].get('p95'))+' мс']]))
    add('**HYPOTHESIS: PARTIALLY SUPPORTED.** Самоперенос системы по наблюдаемому контактному рельсу работоспособен. Уменьшение transverse scatter само по себе не гарантирует правильную касательную; numerical refinement не улучшил development результат. Положение плоскости не влияет на центрированный scatter одних и тех же точек — midpoint полезен через состав среза и перекрытие.')
    add('## 1. Данные и точный охват')
    add(f'Использована папка `{dataset()}`. Она содержит 6 551 исходно размеченный кадр и контрольные заезды из задания. Последний автоматический проход в `LAS_FRAMES_AUTOLABELLED_FROZEN_20260927` этих двух заездов не содержит и не подменяет данный набор. Исходные LAS и предыдущие frozen stages не изменялись.')
    add('Development: new_data_part_01a1, 01a2, 01c2, 01d1, 21, 22; doubleT_platform; roundT_doubleT; roundT_pressureGate_roundT. Held-out: roundT_squareT_pressureGate_squareT и squareT_platform_squareT_switch. Контрольные заезды встречались на прошлых этапах, поэтому это проверка новой конфигурации без её подстройки на них, а не совершенно новый сенсор.')
    add('Development bootstrap: 108 равномерных стартов. Компактный sweep: 36 заранее выбранных стартов; 23 имеют пригодный дальний эталон. Все параметры и исключения сохранены в calibration/. После выбора выполнено парное сравнение M0…M4 с одинаковым выбранным gate. Отдельные отказы не удалялись из таблиц.')
    add(table(['Контрольный заезд','Всего','Seed','Дальний GT','Медиана ≤5 см'],[[r['run'],r['starts'],r['seed_available'],r['evaluated'],num(r['reach_5'].get('p50'))] for r in s['per_run']]))
    add('## 2. Изоляция и начальное направление')
    add('Inference читает только XYZ T; LAS уже в общей системе своего заезда, поэтому выполняется ровно одно преобразование `(p_map−t_T) @ R_T`. В bootstrap передаются только pose T и T+1. Классы 1/2, future clouds и future trajectory не входят в API marching. После STEP1/FINAL STEP2 используются только confirmed CR support, side, anchor и начальный базис. Frozen confidence принимается по статусу STEP2; новый confidence не переинтерпретируется как старый acceptance threshold.')
    add('Сохранено прежнее требование ≥50 см надёжного движения, но теперь только между T и T+1. Ожидание более поздней позы запрещено заданием. Поэтому малое движение, разрыв времени, отсутствие T+1 и отказы bootstrap — START_UNAVAILABLE, а не ошибка продолжения. Это ограничивает общий охват, особенно около остановки платформенного заезда.')
    add(table(['Причина недоступного старта','Количество'],sorted(s['start_unavailable'].items(),key=lambda x:-x[1])))
    add('Проверки: реальная смена classification в LAS не меняет прочитанные XYZ; файловый guard допускает только T; geometry API отвергает label-bearing dict; лишняя future trajectory в pose API отвергается; проверены знак PCA, ортонормальность Bishop, непрерывность индексов, отсутствие повторного добавления overlap и остановка на пустом промежутке. Исходники тестов: `MVP/stages/05_contact_rail_marching/tests/test_isolation.py`.')
    add('## 3. Как работает поиск')
    add('По recent tail оценивается локальная касательная; знак согласован с предыдущей. Базис переносится минимальным вращением. Плоскость ставится в середину предыдущего 8-метрового участка; новое окно имеет 4 м перекрытия и около 4 м новых наблюдений. Локальная поперечная опора повторно подгоняется по confirmed overlap. Затем тот же canonical template ищется в новой половине, в пределах ±3 см. Ни положение ходовых, ни нормативное расстояние до них больше не используются.')
    add('Принятие требует реальных новых уникальных точек, согласованной формы, отсутствия равноценных отдельных кандидатов, согласия со старым overlap и допустимого изменения ориентации. Минимум 3 уникальных позиции; расстояние до шаблона ≤2 см; p90 остатка overlap ≤3 см; суммарный поворот yaw/pitch ≤5° за шаг. Это исследовательские gates, а не железнодорожные нормативы. Жёсткого u-span нет. При пустом новом участке STOP: нет extrapolation через разрыв, Kalman, spline continuation, DBSCAN или прыжка на следующую структуру.')
    add('## 4. Development и выбор конфигурации')
    add('Проверены W=6/8/10/12 м и advance=W/4 либо W/2; recent tail=4/6/8/10 м; bins=0.2/0.3/0.5 м; robust trace, logdet, ellipse q90/q95 и MAD; refinement bounds=1/2/3/5°; roll=1/2/3°; orientation jump=1/2/3/5°; gate=3/5/10/15/20 см; fixed/free anchor; плоскость start/midpoint/end; Bishop против reset-up. Это последовательность сравнений по одному фактору, не полный многомерный grid и не доказанный глобальный optimum.')
    add(table(['Метод при ±3 см','N с GT','Median reach, м','p10 / p90','Wrong suspicion','Median runtime, мс'],[[r['method'],r['starts'],num(r['median_reach']),num(r['reach_p10'])+' / '+num(r['reach_p90']),pc(r['wrong_structure_rate']),num(r['runtime_median_ms'])] for r in paired]))
    add('Критерий — среднее reach≤5cm с одинаковым весом заездов и штрафом за подозрение на ошибочную структуру. M2 имел лучший score. M1 оказался в пределах 5% и выбран как более простой согласно правилу предпочтения простоты. Его медиана немного ниже; это показано, а не скрыто. M3 numerical refinement и M4 roll не дали основания усложнять главный вариант. Все исходные sweep-таблицы сохранены.')
    add('В development score использовался исходный автоматический триггер >20 см. Как показала последующая диагностика, он может срабатывать у конца GT. У выбранных M1 и M2 на development триггеров не было. Даже если полностью убрать штраф M3, его equal-run mean score составит 21.81 м против 25.27 м у M1 и 26.38 м у M2; выбранный метод от этой оговорки не меняется. Колонка Wrong suspicion в development таблице — исходный триггер, не ручное подтверждение другой структуры.')
    detail=[r for r in sweep if r['variant'] in ('M2','M3','M4','plane_start','plane_end','reset_up','fixed_anchor') or r['variant'].startswith(('gate_','objective_'))]
    add(table(['Однофакторное сравнение (base gate ±10см)','Median reach','p90','Score','Runtime, мс'],[[r['variant'],num(r['median_reach']),num(r['reach_p90']),num(r['score']),num(r['runtime_median_ms'])] for r in detail]))
    add('`freeze.json` создан до первого held-out prediction. Он содержит SHA конфигурации, кода, canonical template и неизменённых зависимостей STEP1/FINAL STEP2. После held-out параметры не менялись. Production freeze намеренно не создавался.')
    short=load(OUT/'calibration/window_initialization_diagnostic.json')
    add('Уточнение абляции: исходный W=6 / advance=1.5 начинал внутри 8-метрового seed и не имел новой области на первом шаге. Его исходное низкое значение не является честным сравнением продолжения. После фиксации выполнен отдельный development-only диагностический пересчёт от последних 6 м известной истории: median reach '+num(short['reach']['p50'])+' м, p90 '+num(short['reach']['p90'])+' м при gate ±10 см. Главный алгоритм W=8/4, его параметры и held-out predictions не менялись; новая настройка по контрольным данным не выполнялась.')
    add('## 5. Что именно считается ошибкой и дальностью')
    add('Seed проверяется по классу 2 текущего T: будущие T+2 уже могли проехать ближние точки. Каждая новая точка проверяется исключительно по будущей размеченной поверхности. Fused reference собирается от T+2 до 150 м движения либо до конца непрерывного заезда; T+1 cloud не используется. Очистка — 5 мм voxel, компонент связности от seed, robust profile anchors в 0.5-метровых bins. Кривая не интерполирует пропуски >0.8 м; coarse component connectivity имеет отдельные масштабы 15/37 см, сохранённые в protocol.json.')
    add('Continuous correct reach: начиная с seed, каждый принятый блок должен иметь ≥3 оценённых точки, ≥80% support в доступном GT и ≥95% точек в заданном допуске. Первый плохой или неоцениваемый блок прекращает continuous reach. Максимум отдельных правильных точек не заменяет её. Основная дальность — Euclidean range от LiDAR T; `s_from_seed` — отдельный реконструированный продольный параметр. Отсутствие GT не превращается в нулевую ошибку или доказанный failure.')
    add('Непрерывность здесь определена на уровне последовательных окон, а не как наличие измерения на каждом сантиметре профиля. Целое пустое новое окно не перескакивается; разреженные промежутки внутри окна отражаются в числе реальных возвратов и не заполняются синтетическими точками.')
    add(table(['Допуск','p10','p25','p50','p75','p90','p95','max','95% block-bootstrap CI медианы'],[[f'{t} см',*[num(s['reach'][str(t)].get('p'+str(p))) for p in (10,25,50,75,90,95,100)],' / '.join(num(x) for x in s['block_bootstrap'][str(t)]['median_interval_95'])] for t in (2,5,10)]))
    add('Bootstrap интервал построен по исходным блокам по 20 последовательных индексов, отдельно внутри каждого заезда, 1 000 повторов. Соседние старты зависимы; два заезда не становятся сотнями независимых трасс.')
    add(f'Отдельная проверка исключений: у {s["seed_current_GT_conflicts"]} доступных стартов p95 seed относительно текущего class2 превышает 20 см; у {s["future_component_not_linked"]} future-компонент не связался с seed. Это может быть ошибкой bootstrap, разметки, сменой стороны или разрывом; такие случаи нельзя выдавать за доказанно правильные старты. Основная таблица условна на доступный GT.')
    add(table(['Дальность','Подтверждённое достижение среди ВСЕХ доступных seed'],[[f'{d} м',pc(s['demonstrated_survival_all_seed_available'][str(d)])] for d in (20,30,50,75,100)]))
    add('Последняя доля — консервативная доля подтверждённых достижений, а не утверждение, что каждый неоцениваемый старт ошибочен.')
    add('![Дальность](gallery/reach.png)')
    add('## 6. Неопределённость будущего эталона')
    add('Future fusion — зарегистрированная разметка, а не геодезическая истина. Ошибка nearest-surface может быть оптимистичной: плотное объединение сканов увеличивает шанс близкого соседа. Разные видимые части CR дают разные median/profile anchors. Поэтому отдельно сохранены ошибка точек, ошибка canonical anchor, текущая разметка, fused разметка и визуальные примеры.')
    add('`future_gt_alignment.csv`: nearest distance current class2 → fused future в общей области с отсечением отсутствующих соответствий >30 см. Это оценка согласованности общих поверхностей, не независимая точность регистрации. Медиана per-start p95: '+num(s['alignment_p95'].get('p50',0)*100)+' см; p95 этих p95: '+num(s['alignment_p95'].get('p95',0)*100)+' см. Результат при 1–2 см чувствителен к регистрации, разметке и voxel. Не заявляется физическая точность 1 см.')
    add('## 7. Остановки и причинная диагностика')
    add(table(['Первичная причина детектора','N'],sorted(s['stops'].items(),key=lambda x:-x[1])))
    add(table(['Диагноз после prediction','N','Доля пригодных стартов'],[[k,v,pc(v/n)] for k,v in sorted(diagnoses.items(),key=lambda x:-x[1])]))
    add('Причины разделены: detector STOP не знает GT; offline diagnosis использует oracle и наблюдаемость. Orientation failure означает, что при эталонном направлении continuous reach вырос более чем на 4 м. Seed/initial drift — такое улучшение от oracle seed 8 м. Sensor sparsity: будущая разметка есть, но в новой половине текущего T менее 3 возвратов в 5 см от неё. Если точки есть, а даже oracle не помогает, используется observation/template limit. Wrong structure помечается как подозрение при p95 новых точек >20 см, а не как безусловный факт при несовершенной разметке.')
    add(f'Сработало {s["raw_wrong_structure_triggers"]} первичных триггера >20 см. После проверки охвата GT осталось {s["wrong_structure_after_GT_coverage_review"]} подозрений на другую структуру. В обоих триггерах GT заканчивается перед разрывом времени: start 61 имеет GT до 8.69 м, start 95 — до 12.14 м; оценено только 25.3% и 11.2% соответствующих новых блоков. Большой nearest-distance направлен вдоль пути (до 2.93/3.87 м), median поперечного расстояния проблемных точек — 2.23/3.30 см. Это отсутствие продольного эталона, а не доказанный drift-to-wall. Основные reach-значения оставлены консервативно до конца доступного GT. Оба исходных триггера и первый проблемный срез сохранены в галерее и wrong_structure_investigation.csv.')
    add('![Ошибки и остановки](gallery/drift_and_stops.png)')
    add('## 8. Oracle и разложение ошибки')
    oo=[];oracle_stats={};oracle_gains={}
    for name in sorted(set(r['oracle'] for r in oracles)):
        rr=[r for r in oracles if r['oracle']==name and r['run'] in SPLIT['validation'] and r.get('evaluation_eligible')=='True' and r.get('continuous_reach_5cm')]
        x=[float(r['continuous_reach_5cm']) for r in rr];a=percentiles(x);oracle_stats[name]=a;oo.append([name,len(x),num(a.get('p50')),num(a.get('p90')),num(a.get('p100'))])
    add(table(['Oracle diagnostic','N','Median ≤5cm, м','p90','max'],oo))
    add('ORACLE_SEED использует только class2 T на первых 8/10/12/15 м; далее labels снова отсутствуют. ORACLE_FRAME использует локальную оценочную касательную из current-near/future curve. Совмещённый oracle заменяет и seed, и frame. Эти опыты намеренно не являются production. GT frame тоже является оценкой. Если будущий GT обрывается, oracle останавливается, а не изобретает касательную.')
    seed_audit=load(OUT/'audit/oracle_seed_profile_audit.json')
    add('**Raw ORACLE_SEED не является идеальным стартом или верхней границей качества.** Размеченный class2 включает более широкий профиль, чем подтверждённые рабочим шаблоном точки. Замена всего support меняет задачу overlap. Поэтому низкая дальность такого oracle не доказывает превосходство production seed и не позволяет оценить чистый вклад ошибки начального поиска.')
    add(table(['Состав seed / frame','Стартов','Без единого принятого шага','Median первого overlap p90, см'],[[name,a['starts'],a['no_accepted_march_steps'],num(a['first_overlap_residual_p90'].get('p50',0)*100)] for name,a in seed_audit.items()]))
    add('Дополнительные ORACLE_SEED_TEMPLATE8 и ORACLE_SEED_TEMPLATE8_FRAME рассчитаны после основного held-out прогона как проверка чувствительности. На первых 8 м class2 выбирает сторону; canonical template подгоняется только переносом, затем сохраняются реальные class2-точки внутри его области и в пределах прежнего допуска 2 см. Весь дальнейший поиск сохраняет frozen gates. Это согласует состав seed с production support, но всё ещё зависит от разметки и шаблона: математически perfect seed не получен. Длинные raw seed 10/12/15 м остаются отдельной серией, их длины нельзя сравнивать как чистый эксперимент качества старта.')
    prod={(r['run'],r['start_frame']):r for r in good};paired_oracle=[]
    for name in sorted(set(r['oracle'] for r in oracles)):
        rr=[r for r in oracles if r['oracle']==name and (r['run'],int(r['start_frame'])) in prod and r.get('continuous_reach_5cm')]
        delta=[float(r['continuous_reach_5cm'])-prod[r['run'],int(r['start_frame'])]['continuous_reach_5cm'] for r in rr]
        a=percentiles(delta);oracle_gains[name]=dict(a,over4=float(np.mean(np.array(delta)>4)) if delta else None);paired_oracle.append([name,len(delta),num(a.get('p50')),pc(oracle_gains[name]['over4'])])
    add(table(['Парное сравнение с production','Общих стартов','Median Δreach, м','Улучшение >4 м'],paired_oracle))
    add('Исходные категории причин сохранены: SEED_OR_INITIAL_DRIFT использует raw ORACLE_SEED_8, а не новую TEMPLATE8 серию. Следовательно, число таких причин — результат конкретного контрфактического теста, а не исчерпывающая оценка всех ошибок seed. Дополнительные результаты включены в отдельные поля diagnostic packets и таблицу выше; основной reach, config и prediction не пересчитаны.')
    add('Длинный oracle seed — больше подтверждённой истории; поиск начинается от её последних 8 м. Новые точки появляются только за концом seed. Это отдельно проверено unit test, чтобы не остановиться ложно внутри уже известного участка.')
    add('## 9. Кривизна, уклон, плотность и drift')
    geometry_check=load(OUT/'audit/geometric_hypotheses.json')
    add('Проверка PCA/ковариации: разница между минимальным поперечным trace и суммой двух меньших собственных значений — '+f'{geometry_check["raw_PCA_trace_identity"]["difference"]:.3g}'+' м². Максимальное изменение центрированных objectives при переносе плоскости с сохранением тех же точек — '+f'{geometry_check["max_translation_difference"]:.3g}'+'. Это математическая проверка, а не самостоятельная проверка правильности выделенных CR points.')
    compact=csv_read(OUT/'compactness_metrics.csv');compact=[r for r in compact if r['variant'] in ('M1','M2','M3','M4','objective_logdet','objective_ellipse90','objective_ellipse95','objective_mad')]
    add(table(['Objective / метод','N steps','Spearman scatter ↔ tangent error','Tangent p95, °'],[[r['variant'],r['steps'],num(r['spearman_spread_tangent_error'],3),num(r['angle_p95'])] for r in compact]))
    strata=csv_read(OUT/'curvature_grade_summary.csv');add(table(['Разбиение','Группа','N','Median reach','p90'],[[r['kind'],r['bin'],r['n'],num(r.get('p50')),num(r.get('p90'))] for r in strata]))
    add('Кривизна и pitch вычислены только для оценки по fused curve и не передаются поиску. Уклон в sensor-T coordinates включает наклон системы отсчёта: это не независимая геодезическая продольная отметка.')
    add('![Наблюдаемость](gallery/observability.png)')
    drift=load(OUT/'drift_summary.json');add(table(['Ошибка','N','Spearman со step','Spearman с range'],[[r['metric'],r['n'],num(r['spearman_step'],3),num(r['spearman_range'],3)] for r in drift]))
    add('![Ошибки опоры, yaw и pitch по номеру шага и дальности](gallery/error_progression.png)')
    add('Корреляции описательны: ранние STOP цензурируют дальнейшие ошибки; не следует считать зависимые шаги независимыми измерениями. `observability.csv` содержит returns, unique XYZ, distinct rings и azimuth samples по всем диапазонам 0–150 м. `distance_bins.csv` сохраняет precision-like fractions 1/2/3/5/10/20 см и coverage; покрытие только активной дальности отдельно находится в `per_start_frame.csv`.')
    binrows=csv_read(OUT/'distance_bins.csv');range_table=[]
    for lo,hi in zip((0,10,20,30,40,50,60,75,100,125),(10,20,30,40,50,60,75,100,125,150)):
        rr=[r for r in binrows if float(r['lo'])==lo and int(r['evaluated'])>0];count=sum(int(r['evaluated']) for r in rr)
        matched=sum(float(r['matched_5cm'])*int(r['evaluated']) for r in rr if r['matched_5cm'])/count if count else None
        range_table.append([f'{lo}–{hi}',len(rr),count,pc(matched)])
    add(table(['Диапазон, м','N contributing starts','Оценено точек','Matched ≤5см (вес точек)'],range_table))
    add('## 10. Скорость')
    add(table(['Компонент','Медиана, мс','p95, мс'],[[name,num(m.get('p50')),num(m.get('p95'))] for name,m in bench['per_step'].items()]))
    add('Это отдельный последовательный замер на 24 равномерно выбранных доступных held-out стартах, один процесс, один BLAS-поток, без evaluation/rendering. `unattributed_ms` включает подготовку, bookkeeping и работу терминального шага, не завершившего отдельный таймер. End-to-end включает чтение и sensor transform, seed и marching. Среднее время массового многопроцессного прогона не выдаётся за online latency. При данной реализации нельзя обещать стабильные 10 FPS; графики и future GT относятся к offline работе.')
    add('Кэш ОС перед замером не сбрасывался: LAS ранее читались в исследовании. Поэтому около 8 мс на чтение/transform — warm-I/O, а не гарантированная задержка первого чтения с диска.')
    add('## 11. Разбор примеров')
    for e in examples:
        r=e['row'];k=key(r['run'],r['start_frame']);packet=load(OUT/'failures'/(k+'.json'));last=packet['failed_step']
        add(f'### {r["run"]} / {frames(r["run"])[r["start_frame"]]["file"]}')
        add(f'Категории: {", ".join(e["tags"])}. Continuous ≤5cm: **{num(r.get("continuous_reach_5cm"))} м**; принято до {num(r.get("max_accepted_range"))} м; шагов {r["march_steps"]}. STOP: `{r["stop_reason"]}`. Offline: `{r["terminal_evaluation_reason"]}`. В последней новой половине: current returns рядом с будущим GT {r.get("terminal_observable_returns",0)}, current class2 {r.get("terminal_current_gt_points",0)}, future class2 {r.get("terminal_future_points",0)}. Overlap {last.get("confirmed_overlap_n",0)}, candidate unique {last.get("candidate_n",0)}.')
        add(explain_case(r,packet))
        add(f'[Весь seed 0–8 м](gallery/{k}_bootstrap.png) · [Срез без GT](failures/{k}_without_gt.png) · [Срез с GT](failures/{k}_with_gt.png) · [3D](gallery/{k}_3d.png) · [Полный diagnostic packet](failures/{k}.json)')
        if r.get('wrong_structure_suspected') or r.get('raw_wrong_structure_trigger'):add(f'[Первый срез с триггером расхождения GT](gallery/{k}_first_error_with_gt.png)')
    add('## 12. Ответы на 17 вопросов')
    ofstats=oracle_stats['ORACLE_FRAME'];ofgain=oracle_gains['ORACLE_FRAME'];tseed=oracle_gains['ORACLE_SEED_TEMPLATE8']
    answers=[
      'Minimum transverse spread: математический минимум trace подтверждён с расхождением 1.36×10⁻¹⁵ м². Но на парном development M1 корреляция scatter с ошибкой касательной −0.211, p95 ошибки 0.761°; у M3 p95 2.280°. Уменьшение scatter не подтверждено как универсальный индикатор верного направления.',
      'Midpoint: median reach start/midpoint/end = 8.67/16.08/14.69 м в однофакторном M2 sweep при gate ±10 см. Для тех же самых points перенос плоскости меняет centered objectives максимум на 3.55×10⁻¹⁵; выигрыш в поиске относится к составу окна/overlap, не к самостоятельному уменьшению scatter от переноса начала.',
      'PCA достаточно для выбранного варианта: при gate ±3 см M1/M2/M3 дали median 27.97/30.52/20.05 м. M1 score 25.27 м находится в пределах 5% от M2 26.38 м; выбран более простой M1. Numerical refinement не улучшил результат.',
      'Bishop против reset-up при base gate ±10 см: обе медианы 16.08 м; equal-run scores 16.61 и 16.52 м. Сильное преимущество Bishop на этом наборе не доказано; он выбран для непрерывного переноса roll.',
      'Roll correction не выбрана: при ±3 см M4 median 22.83 м и runtime 1414 мс против M1 27.97 м и 464 мс на том же development наборе.',
      'Выбрано W=8 м / advance=4 м / tail=8 м. В базовом M2 sweep W=8/4 дал median 16.08 м; W=10/5 — 14.83 м, W=12/6 — 14.23 м. Это локальный выбор однофакторного sweep, не глобальная оптимальность всех совместных настроек.',
      'Drift: Spearman anchor error со step/range = 0.365/0.355; tangent error = 0.067/0.041. Якорная ошибка растёт заметнее угловой. Это не линейная скорость в см/м; остановки цензурируют дальнейший drift.',
      f'Средняя дальность ≤5cm: {num(reach.get("mean"))} м; медиана {num(reach.get("p50"))} м на {n} пригодных стартах.',
      f'p90/p95 дальности: {num(reach.get("p90"))}/{num(reach.get("p95"))} м, максимум {num(reach.get("p100"))} м.',
      'Первая наблюдаемая остановка: overlap 580/869, недостаточно support 196/869, нет новых points 80/869. Среди 783 оцениваемых offline diagnosis: observation/template 375, orientation 254, sparsity 18, seed 9; оставшиеся 127 — GT gap, конец заезда или подозрение на смену стороны. Эти группы зависят от диагностических правил, а не являются доказанными физическими причинами.',
      'С raw seed сравнение качества некорректно: 846/869 seed8 не прошли ни одного шага из-за другого состава support. У согласованного с шаблоном seed8 median парного Δreach '+num(tseed.get('p50'))+' м; улучшение >4 м в '+pc(tseed['over4'])+' случаев. Это всё ещё диагностическая замена по несовершенной разметке.',
      'Oracle frame: median reach '+num(ofstats.get('p50'))+' м против production '+num(reach.get('p50'))+' м; median парного Δ '+num(ofgain.get('p50'))+' м. Улучшение >4 м в '+pc(ofgain['over4'])+' общих стартов; разность медиан не равна медиане парных разностей.',
      'С оценочным GT frame: median/p90/max '+num(ofstats.get('p50'))+'/'+num(ofstats.get('p90'))+'/'+num(ofstats.get('p100'))+' м. Это не чистый физический предел LiDAR: остаются шаблон, перекрытие, состав seed и ошибки GT.',
      'Matched ≤5 см по дальности (с весом точек): '+ '; '.join(row[0]+' м — '+row[3]+' / '+str(row[2])+' оценённых точек' for row in range_table if row[2]) + '. Это условно на найденные и оцениваемые points; прекращение детекции снижает полноту, но не обязано снижать такую precision.',
      f'Wrong-structure suspected: {diagnoses.get("WRONG_STRUCTURE_SUSPECTED",0)} / {n}; все такие случаи включены в галерею. Это требует визуального подтверждения при неточной разметке.',
      'Overlap останавливает 580/869 стартов (66.7%). Подтверждённой цепочки перехода на стену здесь нет; следовательно, чувствительность защиты именно к такой цепочке оценить нельзя. Защита также может преждевременно остановить корректный рельс.',
      'Как основа следующего исследования — да. Production-ready статус не присваивается; следующий шаг стоит выбирать после просмотра sparse, orientation и GT-conflict примеров.'
    ]
    add('\n\n'.join(f'{j+1}. {a}' for j,a in enumerate(answers)))
    add('## 13. Артефакты и воспроизведение')
    add(f'Сохранены {len(las)} набора LAS (overlay + predicted map + predicted sensorT), минимум 8 GIF, последние срезы всех доступных held-out стартов, шаги, индексы точек и терминальные JSON. Overlay сохраняет исходную classification, порядок и все исходные поля; prediction находится только в Extra Bytes. Проверка read-back сравнивает все исходные поля побитно. Код: `MVP/stages/05_contact_rail_marching/`.')
    add('Основные таблицы: per_start_frame.csv, per_step.csv, per_predicted_point.csv.gz, reach_summary.csv, distance_bins.csv, failure_summary.csv, method_ablation.csv, orientation_metrics.csv, compactness_metrics.csv, future_gt_alignment.csv, oracle_comparison.csv, runtime.csv, las_exports.csv. Полные paths, source indices, local frames, кандидаты и point-level GT distances доступны в packets/NPZ; исходные данные не заменяются экспортом.')
    add('Ограничения: один LiDAR/существующая регистрация; небольшая development выборка; старые известные held-out заезды; часть стартов недоступна по строгому T+1; ошибки разметки; плотная fused surface может скрывать локальные расхождения; доверие не является вероятностью. Нет экстраполяции через gaps и нет автоматического production freeze.')
    report='\n\n'.join(lines)+'\n';(OUT/'REPORT_CONTACT_MARCHING.md').write_text(report,encoding='utf-8')
    build_gallery(rows,s,examples,las,report)
    save(OUT/'REPORT_FACTS.json',dict(summary=s,benchmark=bench,examples=len(examples),las_sets=len(las),report_lines=len(report.splitlines())))

def build_gallery(rows,s,examples,las,report):
    tags={(e['row']['run'],e['row']['start_frame']):e['tags'] for e in examples};export={(r['run'],r['start_frame']):r for r in las};cards=[]
    translations={'best':'Лучшие','median':'Средние','early':'Ранние отказы','orientation':'Ориентация','sparsity':'Мало точек','wrong':'Проверка больших расхождений','curve':'Кривые','grade':'Уклон','gtgap':'Разрыв GT','gtconflict':'Seed / разметка: конфликт','las':'LAS','all':'Все доступные старты','representative':'Разобранные примеры'}
    for r in sorted([r for r in rows if r['seed_available']],key=lambda r:-(r.get('continuous_reach_5cm') or 0)):
        run=r['run'];i=r['start_frame'];k=key(run,i);tt=list(tags.get((run,i),[]));extra=export.get((run,i));representative=bool(tt)
        if representative:tt.append('representative')
        if extra:tt.append('las')
        if r['terminal_evaluation_reason']=='SENSOR_SPARSITY':tt.append('sparsity')
        if r['terminal_evaluation_reason']=='ORIENTATION_FAILURE':tt.append('orientation')
        if r.get('wrong_structure_suspected'):tt.append('wrong')
        if r['terminal_evaluation_reason'] in ('CR_GT_GAP','RUN_END','CR_SIDE_CHANGE_SUSPECTED','GT_UNAVAILABLE_AHEAD'):tt.append('gtgap')
        frame=frames(run)[i]['file'];links=f'<a href="failures/{k}.json">Диагностика JSON</a> · <a href="heldout/{k}/prediction.json">Все шаги</a>'
        if representative:links+=f' · <a href="gallery/{k}_3d.png">3D</a> · <a href="gallery/{k}_bootstrap.png">Весь seed 0–8 м</a>'
        if r.get('wrong_structure_suspected') or r.get('raw_wrong_structure_trigger'):links+=f' · <a href="gallery/{k}_first_error_with_gt.png">Первый проблемный срез</a>'
        if (OUT/'animations'/(k+'.gif')).exists():links+=f' · <a href="animations/{k}.gif">Анимация</a>'
        if extra:links+=' · '+' · '.join(f'<a href="{extra[f]}">{label}</a>' for f,label in (('overlay','LAS overlay'),('predicted_map','CR map'),('predicted_sensorT','CR sensor T')))
        cards.append(f'<article class="card" data-tags="{" ".join(tt)}" data-search="{html.escape(run+" "+frame)}"><div class="cardhead"><h3>{html.escape(run)} / {frame}</h3><b>≤5 см: {num(r.get("continuous_reach_5cm"))} м</b></div><p>Принято до {num(r.get("max_accepted_range"))} м · шагов {r["march_steps"]} · STOP: {html.escape(r["stop_reason"])}<br>Анализ: {html.escape(r["terminal_evaluation_reason"])}</p><div class="pair"><a href="failures/{k}_without_gt.png"><img loading="lazy" src="failures/{k}_without_gt.png" alt="Последний срез без GT"></a><a href="failures/{k}_with_gt.png"><img loading="lazy" src="failures/{k}_with_gt.png" alt="Последний срез с GT"></a></div><p>{links}</p></article>')
    options=''.join(f'<option value="{k}">{v}</option>' for k,v in translations.items());options=options.replace('value="representative"','value="representative" selected')
    content='''<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>STEP 5 · Contact rail marching</title><style>
    :root{color-scheme:light}body{margin:0;background:#edf2f5;color:#142d3d;font:16px/1.5 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:30px}h1{font-size:32px;line-height:1.2}h2{margin-top:32px}a{color:#066c86}.intro,.card,details{background:white;border:1px solid #d6e1e7;border-radius:12px;padding:22px;margin:18px 0}.metrics{display:flex;gap:16px;flex-wrap:wrap}.metric{flex:1;min-width:180px;background:#eff7f8;padding:16px;border-radius:8px}.metric b{font-size:28px;display:block}.controls{position:sticky;top:0;background:#e6eff4ed;backdrop-filter:blur(8px);padding:16px;z-index:10;display:flex;gap:12px;flex-wrap:wrap}select,input{padding:9px;font:inherit;border:1px solid #9cb5c2;border-radius:6px}.cardhead{display:flex;gap:15px;align-items:center;justify-content:space-between}.cardhead h3{font-size:17px;overflow-wrap:anywhere}.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}.pair img,.overview{width:100%;height:auto}.muted{color:#57707e}.hidden{display:none}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:8px;border-bottom:1px solid #d8e4e8;text-align:left}pre{white-space:pre-wrap;font:14px/1.55 system-ui;overflow-wrap:anywhere}@media(max-width:720px){main{padding:12px}.pair{grid-template-columns:1fr}.cardhead{display:block}}
    </style><main><h1>Продолжение контактного рельса<br>по одному облаку LiDAR</h1>'''
    content+=f'<section class="intro"><p><b>STEP 5 · исследование завершено.</b> Классы и будущие облака используются только после поиска. Сам алгоритм продолжает CR по реальным точкам T и останавливается при неподтверждённом участке.</p><div class="metrics"><div class="metric"><b>{s["seed_available"]} / {s["total_starts"]}</b>bootstrap доступен</div><div class="metric"><b>{num(s["reach"]["5"].get("p50"))} м</b>медианная непрерывная дальность ≤5 см</div><div class="metric"><b>{num(s["reach"]["5"].get("p90"))} м</b>p90 дальности</div></div><p class="muted">Основные метрики: {s["meaningful_evaluated"]} стартов с доступным будущим эталоном. Регистрация и разметка не идеальны; это не утверждение геодезической точности.</p><p><a href="REPORT_CONTACT_MARCHING.md">Полный отчёт</a> · <a href="per_start_frame.csv">Все старты</a> · <a href="per_step.csv">Все шаги</a> · <a href="per_predicted_point.csv.gz">Все найденные точки</a> · <a href="las_exports.csv">LAS</a> · <a href="freeze.json">Протокол фиксации</a></p></section>'
    content+='<img class="overview" src="gallery/reach.png" alt="Непрерывная корректная дальность"><details><summary>Полный текст научного отчёта</summary><pre>'+html.escape(report)+'</pre></details>'
    content+=f'<div class="controls"><select id="filter">{options}</select><input id="search" placeholder="Заезд или имя LAS"><span id="count"></span></div>'+''.join(cards)
    content+='''<script>const f=document.getElementById('filter'),q=document.getElementById('search'),cards=[...document.querySelectorAll('.card')];function update(){let n=0;for(const c of cards){const ok=(f.value==='all'||c.dataset.tags.split(' ').includes(f.value))&&c.dataset.search.toLowerCase().includes(q.value.toLowerCase());c.classList.toggle('hidden',!ok);n+=ok}document.getElementById('count').textContent=n+' примеров'}f.onchange=update;q.oninput=update;update();</script></main></html>'''
    (OUT/'gallery.html').write_text(content,encoding='utf-8')

if __name__=='__main__':main()
