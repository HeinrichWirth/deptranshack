"""Final auditable Russian report. Never upgrades a failed quality gate."""
from . import ROOT,OUT
import long_common as lc
import numpy as np
import csv,html,json,time,hashlib,platform
from collections import Counter


def rows(path):return lc.csv_read(path) if path.exists() else []
def q(v,p):
    a=np.array([float(x) for x in v if x is not None and str(x) not in ('','nan','None')]);a=a[np.isfinite(a)]
    return None if not len(a) else float(np.percentile(a,p))
def fmt(v,d=2):return '—' if v is None else f'{v:.{d}f}'
def table(headers,data):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in data)+'\n'


def main():
    selected=lc.load(OUT/'SELECTION_LOCK.json');config=selected['config'];dev=OUT/'development/combined_r8_o8';bench=OUT/'benchmark/locked_r8_o8'
    dq=lc.load(dev/'QUALITY.json');bq=lc.load(bench/'QUALITY.json');clean=lc.load(OUT/'clean_timing.json')
    devtime=[r for p in dev.glob('*/timings.json') for r in lc.load(p)];btime=[r for p in bench.glob('*/timings.json') for r in lc.load(p)];alltime=devtime+btime
    clean_v=[r for r in clean if r['backend']=='v2'];clean_r=[r for r in clean if r['backend']=='reference']
    normal=[r for r in clean_v if r['status'].startswith('INCREMENTAL')];repair=[r for r in clean_v if r['status']=='INCREMENTAL_REPAIR'];extend=[r for r in clean_v if r['status']=='INCREMENTAL_REPAIR_EXTEND']
    reuse=[r for r in clean_v if r['status']=='INCREMENTAL_REUSE'];fallback=[r for r in alltime if r['fallback']]
    performance=dict(normal_median_ms=q([r['total_ms'] for r in normal],50),normal_p95_ms=q([r['total_ms'] for r in normal],95),
        reuse_median_ms=q([r['total_ms'] for r in reuse],50),repair_median_ms=q([r['total_ms'] for r in repair],50),repair_extend_median_ms=q([r['total_ms'] for r in extend],50),
        fallback_fraction=len(fallback)/len(alltime),normal_samples=len(normal),clean_frames=len(clean),target_ms=150,acceptable_ms=250)
    performance['good_target_pass']=performance['normal_median_ms'] is not None and performance['normal_median_ms']<150
    performance['pass']=performance['normal_median_ms'] is not None and performance['normal_median_ms']<250
    performance['tier']='good' if performance['good_target_pass'] else 'acceptable' if performance['pass'] else 'miss'
    # Development fail is terminal even if a held-out aggregate happens to improve.
    quality=dict(pass_=bool(dq['pass_primary_gates'] and bq['pass_primary_gates']),development=dq,benchmark=bq,criteria=lc.load(ROOT/'MVP/c4_v2/plan.json')['gates'],
        approximate_GT=True,benchmark_parameters_changed=False,production_accepted=False)
    quality['pass']=quality.pop('pass_');lc.save(OUT/'quality_gate.json',quality);lc.save(OUT/'performance_gate.json',performance)
    lc.csv_write(OUT/'development_quality.csv',dq['summary']);lc.csv_write(OUT/'benchmark_quality.csv',bq['summary'])
    for name,predicate in [('incremental_timing',lambda r:r['status'].startswith('INCREMENTAL')),('full_timing',lambda r:r['status'] in ('COLD_FULL_BOOTSTRAP','FALLBACK_FULL','STATE_RESET_POSE_GAP')),('repair_timing',lambda r:'REPAIR' in r['status']),('extension_timing',lambda r:'EXTEND' in r['status']),('fallbacks',lambda r:r['fallback'])]:
        lc.csv_write(OUT/(name+'.csv'),[{k:v for k,v in r.items() if k!='native_stats'} for r in alltime if predicate(r)])
    lc.csv_write(OUT/'persistent_state_stats.csv',[{k:v for k,v in r.items() if k!='native_stats'} for r in alltime])
    lc.csv_write(OUT/'cpu_utilization.csv',[dict(backend=r['backend'],run=r['run'],frame=r['frame'],total_ms=r['total_ms'],average_active_cores=r.get('average_active_cores',r.get('active_cores'))) for r in clean])
    seams=[r for folder in (dev,bench) for p in folder.glob('*/seams.csv') for r in rows(p)];lc.csv_write(OUT/'seam_metrics.csv',seams)
    drift=[];worst=[]
    for label,root,timings in [('development',dev,devtime),('benchmark',bench,btime)]:
        lookup={(r['run'],r['frame']):r for r in timings};stationrows=rows(root/'matched_stations.csv');group={}
        for r in stationrows:group.setdefault((r['run'],int(r['frame'])),[]).append(r)
        for key,rr in group.items():
            tr=lookup[key];value=dict(group=label,run=key[0],frame=key[1],origin_frame=tr['origin_frame'],frames_since_cold=tr['frames_since_cold'],meters_since_cold=tr['meters_since_cold'],status=tr['status'],
                far_reference_p95=q([r['far_reference'] for r in rr],95),far_variant_p95=q([r['far_variant'] for r in rr],95),
                far_reference_max=q([r['far_reference'] for r in rr],100),far_variant_max=q([r['far_variant'] for r in rr],100),cr_disagreement_p95=q([r['cr_change'] for r in rr],95))
            value['delta_p95']=value['far_variant_p95']-value['far_reference_p95'];drift.append(value);worst.append(value)
    worst.sort(key=lambda r:r['delta_p95'],reverse=True);lc.csv_write(OUT/'drift_metrics.csv',drift);lc.csv_write(OUT/'worst_cases.csv',worst)
    horizon=[]
    for p in (OUT/'async').glob('*/result.json'):
        new=lc.load(p);run=p.parent.name;oldpath=ROOT/'results_performance_final2/async/batch'/run/'result.json'
        old=lc.load(oldpath) if oldpath.exists() else None
        for name,data in [('reference_previous_sprint',old),('v2',new)]:
            if data is None:continue
            rr=data['frames'];available=[r for r in rr if r['state_available']];positive=sum((r.get('remaining_horizon') or 0)>0 for r in rr)
            horizon.append(dict(run=run,backend=name,raw_frames=len(rr),positive_horizon_fraction=positive/len(rr),availability_fraction=len(available)/len(rr),
                age_median_s=q([r.get('geometry_age_seconds') for r in available],50),age_p95_s=q([r.get('geometry_age_seconds') for r in available],95),
                age_distance_median_m=q([r.get('distance_since_geometry_source') for r in available],50),remaining_horizon_median_m=q([r.get('remaining_horizon') for r in available],50),
                raw_service_p95_ms=q([r['service_seconds']*1000 for r in rr],95),pending_max=data['max_pending'],solves=len(data['solves'])))
    lc.csv_write(OUT/'horizon_metrics.csv',horizon)
    live=lc.load(OUT/'live_v2/RUN_COMPLETE.json')
    live_ms=[r['service_seconds']*1000 for r in live['frames']]
    performance.update(raw_live_frames=len(live_ms),raw_live_p95_ms=q(live_ms,95),raw_live_max_ms=q(live_ms,100),raw_live_over_100ms=sum(x>100 for x in live_ms),raw_live_pending_max=live['max_pending'])
    lc.save(OUT/'performance_gate.json',performance)
    for name,data in [('development',dq),('benchmark',bq)]:
        horizons_a=[r['reference_horizon'] for r in data['cases']];horizons_b=[r['variant_horizon'] for r in data['cases']]
        quality[name+'_horizon_gate']=dict(reference_median=q(horizons_a,50),v2_median=q(horizons_b,50),reference_p90=q(horizons_a,90),v2_p90=q(horizons_b,90),
            median_pass=q(horizons_a,50)-q(horizons_b,50)<=2,p90_pass=q(horizons_a,90)-q(horizons_b,90)<=2)
    quality['async_availability_comparison']=horizon
    lc.save(OUT/'quality_gate.json',quality)
    diagnostic=rows(OUT/'failure_diagnosis.csv')
    diagnostictable=table(['Запись / кадр','Reference / persistent / full native, far p95, см','Δ CR / Δ far относительно CR, см'],[
        [r['run']+' / '+r['frame'],' / '.join(fmt(float(r[k])*100) for k in ('reference_far_p95','persistent_far_p95','native_full_far_p95')),
         fmt(float(r['cr_disagreement_p95'])*100)+' / '+fmt(float(r['relative_far_cr_disagreement_p95'])*100)] for r in diagnostic if r.get('stations')])
    stats=Counter(r['status'] for r in alltime);fail_reasons=Counter(r['reason'] for r in alltime if r['fallback'] or r['status']=='UPDATE_FAILED_REUSE_PREVIOUS')
    reference_c4=[r['timing'].get('T_C4_MARCHING',0)*1000 for r in clean_r]
    intro=table(['Показатель','Результат'],[
        ['Reference full solve, median / p95, мс',f"{fmt(q([r['total_ms'] for r in clean_r],50))} / {fmt(q([r['total_ms'] for r in clean_r],95))}"],
        ['Reference C4, median / p95, мс',f'{fmt(q(reference_c4,50))} / {fmt(q(reference_c4,95))}'],
        ['V2 успешный cold, median / p95, мс',f"{fmt(q([r['total_ms'] for r in clean_v if r['status']=='COLD_FULL_BOOTSTRAP' and r['available']],50))} / {fmt(q([r['total_ms'] for r in clean_v if r['status']=='COLD_FULL_BOOTSTRAP' and r['available']],95))}"],
        ['V2 принятый incremental, median / p95, мс',f"{fmt(performance['normal_median_ms'])} / {fmt(performance['normal_p95_ms'])}"],
        ['V2 reuse / repair / repair+extend, median, мс',f"{fmt(performance['reuse_median_ms'])} / {fmt(performance['repair_median_ms'])} / {fmt(performance['repair_extend_median_ms'])}"],
        ['Full fallback, все последовательные кадры',f"{len(fallback)} / {len(alltime)} = {100*performance['fallback_fraction']:.2f}%"],
        ['Потоки / средние активные ядра V2',f"{config['threads']} / {fmt(q([r.get('average_active_cores',r.get('active_cores')) for r in clean_v],50))}"],
        ['Development / benchmark, полные кадры',f'{len(devtime)} / {len(btime)}'],
        ['Качество / скорость <150 мс',f"{'PASS' if quality['pass'] else 'FAIL'} / {'PASS' if performance['good_target_pass'] else 'FAIL'}"],
        ['Допустимый бюджет median <250 мс','PASS' if performance['pass'] else 'FAIL'],
    ])
    qualitytables=''
    for name,data in [('Development',dq),('Независимый benchmark',bq)]:
        qualitytables+=f'### {name}\n\n'+table(['Дальность, м','Общих станций / стартов','Far p95 reference, см','Far p95 V2, см','Изменение, см','Gate'],[
            [f"{r['lo']}–{r['hi']}",f"{r['stations']} / {r['starts']}",fmt(None if r['far_reference_p95'] is None else r['far_reference_p95']*100),fmt(None if r['far_variant_p95'] is None else r['far_variant_p95']*100),fmt(None if r['far_p95_delta_m'] is None else r['far_p95_delta_m']*100),'PASS' if r['pass'] else 'FAIL / недостаточно данных'] for r in data['summary'][:3]])+'\n'
        qualitytables+=table(['Катастрофические ошибки / покрытие','Reference','V2','Новых / потеря'],[[r['metric'],r.get('reference',''),r.get('variant',''),r.get('value',r.get('loss_pp',''))] for r in data['summary'][3:]])+'\n'
    worsttable=table(['Запись / кадр','После cold, м','Far p95 reference → V2, см','Δ CR p95, см','Состояние'],[
        [f"{r['run']} / {r['frame']}",fmt(r['meters_since_cold']),f"{fmt(r['far_reference_p95']*100)} → {fmt(r['far_variant_p95']*100)}",fmt(None if r['cr_disagreement_p95'] is None else r['cr_disagreement_p95']*100),r['status']] for r in worst[:12]])
    saved_work=[r for r in alltime if r['status'].startswith('INCREMENTAL')];den=sum(r['reused_m']+r['recomputed_m'] for r in saved_work);reduction=None if not den else sum(r['reused_m'] for r in saved_work)/den
    text=f'''# C4 V2 — итоговый отчёт

**Вердикт: D. Persistent V2 не принят в production. По умолчанию сохранён reference `native_batch_spatial`.**

Ускорение C++ candidate solver подтверждено отдельно. Сохранение карты увеличивает доступный горизонт, но нарушает заранее заданные ограничения на ухудшение точности. Более длинная выданная линия не означает более точный прогноз. Файл `FINAL_C4_V2.freeze` не создаётся.

{intro}

## Что реализовано

Новая папка `MVP/c4_v2`. Никакие файлы frozen STEP1, STEP2, C4, C4_SMOOTH, RAW seed adapter, STEP6, final_pipeline и прошлых performance-этапов не изменены. 435 проверок контрольных сумм для 256 уникальных файлов сохранены в `frozen_integrity.json`.

Полный C++17 marching: хранение причинной истории в MAP, индекс ROI, текущая PCA, перенос локальной системы, точный nearest-neighbor, шаблонные residuals, Cauchy objective, собственная конечная разность, ограниченный LM с явным решением 2×2, оценка кандидатов, beam=3 и подтверждение tentative. Внутри обновления нет SciPy, Python callbacks и Python dictionaries. GIL отпущен на время native marching. Python выполняет чтение LAS, управление картой, fallback и неизменённые downstream этапы.

Постоянный пул живёт весь срок engine. Независимые seed/candidate fits выполняются параллельно; зависимые шаги marching остаются последовательными. Предопределённые слоты результатов сохраняют порядок независимо от потоков. Портативная сборка использует C++17, оптимизацию компилятора без fast-math и без `-march=native`.

На обычном обновлении STEP1/STEP2 не вызываются. Мapped prefix сохраняется, текущие точки подтверждают overlap, хвост заменяется только при подтверждённых новых наблюдениях. Исторические point keys сохраняют исходный frame/row. Старые точки карты не становятся точками текущего кадра. Разрыв pose/time обнуляет состояние. Неудача обновления сохраняет прежнюю геометрию и явно помечается как stale; async worker не публикует её как свежую.

## Протокол и выбор параметров

Пять полных development-записей: roundT_squareT_pressureGate_squareT, roundT_pressureGate_roundT, squareT_platform_squareT_switch, new_data_part_01a2, new_data_part_01d1. Старые 200 стартов принадлежат этим же записям и здесь считаются development, не независимым benchmark. Benchmark — оставшиеся шесть записей; состояние создаётся с нулевого кадра отдельно в каждой записи. Всего в выбранном полном development-прогоне {len(devtime)} кадров, в benchmark {len(btime)}.

Проверены LM caps 4/6/8/12 и GRID_LM 5×5/7×7/9×9 на 91 непустом сохранённом fit. 12 итераций LM сохраняли выбранный пик в пределах 6 мм в 66/91 случаях; 4 итерации — в 13/91. Это диагностическое сравнение, а не критерий production. Полный native solver отдельно проверен на 200 development-стартах: downstream gate пройден, но измерений 75–100 м мало.

Ремонт 4/6/8/12 м и overlap 4/8/12 м проверены ограниченным причинным screen по пяти записям; два лучших полноценных варианта 6/8 и 8/8 затем прогнаны от начала до конца. Оба нарушили quality gates. Выбран 8/8: меньше дополнительных случаев >20 см. Это фиксированный диагностический кандидат, а не скрытое разрешение на production.

Итоговая конфигурация: `{json.dumps(config)}`. В `SELECTION_LOCK.json` зафиксированы время, контрольная сумма протокола и причина выбора до benchmark. На benchmark настройки не менялись. Пороги качества: +1 см far p95 в каждом диапазоне; максимум один новый старт >20 см; ни одного нового >50 см, wrong-side или crossing; потеря покрытия ≥30/50/75 м не более 2 п.п. Недостаток общих измерений не засчитывается как доказательство качества.

## Качество на одинаковых срезах

Все методы пересечены с одними и теми же плоскостями reference. Ошибка считается относительно одного post-inference GT. Не смешиваются улучшение покрытия и точность на общих станциях. Полные near/far/center/contact показатели лежат в `development_quality.csv` и `benchmark_quality.csv`. GT получен из реальной разметки и сам неточен; цифры — воспроизводимые сравнительные измерения относительно этого эталона, а не абсолютная точность лидарной системы. Contact GT доступен не на всех выбранных стартах; число станций указано отдельно. Δ CR означает расхождение прогнозов, не ошибку относительно GT.

{qualitytables}

## Худшие случаи и причина отказа

{worsttable}

Сохранение карты меняет источник ошибки. В full solve STEP1 каждый раз заново измеряет near running rails. В persistent варианте эти детекторы правильно пропускаются, но frozen RAW seed остаётся исходным: его реальные 0–8 м observations не подменяются новыми выдуманными точками. STEP6 продолжает использовать первоначальную относительную геометрию. При движении ошибки исходного seed и зарегистрированной карты оказываются общими для многих следующих кадров. Tail repair исправляет только контактный хвост; он не переоценивает near/far offsets на уже замороженном префиксе.

Это объяснение проверяется сравнением ошибки CR, ошибки пары и расстояния после bootstrap в `drift_metrics.csv` и графиках. Корреляция с пройденным расстоянием сама по себе не доказывает конкретную физическую причину. Регистрационные ошибки и неоднозначность размеченного эталона остаются возможными. Мы не меняем GT и не удаляем неудобные случаи из gate.

Дополнительная проверка на худших held-out случаях: с той же зафиксированной конфигурацией выполнен full native solve без persistence. Выбор этих случаев — диагностика после benchmark, не подбор параметров. Например, roundT_doubleT / 212: reference far p95 8,84 см, persistent 30,61 см, full native 8,33 см; изменение самого CR около 2,82 см, относительного положения far/CR — 23,38 см. Здесь основная регрессия связана с продолжением относительной геометрии от старого seed. На doubleT_platform / 88 преобладает другое: изменение CR около 10,04 см при изменении относительного offset около 3,00 см. Поэтому сводить все отказы к одному solver или только к неточному GT нельзя.

{diagnostictable}

Есть также различие между **неудачным обновлением** и **full fallback**. Первое оставляет карту доступной, но её доказательная свежесть уменьшается. Распределение: {dict(stats)}. Причины остановки/отказа: {dict(fail_reasons)}. Полные строки — `fallbacks.csv` и `persistent_state_stats.csv`.

## Что реально экономится и что остаётся дорогим

Доля пропущенной длины marching на принятых incremental-обновлениях: {fmt(None if reduction is None else reduction*100)}%. Это геометрическая оценка reused/(reused+repair+extension), не процент экономии CPU: начальное 8-метровое подтверждение, чтение кадра и downstream пересчёт остаются.

Native fitting уже не вызывает SciPy. Однако обычный update включает чтение и подготовку текущего LAS, overlap validation, а при ремонте — native ROI/marching, формирование полного численного контракта, C4_SMOOTH и STEP6. Последние frozen функции по-прежнему пересчитывают накопленную геометрию в системе исходного seed. Поэтому стоимость ремонта нельзя оценивать одной лишь длиной нового хвоста. Профили `reference_profile.txt` и `v2_profile.txt` отдельно показывают эти затраты.

Микроbenchmark параллельного batch показывает работу нескольких ядер, но full solve масштабируется слабее: при 1/2/4/8/16 потоках медиана full solve составила примерно 607/544/469/429/432 мс на фиксированном development timing cohort. При 8 потоках среднее число активных ядер полного процесса около 1.21. Причины — последовательные шаги, часто всего 5 задач в текущем шаге, чтение/трансформации и неизменённый Python downstream. Значения и контроль детерминизма находятся в `thread_scaling.csv` и `full_thread_scaling.csv`; это не обещание 8-кратного end-to-end ускорения.

Чистые timing измерения выполнялись последовательно после завершения конкурирующих development jobs. CSV всех полных прогонов дополнительно содержит время под конкурентной нагрузкой и не заменяет чистый benchmark. p95, CPU time, число native fits/evaluations/tasks и Python→native calls сохранены; счётчики engine накопительные, их разности между кадрами дают работу одного update.

`native_counters.csv` отдельно измеряет 135 причинных обновлений: число вызовов marching, ingest новых history frames, validation batches, candidate fits и residual evaluations. Это позволяет проверить, что Python не вызывается из residual loop; число API переходов определяется обновлениями/подгрузкой кадров, а не тысячами residual evaluations.

## Async, возраст геометрии и полезный горизонт

{table(['Запись / backend','Положительный горизонт','Возраст median / p95, с','Остаток median, м','raw service p95, мс'],[[r['run']+' / '+r['backend'],fmt(100*r['positive_horizon_fraction'])+'%',fmt(r['age_median_s'])+' / '+fmt(r['age_p95_s']),fmt(r['remaining_horizon_median_m']),fmt(r['raw_service_p95_ms'])] for r in horizon])}

Один активный worker и один заменяемый pending request: очередь старых кадров не накапливается. Таблица использует фактически измеренные solve durations и event replay по исходным timestamp. Reference full-run — сохранённый async-прогон прошлого sprint; это явно исторический baseline. Отдельный live-тест выполняет реальный background worker с темпом 10 Гц. `COMPUTE_EXPIRED` и `OBSERVATION_SHORT_HORIZON` разделены: первый означает, что геометрия устарела по времени вычисления/прохождения, второй — что уже исходный solve видел лишь около 8 м. Положительный горизонт не отменяет quality FAIL.

Live 10 Гц: {len(live_ms)} кадров, raw-service p95 {fmt(q(live_ms,95))} мс, максимум {fmt(q(live_ms,100))} мс, вызовов дольше 100 мс — {sum(x>100 for x in live_ms)}, максимальный pending — {live['max_pending']}. Это responsiveness потребителя, не 10 Гц полного пересчёта геометрии.

## Стык, дрейф и повреждение хвоста

Подтверждённый опубликованный prefix сохраняется побитово. В `seam_metrics.csv` поперечная невязка стыка measured относительно предыдущей касательной, а tangent jump — угол соседних frames. Эта поперечная величина включает естественную кривизну на шаге между соседними samples; она не является чистой искусственной discontinuity. В `tail_stress.csv` отдельно внесены 1/3/5/10 см в persistent tail выбранной конфигурации и записаны реакции/остаточное расхождение. Если нет свежих точек у хвоста, подтверждение невозможно: карта остаётся stale, а не объявляется исправленной. Проверка на невидимом участке не доказывает восстановление. Предварительный repair=6 stress сохранён отдельно в `tail_stress_r6.csv`.

## Доставка и воспроизведение

`MVP/c4_v2/cli.py` поддерживает `--c4-backend reference|v2` и `C4_BACKEND`. Default — reference. Для v2 отсутствие native-модуля вызывает явную ошибку; автоматического переключения на другой backend нет. `--config results_c4_v2/selected_config.json` воспроизводит выбранный эксперимент. Исходные LAS не изменяются.

Docker: `deptrans-c4-v2:experimental`. Многостадийная переносимая сборка: компилятор только в build stage, runtime наследует зафиксированную среду. Образ не содержит dataset. Windows и Linux unit checks проверяют causal history, bounded fitting, unique support, round-trip координат, local station projection, отсутствие SciPy callback и детерминизм 1/2/4/8/16 потоков. Linux smoke не заменяет полный Linux accuracy benchmark.

Команды и пути находятся в `MVP/c4_v2/README.md`. `MANIFEST.json` содержит SHA-256 исходников и итоговых артефактов. Никаких C4 V3 или скрытого изменения frozen модели не создано.

## Решение A/B/C/D

- A: persistence со старым solver не доказала production speed; короткий control ограничен 100 кадрами каждой из пяти записей.
- B: native solver отдельно ускоряет full solve и прошёл development quality gate; его latency всё ещё выше целевого incremental бюджета.
- C: combined pipeline не принят: quality gates нарушены.
- **D: сохранить рабочий reference. Persistent V2 оставить исследовательским результатом с количественно зафиксированным отказом.**

Production freeze отсутствует намеренно. Новый adapter `final_pipeline_v2`, который делал бы V2 backend по умолчанию, не создаётся при FAIL.
'''
    (OUT/'REPORT_C4_V2_FINAL.md').write_text(text,encoding='utf-8')
    # Small self-contained renderer: no CDN/network dependency.
    lines=text.splitlines();body=[];inside=False;tablelines=[]
    def flush_table():
        nonlocal tablelines
        if not tablelines:return
        body.append('<table>')
        for j,line in enumerate(tablelines):
            if j==1:continue
            tag='th' if j==0 else 'td';cells=[x.strip() for x in line.strip('|').split('|')]
            body.append('<tr>'+''.join(f'<{tag}>'+html.escape(c)+f'</{tag}>' for c in cells)+'</tr>')
        body.append('</table>');tablelines=[]
    for line in lines:
        if line.startswith('|'):tablelines.append(line);continue
        flush_table()
        if line.startswith('#'):
            n=len(line)-len(line.lstrip('#'));body.append(f'<h{n}>'+html.escape(line[n:].strip())+f'</h{n}>')
        elif line:body.append('<p>'+html.escape(line)+'</p>')
    flush_table()
    head='<!doctype html><meta charset="utf-8"><title>C4 V2 — полный отчёт</title><style>body{font:16px/1.55 system-ui;max-width:1300px;margin:36px auto;padding:0 25px;color:#203348;background:#f8fafc}h1,h2,h3{color:#123452}table{border-collapse:collapse;width:100%;font-size:14px;background:white;margin:20px 0}td,th{border:1px solid #ccd7e0;padding:8px;text-align:left}th{background:#dfebf6}a{color:#0565ad}.warning{background:#ffe7df;padding:18px;border-left:6px solid #a32b14}</style><p><a href="gallery.html">Галерея: лучшие, средние, худшие случаи</a> · <a href="REPORT_C4_V2_FINAL.md">Markdown</a></p>'
    (OUT/'REPORT_C4_V2_FINAL.html').write_text(head+'\n'.join(body),encoding='utf-8')
    recommendation='''# FINAL C4 V2 RECOMMENDATION

**D — не переводить persistent C4 V2 в production.**

Код: MVP/c4_v2. Результаты: results_c4_v2.
Рабочий backend по умолчанию: reference → frozen native_batch_spatial.
Новый native solver полезен отдельно, но persistent карта нарушила quality gates.
Ни скорость повторного использования, ни более длинный горизонт не компенсируют ухудшение точности.
FINAL_C4_V2.freeze не создан. Настройки benchmark не изменялись после SELECTION_LOCK.json.
Подробные цифры, ограничения GT и примеры: REPORT_C4_V2_FINAL.html и gallery.html.
'''
    (OUT/'FINAL_C4_V2_RECOMMENDATION.md').write_text(recommendation,encoding='utf-8')
    print('REPORT',performance,'quality',quality['pass'],flush=True)

if __name__=='__main__':main()
