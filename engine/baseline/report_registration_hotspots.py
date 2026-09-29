"""Post-run causality/equality checks and worst-case bottleneck report."""
import sys,json,runpy,html,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'.runtime'))
import numpy as np
name='INPUT5HZ_SOLVE2_REGOPT'
sys.argv=[str(ROOT/'COPY_MAIN/report_direction.py'),name]
runpy.run_path(sys.argv[0],run_name='__main__')
dest=ROOT/'COPY_MAIN/results'/name;p=dest/'payload'
base=ROOT/'COPY_MAIN/results/INPUT5HZ_SOLVE2_POSE1_ASYNC/payload'
def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def lines(path):return [json.loads(x) for x in path.read_text().splitlines()]
def stats(a):return dict(n=len(a),median=float(np.median(a)),p95=float(np.quantile(a,.95)),maximum=float(max(a)))
a=lines(p/'arrivals.jsonl');b=lines(base/'arrivals.jsonl');s=load(p/'SUMMARY.json');old=load(base/'SUMMARY.json')
for x,y in zip(load(p/'direction_poses.json'),load(base/'direction_poses.json')):assert x['pose']==y['pose']
jobs=load(p/'solves.json');oldjobs={r['frame']:r for r in load(base/'solves.json')}
assert set(oldjobs)=={j['frame'] for j in jobs}
exact=0
for j in jobs:
    i=j['frame'];assert j['final_geometry_available']==oldjobs[i]['final_geometry_available']
    if not j['final_geometry_available']:continue
    for name in ('prediction.npz','point_provenance.npz'):
        with np.load(p/'solves'/f'{i:06d}'/name) as l,np.load(base/'solves'/f'{i:06d}'/name) as r:
            assert set(l.files)==set(r.files)
            for k in l.files:np.testing.assert_array_equal(l[k],r[k])
    exact+=1
profile=load(ROOT/'COPY_MAIN/results/REGISTRATION_PARALLEL/PROFILE.json')
trial=profile['recovery_trials'];nn_share=sum(t['nn_ms'] for t in trial)/sum(t['ms'] for t in trial)
def diagnostic(rows,folder):
    jobs=load(folder/'solves.json');io=load(folder/'COPY_IO_TIMINGS.json')
    pubs=load(folder/'publications.json')
    active=[(r['raw_deadline']+r['deadline_ms']/1000-rows[r['snapshot']['source_frame']]['raw_deadline'])*1000 for r in rows if r['snapshot']]
    return dict(read_start_lag_ms=stats([r['producer_lateness_ms'] for r in rows]),
                ready_lag_ms=stats([r['deadline_ms'] for r in rows]),
                late_before_read_over_200ms=[r['source_frame_index'] for r in rows if r['producer_lateness_ms']>200],
                geometry_queue_ms=stats([(j['started']-j['requested'])*1000 for j in jobs]),
                export_queue_ms=stats([r['ms'] for r in io if r['operation']=='queue_solve_wait']),
                last_available_model_age_ms=stats(active),
                publication_gap_ms=stats([(y['clock']-x['clock'])*1000 for x,y in zip(pubs,pubs[1:])]),
                slowest_inputs=[{k:r[k] for k in ('source_frame_index','producer_ms','producer_lateness_ms','registration_ms','deadline_ms')} for r in sorted(rows,key=lambda r:r['producer_ms'],reverse=True)[:12]])
before=diagnostic(b,base);after=diagnostic(a,p)
verification=load(dest/'VERIFICATION.json')
verification.update(exact_vs_async=dict(main_poses=755,direction_poses=377,predictions=exact,provenance=exact),bottlenecks_before=before,bottlenecks_after=after,hard_realtime_qualified=False)
(dest/'VERIFICATION.json').write_text(json.dumps(verification,indent=2)+'\n')
timings=load(p/'STAGES_MS.json')['timings'];oldtimes=load(base/'STAGES_MS.json')['timings']
with (dest/'STAGES_MS.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['stage','count','median_ms','p95_ms','max_ms'])
    for k,v in timings.items():w.writerow([k,v['n'],v['median_ms'],v['p95_ms'],v['max_ms']])
bench=[]
for bounded,reuse,parallel in ((False,False,False),(False,True,False),(True,True,False),(True,True,True)):
    values=[x['ms'] for x in profile['recovery_bench'] if (x['bounded'],x['reuse'],x['parallel'])==(bounded,reuse,parallel)]
    bench.append((f'граница={bounded}, кеш={reuse}, параллельно={parallel}',stats(values)))
table='\n'.join(f'| {n} | {v["median"]:.1f} | {v["maximum"]:.1f} |' for n,v in bench)
stage_table='\n'.join(f'| {k} | {v["median_ms"]:.2f} | {v["p95_ms"]:.2f} | {v["max_ms"]:.2f} |' for k,v in timings.items())
pub=s['publication_latency_ms'];oldpub=old['publication_latency_ms']
text=f'''# Узкие места и пределы realtime: основная сшивка

**Требование «никакого накопления и догоняния» пока не выполнено.**
Ни средняя длительность записи, ни p95 не являются гарантией максимальной задержки.
Текущая симуляция сохраняет каждый выбранный кадр и после задержки обрабатывает его
позже расписания. Ограниченный transport на один элемент не запрещает читать
просроченные кадры из DB3. Это не настоящий DDS subscriber с политикой latest-only.

## Доказанное узкое место

На исходном кадре 232 после бокса сработало восстановление по пяти начальным положениям.
Все пять попыток выполнили по 60 итераций: суммарно 300. В каждой использовалось
10872 точки текущего скана и 7348 точек опоры. Поиск ближайших соседей занял
{nn_share*100:.1f}% времени попыток. Переиспользование дерева убирает лишь несколько мс.
У обычных кадров поиск соседей тоже занимает значительную часть сшивки.

Очередь геометрии в прежнем прогоне: максимум {before['geometry_queue_ms']['maximum']:.2f} мс;
ожидание очереди диагностической записи: {before['export_queue_ms']['maximum']:.3f} мс.
Они не объясняют секундный провал. Основная блокировка находится ДО отправки кадра
в геометрию, внутри последовательного чтения/регистрации.

## Проверенные ускорения

1. Ограничение NN-поиска тем же расстоянием, по которому исходный ICP уже отбрасывал
   соответствия. Это точный поиск внутри прежнего допуска, без приближённого NN.
2. Повторное использование поискового дерева неизменной опоры.
3. Пять прежних гипотез восстановления считаются параллельно. Порядок проверки,
   выбор результата, пороги и 60 итераций сохранены. Неудачные гипотезы не заменены
   молчаливым согласием с прогнозом движения.

Изолированное восстановление, три повтора каждого варианта; это НЕ live-замер:

| Вариант | Медиана, мс | Максимум, мс |
|---|---:|---:|
{table}

## Полный причинный прогон

| Метрика | До ускорения | После |
|---|---:|---:|
| Публикация: медиана, мс | {oldpub['median']:.1f} | {pub['median']:.1f} |
| Публикация: p95, мс | {oldpub['p95']:.1f} | {pub['p95']:.1f} |
| Публикация: максимум, мс | {oldpub['max']:.1f} | {pub['max']:.1f} |
| Восстановление, мс | {oldtimes['registration_recover_endpoint_ms']['max_ms']:.1f} | {timings['registration_recover_endpoint_ms']['max_ms']:.1f} |
| Максимальное опоздание начала чтения, мс | {before['read_start_lag_ms']['maximum']:.1f} | {after['read_start_lag_ms']['maximum']:.1f} |
| Кадров с началом чтения позже 200 мс | {len(before['late_before_read_over_200ms'])} | {len(after['late_before_read_over_200ms'])} |
| Полный прогон, с | {old['wall_seconds']:.3f} | {s['wall_seconds']:.3f} |

Длительность записи {s['original_recording_seconds']:.3f} с. Вход 5 Гц, расчёт рельсов
2,5 Гц, направление из исходного T+1; будущие облака не поступают в детекторы.
Все 755 основных поз, 377 дополнительных поз, {exact} прогнозов и provenance совпали
точно с предыдущим вариантом. Исходные кадры не пропускались дополнительно; отсутствие
надёжной позы на кадрах 218–230 осталось явно обозначенным. Никакой интерполяции через
бокс или использования ранее сохранённых поз при inference не было.

Последняя доступная модель при обработке нового входа: медиана возраста
{after['last_available_model_age_ms']['median']:.1f} мс, p95
{after['last_available_model_age_ms']['p95']:.1f} мс, максимум
{after['last_available_model_age_ms']['maximum']:.1f} мс. Это возраст исходного измерения,
а не ошибка координат. Период обновления модели и её возраст — разные величины.
Максимальный интервал между публикациями {after['publication_gap_ms']['maximum']:.1f} мс,
в том числе отсутствие пригодной геометрии при перекрытии обзора.

## Что нужно для запрета догоняния

* Приём свежих облаков по исходному расписанию должен быть независим от ICP. Для
  ожидания тяжёлого расчёта — один заменяемый последний кадр вместо FIFO.
* Нужны deadline и явное состояние просрочки. Результат, опоздавший к своему сроку,
  не становится актуальным от одного переноса в новые координаты.
* Длительное восстановление нужно выполнять отдельно; пока поза не подтверждена,
  нельзя изображать её достоверной или незаметно продолжать старый путь.
* Проверка попаданий свежих точек в габарит должна работать независимо от поиска
  рельсов раз в 400 мс. Сейчас окраска в просмотрщике — послепрогонная проверка,
  её нельзя считать измеренным онлайн-детектором препятствий.

Latest-only убирает накопление, но ценой явно учтённых необработанных кадров;
он сам по себе не гарантирует непрерывное обнаружение. Тяжёлый вызов нельзя надёжно
остановить отменой уже запущенного Python Future: нужен контроль внутри вычислений
или изолированный вычислительный процесс. Этот механизм ещё не реализован и не
выдаётся за результат данного теста.

Сначала следует ускорять численное ядро поиска соседей/ICP и отдельно измерять
свежий габаритный контроль; затем проверять deadline, число пропусков и периоды
недоступности под нагрузкой. Наличие очереди нельзя скрывать уменьшением числа
публикаций или исключением кадров с препятствием из отчёта.

## Воспроизводимость и ограничения

Оптимизация включается `COPY_REGISTRATION_ACCEL=parallel`; исходные файлы
COPY_MAIN/registration и замороженные стадии не менялись. Изменения находятся
в COPY_MAIN/adapter/registration_acceleration.py и подключении адаптера.
Три повтора микротеста, один полный прогон. База копируется в локальную Linux FS,
копирование исключено из таймера. Контейнер запускается с --rm, LAS не создаются.
Имеющаяся запись не содержит одометрии/IMU; независимого GT точности здесь нет.

## Все этапы, мс

Вложенные и параллельные таймеры нельзя складывать.

| Этап | Медиана | p95 | Максимум |
|---|---:|---:|---:|
{stage_table}
'''
(dest/'REPORT.md').write_text(text,encoding='utf-8')
chart=''
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(12,7),constrained_layout=True)
    for rows,label in ((b,'Before'),(a,'After')):
        t=[(r['raw_deadline']-rows[0]['raw_deadline']) for r in rows]
        for ax in axes:ax.plot(t,[r['producer_lateness_ms'] for r in rows],label=label,linewidth=1)
    axes[0].set_title('Delay before reading the next cloud (queue catch-up)');axes[1].set_xlim(20,32)
    for ax in axes:ax.set_xlabel('Recording time, s');ax.set_ylabel('Lateness, ms');ax.legend();ax.grid(alpha=.25)
    fig.savefig(dest/'queue_delay.png',dpi=150);plt.close(fig)
    chart='<img style="width:100%" src="queue_delay.png" alt="Задержки начала чтения до и после" />'
except ImportError:pass
(dest/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Узкие места realtime</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:20px;background:#f4f7fb;color:#14283d}pre{white-space:pre-wrap;line-height:1.55;background:white;padding:24px;border-radius:12px}</style>'+chart+'<pre>'+html.escape(text)+'</pre>',encoding='utf-8')
print(json.dumps(dict(exact=exact,before=before,after=after,publication=pub),ensure_ascii=False,indent=2))
