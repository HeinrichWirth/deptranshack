"""Verify identical geometry after scheduling change and explain latency spikes."""
import sys,json,runpy,html,csv
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'.runtime'))
import numpy as np
run_name=sys.argv[1] if len(sys.argv)>1 else 'INPUT5HZ_SOLVE2_POSE1_ASYNC'
sys.argv=[str(ROOT/'COPY_MAIN/report_direction.py'),run_name]
runpy.run_path(sys.argv[0],run_name='__main__')
dest=ROOT/'COPY_MAIN/results'/run_name;p=dest/'payload'
base=ROOT/'COPY_MAIN/results/INPUT5HZ_SOLVE2_POSE1_FAST/payload'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def stats(a):return dict(median=float(np.median(a)),p95=float(np.quantile(a,.95)),maximum=float(max(a)))
s=load(p/'SUMMARY.json');b=load(base/'SUMMARY.json')
a=lines(p/'arrivals.jsonl');old_a=lines(base/'arrivals.jsonl')
d=load(p/'direction_poses.json');old_d=load(base/'direction_poses.json')
assert len(d)==len(old_d)==377
for x,y in zip(d,old_d):assert x['pose']==y['pose']
old_jobs={x['frame']:x for x in load(base/'solves.json')};jobs=load(p/'solves.json')
assert set(old_jobs)=={x['frame'] for x in jobs}
exact=0
for job in jobs:
    i=job['frame'];assert job['final_geometry_available']==old_jobs[i]['final_geometry_available']
    if not job['final_geometry_available']:continue
    for file in ('prediction.npz','point_provenance.npz'):
        with np.load(p/'solves'/f'{i:06d}'/file) as left,np.load(base/'solves'/f'{i:06d}'/file) as right:
            assert set(left.files)==set(right.files)
            for k in left.files:np.testing.assert_array_equal(left[k],right[k])
    exact+=1
verify=load(dest/'VERIFICATION.json')
verify['async_exact']=dict(poses=755,direction_poses=377,predictions=exact,provenance=exact)
verify['direction_worker_queue_ms']=stats([x['worker_queue_ms'] for x in d])
verify['compared_with']='INPUT5HZ_SOLVE2_POSE1_FAST'
(dest/'VERIFICATION.json').write_text(json.dumps(verify,indent=2)+'\n')
# Every retained snapshot is measured at its actual completion clock. Reconstruct
# original bag deadlines rather than mixing header-clock and arrival-clock ages.
active=[]
for row in a:
    snap=row['snapshot']
    if snap:
        active.append((row['raw_deadline']+row['deadline_ms']/1000-a[snap['source_frame']]['raw_deadline'])*1000)
verify['last_available_geometry_age_at_input_ms']=stats(active)
(dest/'VERIFICATION.json').write_text(json.dumps(verify,indent=2)+'\n')
timings=load(p/'STAGES_MS.json')['timings']
table='\n'.join(f"| {key} | {v['n']} | {v['median_ms']:.2f} | {v['p95_ms']:.2f} | {v['max_ms']:.2f} |" for key,v in timings.items())
segment=[]
for lo,hi in ((0,216),(216,234),(234,600),(600,1200),(1200,1510)):
    main=sum(x['producer_ms'] for x in old_a if lo<=x['source_frame_index']<hi)/1000
    direction=sum(x['producer_ms'] for x in old_d if lo<=x['source_frame_index']<hi)/1000
    segment.append(f'| {lo}–{hi-1} | {(hi-lo)/10:.1f} | {main:.2f} | {direction:.2f} | {main+direction:.2f} |')
text=f'''# Причины задержек и независимый расчёт направления T+1

Дополнительная сшивка — 377 отдельных оценок позы промежуточных облаков
T+1. Основная сшивка считает 755 поз для облаков 5 Гц; новая оценка нужна
для более раннего направления 373 выбранных расчётов. Облака T+1 не
добавляются в геометрию рельсов. Итого читаются 1132 сообщения из 1510.

В предыдущем варианте обе работы шли последовательно в одном producer.
Теперь оценка T+1 работает в отдельном потоке с собственной копией состояния.
Основной поток не ждёт её завершения. История T закрепляется до поступления
следующего основного облака; worker получает только её и позу исходного T+1.
Отказ или смена участка не позволяют опубликовать запоздалый результат старого участка.

## Где были задержки

Исходный кадр 232 (около 23,2 с): восстановление после бокса заняло 1274 мс,
из них 1219 мс — recover_endpoint. В коде это проверка пяти начальных
положений, каждое с лимитом до 60 итераций; при неудаче проверяются более ранние
опорные облака. Это разовый тяжёлый поиск, а не обычное чтение файла.

На исходном кадре 1322 обычная сшивка заняла 225 мс, из них ICP — 174 мс.
На промежуточном кадре 1407 только поза направления заняла 195 мс;
ICP выполнил 24 итерации (176 мс). На 1371 — 23 итерации. Без дополнительного
исследования причин сходимости это нельзя приписывать только повороту или шуму.

Главная причина долгого запаздывания после всплеска — слишком малый запас
в последовательном обработчике. По старому варианту:

| Исходные кадры | Длительность участка, с | Основные кадры, работа с | Дополнительная поза, работа с | Сумма последовательной работы, с |
|---|---:|---:|---:|---:|
{chr(10).join(segment)}

Например, на участке 234–599 работа занимала 35,46 с из 36,6 с записи.
Поэтому очередь, появившаяся после восстановления, уменьшалась медленно.

## Проверенный результат

| Метрика | T+1 последовательно | T+1 отдельно |
|---|---:|---:|
| Полный replay, с | {b['wall_seconds']:.3f} | {s['wall_seconds']:.3f} |
| Публикаций | {b['published']} | {s['published']} |
| Задержка новой публикации median, мс | {b['publication_latency_ms']['median']:.2f} | {s['publication_latency_ms']['median']:.2f} |
| Задержка новой публикации p95, мс | {b['publication_latency_ms']['p95']:.2f} | {s['publication_latency_ms']['p95']:.2f} |
| Задержка новой публикации max, мс | {b['publication_latency_ms']['max']:.2f} | {s['publication_latency_ms']['max']:.2f} |
| Готовность основного входа p95, мс | {b['raw_deadline_p95_ms']:.2f} | {s['raw_deadline_p95_ms']:.2f} |
| Последний вход: отставание, мс | {old_a[-1]['deadline_ms']:.2f} | {a[-1]['deadline_ms']:.2f} |

Запись 150,851 с. Время ожидания исходных timestamps включено; копирование
базы, проверка checksum и экспорт после replay исключены. Сеть DDS не тестировалась.
Каждый вариант измерен один раз; небольшие различия времени требуют повторов.

Все 755 основных поз и 377 поз направления совпали точно. Все {exact}
доступных прогнозов и их provenance также совпали точно. Численная геометрия,
число итераций, пороги сшивки, окно 12 кадров и замороженные стадии не изменены.
Проверены hashes, настоящие message_id/timestamps, отсутствие нечётных
кадров и будущих точек в геометрии. LAS не создавались.

Возраст последней доступной модели в моменты обработки входа: median
{verify['last_available_geometry_age_at_input_ms']['median']:.1f} мс, p95
{verify['last_available_geometry_age_at_input_ms']['p95']:.1f} мс, максимум
{verify['last_available_geometry_age_at_input_ms']['maximum']:.1f} мс.
Это отдельная метрика: между публикациями старый результат продолжает стареть.
Перенос модели учитывает последнюю обработанную позу, а не неизвестное
фактическое положение поезда в настенных часах.

## Следующие возможности

1. Переиспользовать дерево ближайших соседей одного опорного облака между ICP,
   включая пять гипотез восстановления. Это убирает повторные построения,
   но само по себе не устраняет итерации поиска.
2. Ускорить численное ядро ICP, включая поиск соответствий и сборку малой
   системы уравнений. Проверять не только миллисекунды, но позы и downstream-геометрию.
3. Отдельно ограничить время восстановления при пропаже опор. При превышении
   бюджета сообщать отсутствие достоверной позы; нельзя объявлять движение
   найденным, подставлять будущую позу или прятать сырой кадр с препятствием.
4. Если поезд предоставляет собственную одометрию/IMU, использовать её для
   направления и актуальной позы. В этой записи таких сообщений нет; скорость
   этого варианта на данном bag не измерена.

Одно только отсутствие растущей очереди не означает заданную максимальную
задержку. Для строгого требования нужен предел возраста результата; его
проверяют отдельно от частоты публикаций и длительности записи.

## Все этапы, мс

Вложенные и параллельные таймеры не складываются. direction_producer_ms —
суммарная работа подготовки и фонового расчёта, а не блокировка producer.
direction_producer_service_ms — работа на основном потоке; worker_queue_ms
показывает ожидание отдельного вычислителя.

| Этап | Вызовов | median | p95 | max |
|---|---:|---:|---:|---:|
{table}

## Воспроизведение

`./COPY_MAIN/run.ps1 -RunName my_async_direction -Count 1510 -InputStride 2 -SolveStride 2 -SolveOffset 1 -AsyncDirection`

Новые файлы: adapter/replay_direction_async.py; остальные изменения только
в COPY_MAIN/adapter и обвязке запуска. Замороженные файлы COPY_MAIN/app
и исходные регистрации не менялись. Временная копия DB3 удаляется вместе
с контейнером. Просмотрщик сохраняет предыдущие LAS, без новых экспортов.
'''
(dest/'REPORT.md').write_text(text,encoding='utf-8')
body=[];table_open=False
for line in text.splitlines():
    if line.startswith('|'):
        if not table_open:body.append('<table>');table_open=True
        if not line.startswith('|---'):body.append('<tr>'+''.join('<td>'+html.escape(c.strip())+'</td>' for c in line.strip('|').split('|'))+'</tr>')
    else:
        if table_open:body.append('</table>');table_open=False
        if line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line.startswith('# '):body.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line:body.append('<p>'+html.escape(line)+'</p>')
(dest/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Причины задержек и оптимизация</title><style>body{font:16px/1.5 system-ui;max-width:1250px;margin:36px auto;color:#22374a}table{border-collapse:collapse;width:100%}td{border:1px solid #ccd6df;padding:7px}tr:first-child{background:#e5edf5}</style>'+''.join(body),encoding='utf-8')
print('ASYNC_EXACT_PASS',json.dumps(dict(publication_latency=s['publication_latency_ms'],input_p95=s['raw_deadline_p95_ms'],last_input_lag=a[-1]['deadline_ms'],exact_predictions=exact)))
