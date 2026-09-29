"""Post-inference verification of immediate original T+1 direction-only mode."""
import sys,json,hashlib,sqlite3,tarfile,csv,html,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'.runtime'))
import numpy as np
HERE=ROOT/'COPY_MAIN'
dest=HERE/'results'/(sys.argv[1] if len(sys.argv)>1 else 'INPUT5HZ_SOLVE2_POSE1');out=dest/'payload'
base=HERE/'results/INPUT5HZ_SOLVE2_REPLAY/payload'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def stats(values):
    return dict(n=len(values),median=float(np.median(values)),p95=float(np.quantile(values,.95)),max=float(max(values))) if values else dict(n=0)
if not (out/'COMPLETE.json').exists():
    with tarfile.open(dest/'results.tar.gz') as archive:archive.extractall(dest,filter='data')
s=load(out/'SUMMARY.json');old=load(base/'SUMMARY.json')
poses=lines(out/'poses.jsonl');previous=lines(base/'poses.jsonl')
arrivals=lines(out/'arrivals.jsonl');oldarr=lines(base/'arrivals.jsonl')
jobs=load(out/'solves.json');directions=load(out/'direction_poses.json');audits=load(out/'causality_audit.json')['solves']
assert len(poses)==len(arrivals)==755 and len(directions)==377
assert s['total_payloads_read']==1132 and s['not_read_messages']==378
assert s['source_message_payload_sha256']==old['source_message_payload_sha256']
for p,b,a in zip(poses,previous,arrivals):
    assert p==b,('retained pose changed',p['frame'])
    assert a['source_frame_index']==2*a['frame']
directionmap={d['frame']:d for d in directions}
db=ROOT/'datas/test_synthetic/cloud_with_fake_obj/cloud_with_fake_obj_0.db3'
with sqlite3.connect(db.as_uri()+'?mode=ro',uri=True) as con:
    schedule=con.execute('SELECT id,timestamp FROM messages ORDER BY timestamp,id').fetchall()
    for d in directions:
        original=d['source_frame_index'];i=d['frame']
        assert i%2==1 and original==i*2+1
        assert (d['message_id'],d['bag_time_ns'])==schedule[original]
        payload=con.execute('SELECT data FROM messages WHERE id=?',(d['message_id'],)).fetchone()[0]
        assert hashlib.sha256(payload).hexdigest()==d['payload_sha256']
        assert 0<d['header_time_ns']-arrivals[i]['header_time_ns']<150_000_000
        assert d['ready_clock']>=d['raw_deadline']
for a in audits:
    i=a['frame'];assert i%2==1 and a['source_frame_index']==i*2
    assert a['direction_pose_only'] and a['max_original_pose_frame']==i*2+1
    assert all(k%2==0 and k<=i*2 for k in a['original_cloud_frames'])
    assert a['original_cloud_frames']==[k*2 for k in a['cloud_frames']]
    assert min(a['cloud_frames'])>=max(0,i-11)
checks=0
for job in jobs:
    i=job['frame'];folder=out/'solves'/f'{i:06d}';summary=load(folder/'summary.json');done=load(folder/'PREDICTION_COMPLETE.json')
    assert summary['direction_pose_source_frame_index']==i*2+1
    assert not done['uses_annotation_labels'] and not done['uses_future_clouds']
    if summary['pose_wait_seconds'] is not None:assert .09<summary['pose_wait_seconds']<.11
    for name,digest in done['files'].items():assert sha(folder/name)==digest
    with np.load(folder/'point_provenance.npz') as z:
        rows=z['rows']
        assert not len(rows) or (rows['source_frame'].min()>=max(0,i-11) and rows['source_frame'].max()<=i)
        allowed={r['frame']:r for r in summary['source_identity_map']}
        for k in np.unique(rows['source_frame']):
            assert allowed[int(k)]['source_frame_index']==int(k)*2
            assert rows['source_point_index'][rows['source_frame']==k].max()<arrivals[int(k)]['source_points']
    checks+=1
candidate=load(out/'CANDIDATE_SOURCE.json')
previous_candidate=load(base/'CANDIDATE_SOURCE.json')
for name,digest in candidate.items():
    if name.startswith(('/candidate/','/work/registration/')):assert digest==previous_candidate[name]
    path=HERE/'app'/name[len('/candidate/'):] if name.startswith('/candidate/') else HERE/name[len('/work/'):]
    snapshot=dest/'adapter_snapshot'/path.name
    if name.startswith('/work/adapter/') and snapshot.exists():path=snapshot
    assert sha(path)==digest
assert load(out/'freeze_before.json')['pass_'] and s['frozen_runtime_unchanged']
assert not list(out.rglob('*.las'))
oldjobs={r['frame']:r for r in load(base/'solves.json')};diff=[];ranges=[];lost=[];gained=[]
for job in jobs:
    i=job['frame'];b=oldjobs.get(i)
    if b is None:continue
    if b['final_geometry_available'] and not job['final_geometry_available']:lost.append(i*2)
    if job['final_geometry_available'] and not b['final_geometry_available']:gained.append(i*2)
    if not (b['final_geometry_available'] and job['final_geometry_available']):continue
    with np.load(out/'solves'/f'{i:06d}'/'prediction.npz') as a,np.load(base/'solves'/f'{i:06d}'/'prediction.npz') as b:
        ranges.append(dict(source_frame=i*2,before=float(b['s'][-1]),after=float(a['s'][-1])))
        for lo,hi in ((0,8),(8,30),(30,50),(50,75),(75,100)):
            grid=np.arange(max(lo,a['s'][0],b['s'][0]),min(hi,a['s'][-1],b['s'][-1])+1e-8,1)
            if not len(grid):continue
            row=dict(source_frame=i*2,band=f'{lo}-{hi}')
            for key in ('C','pair'):
                av=a[key];bv=b[key];shape=av.shape[1:]
                A=np.column_stack([np.interp(grid,a['s'],v) for v in av.reshape(len(av),-1).T]).reshape((len(grid),)+shape)
                B=np.column_stack([np.interp(grid,b['s'],v) for v in bv.reshape(len(bv),-1).T]).reshape((len(grid),)+shape)
                d=np.linalg.norm(A-B,axis=-1)
                if key=='pair':
                    swapped=np.linalg.norm(A-B[:,::-1],axis=-1)
                    if swapped.mean()<d.mean():d=swapped
                row[key]=dict(median_cm=float(np.median(d)*100),max_cm=float(d.max()*100))
            diff.append(row)
pubs=load(out/'publications.json')
details=[]
for p in pubs:
    i=p['source_frame'];d=directionmap[i]
    details.append(dict(source_frame=i*2,wait_next_measurement_ms=(d['raw_deadline']-arrivals[i]['raw_deadline'])*1000,
                        direction_ready_lag_ms=(d['ready_clock']-d['raw_deadline'])*1000,
                        after_direction_to_publish_ms=(p['clock']-d['ready_clock'])*1000,
                        total_ms=p['latency_from_raw_ms']))
comparison=dict(pass_=True,retained_poses_exact=755,direction_payloads_verified=377,causal_jobs=checks,
                source_clouds_even_only=True,future_clouds_in_geometry=False,source_unchanged=s['source_unchanged'],
                lost=lost,gained=gained,horizons=ranges,geometry_differences=diff,
                latency_components={k:stats([r[k] for r in details]) for k in details[0] if k!='source_frame'},
                before=old['publication_latency_ms'],after=s['publication_latency_ms'],
                publication_intervals_ms=stats([(b['clock']-a['clock'])*1000 for a,b in zip(pubs,pubs[1:])]),
                final_arrival_lag_ms=arrivals[-1]['deadline_ms'])
(dest/'VERIFICATION.json').write_text(json.dumps(comparison,indent=2)+'\n')
(dest/'LATENCY_COMPONENTS.json').write_text(json.dumps(details,indent=2)+'\n')
exact_note=''
if dest.name.endswith('_FAST'):
    trial=HERE/'results/INPUT5HZ_SOLVE2_POSE1/payload'
    prior=load(trial/'direction_poses.json')
    assert len(prior)==len(directions)
    for a,b in zip(directions,prior):assert a['pose']==b['pose']
    old_ids={r['frame']:r for r in load(trial/'solves.json')}
    assert {r['frame'] for r in jobs}==set(old_ids)
    exact=0
    for job in jobs:
        i=job['frame'];assert job['final_geometry_available']==old_ids[i]['final_geometry_available']
        if not job['final_geometry_available']:continue
        with np.load(out/'solves'/f'{i:06d}'/'prediction.npz') as a,np.load(trial/'solves'/f'{i:06d}'/'prediction.npz') as b:
            assert set(a.files)==set(b.files)
            for key in a.files:np.testing.assert_array_equal(a[key],b[key])
        exact+=1
    comparison['reference_refresh_removed_exact']=dict(direction_poses=len(prior),predictions=exact)
    (dest/'VERIFICATION.json').write_text(json.dumps(comparison,indent=2)+'\n')
    exact_note=f'Удаление ненужного обновления опорных нормалей проверено на всех {len(prior)} дополнительных позах и {exact} доступных прогнозах: результаты совпали точно с первым вариантом T+1. Вариант до оптимизации сохранён в INPUT5HZ_SOLVE2_POSE1 вместе со снимком адаптера.'
timings=load(out/'STAGES_MS.json')['timings'];beforetimings=load(base/'STAGES_MS.json')['timings']
with (dest/'STAGES_MS.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.writer(f);w.writerow(['stage','calls','median_ms','p95_ms','max_ms'])
    for k,v in timings.items():w.writerow([k,v['n'],v['median_ms'],v['p95_ms'],v['max_ms']])
table='\n'.join(f"| {k} | {v['n']} | {v['median_ms']:.2f} | {v['p95_ms']:.2f} | {v['max_ms']:.2f} |" for k,v in timings.items())
bands=[]
for band in ('0-8','8-30','30-50','50-75','75-100'):
    rows=[r for r in diff if r['band']==band]
    if rows:bands.append('| '+band+' | '+str(len(rows))+' | '+' | '.join(f'{v:.2f}' for key in ('C','pair') for v in (np.median([r[key]['median_cm'] for r in rows]),np.quantile([r[key]['median_cm'] for r in rows],.95),max(r[key]['max_cm'] for r in rows)))+' |')
components=comparison['latency_components']
text=f'''# Направление из исходного T+1, облака 5 Гц / расчёт 2,5 Гц

Исправлена связь между отбором облаков и источником направления. Для расчётного
исходного T берётся поза исходного T+1, а не следующего принятого T+2.
Облака для рельсов остаются 0,2,4…1508; выбранные расчёты — 2,6,10…1506.
Прямое продолжение +20 м и замороженные численные стадии не менялись.

{exact_note}

## Полный последовательный прогон

| Показатель | Прежде, поза T+2 | Теперь, поза T+1 |
|---|---:|---:|
| Время replay, с | {old['wall_seconds']:.3f} | {s['wall_seconds']:.3f} |
| Расчётов | {old['completed_solves']} | {s['completed_solves']} |
| Публикаций | {old['published']} | {s['published']} |
| Задержка публикации median, мс | {old['publication_latency_ms']['median']:.2f} | {s['publication_latency_ms']['median']:.2f} |
| Задержка публикации p95, мс | {old['publication_latency_ms']['p95']:.2f} | {s['publication_latency_ms']['p95']:.2f} |
| Задержка публикации max, мс | {old['publication_latency_ms']['max']:.2f} | {s['publication_latency_ms']['max']:.2f} |

Длительность исходной записи {s['original_recording_seconds']:.3f} с.
Задержка завершения последнего принятого входного кадра относительно его
срока {arrivals[-1]['deadline_ms']:.2f} мс. Копирование DB3, проверка SHA256,
экспорт результатов и последующие проверки исключены из времени replay.
Прогнозы заранее не считались, чтение идёт по исходным временам без DDS-транспорта.

## Что означает задержка

Отсчёт идёт от исходного времени кадра T в расписании до готовности нового
прогноза в планировщике, до записи диагностического NPZ. Это не сеть.
Разложение по тем же {len(details)} опубликованным результатам:

| Часть задержки | median, мс | p95, мс |
|---|---:|---:|
| До измерения T+1 | {components['wait_next_measurement_ms']['median']:.2f} | {components['wait_next_measurement_ms']['p95']:.2f} |
| От срока T+1 до готовности его позы, включая возможную очередь | {components['direction_ready_lag_ms']['median']:.2f} | {components['direction_ready_lag_ms']['p95']:.2f} |
| От готовности позы до публикации | {components['after_direction_to_publish_ms']['median']:.2f} | {components['after_direction_to_publish_ms']['p95']:.2f} |

Медианы частей не обязаны складываться в медиану общей задержки.
По каждому результату точная сумма находится в LATENCY_COMPONENTS.json.
Между обновлениями возраст последнего результата дополнительно растёт.

## Как получена поза без изменения сшивки 5 Гц

В этой DB3 нет готовой одометрии: только PointCloud2. Промежуточное облако
T+1 читается для ICP относительно уже имеющегося состояния T. Это отдельная
пробная оценка: после неё состояние основного ICP восстанавливается.
755 сохранённых поз основного потока совпали с прежним режимом точно,
включая отказы. Проверено восстановление состояния также при исключении.
Точки T+1 не создают FrameData, не добавляются в историю/пространственный
индекс рельсов и не поступают в STEP1/STEP2/C4/STEP6. В worker передаётся
только sanitized pose T+1 с настоящим timestamp.

Прочитано 755 основных облаков и 377 дополнительных облаков только для позы,
всего 1132 из 1510; 378 сообщений не читались. Расчёт направления выполняется
только для выбранных T. Цена копирования состояния включена в direction_registration_ms.
Поза не интерполируется, движение из будущего T+2 не используется.
Если T+1 не удалось зарегистрировать, расчёт T не подменяется позой T+2.

## Проверки и изменения результата

Проверены все {checks} расчётов: original pose <= T+1; облака только чётных
исходных кадров не позже T; окно до 12 принятых кадров; точная provenance;
hashes файлов результатов, 377 payload T+1, неизменность runtime и исходника.
Новые недоступные решения на общих рассчитанных кадрах: {lost}.
Новые доступные: {gained}. Иной интервал измерения направления может менять
геометрию; это не обещание точного совпадения с предыдущим направлением T+2.

На общей дальности сравниваются сохранённые модели в координатах одного
исходного лидара. Это расхождение двух прогнозов, не независимый GT.
Медиана/p95 — по медианным ошибкам внутри каждого кадра; max — худшая точка.

| Дальность, м | Кадров | CR median, см | CR p95, см | CR max, см | Пара median, см | Пара p95, см | Пара max, см |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(bands)}

На исходном кадре 118 заметное расхождение: направление по T+1 отличается от
T+2 на 0,399 градуса, база движения 0,393 м вместо 0,786 м. Максимальное
расхождение ближних моделей — 66,07 см для контактного рельса и 34,63 см
для пары. Это не измеренная ошибка по GT; меньшая база и смена выбора
геометрии требуют отдельной визуальной проверки. Порогов под этот кадр не меняли.

## Все этапы в миллисекундах

Вложенные и параллельные таймеры не складываются. Direction-строки —
дополнительная работа позы T+1, остальные — прежний поток 5 Гц и геометрия.

| Этап | Вызовов | median | p95 | max |
|---|---:|---:|---:|---:|
{table}

## Воспроизведение

`./COPY_MAIN/run.ps1 -RunName my_pose1 -Count 1510 -InputStride 2 -SolveStride 2 -SolveOffset 1 -ImmediateDirection`

Реализация отдельная: adapter/replay_direction.py, direction_scheduler.py,
direction_probe.py. Прежний replay.py сохранён для сравнения. Геометрические
файлы COPY_MAIN/app и frozen stages не менялись. В этом замере новых LAS нет;
просмотрщик пока показывает предыдущий INPUT5HZ_SOLVE2_REPLAY. Временная
копия DB3 удаляется с одноразовым контейнером; CONTAINER_CLEANUP.json фиксирует проверку.
'''
(dest/'REPORT.md').write_text(text,encoding='utf-8')
body=[];inside=False
for line in text.splitlines():
    if line.startswith('|'):
        if not inside:body.append('<table>');inside=True
        if not line.startswith('|---'):body.append('<tr>'+''.join('<td>'+html.escape(c.strip())+'</td>' for c in line.strip('|').split('|'))+'</tr>')
    else:
        if inside:body.append('</table>');inside=False
        if line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line.startswith('# '):body.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line:body.append('<p>'+html.escape(line)+'</p>')
(dest/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Направление T+1</title><style>body{font:16px/1.5 system-ui;max-width:1250px;margin:32px auto;color:#20344a}table{border-collapse:collapse;width:100%}td{border:1px solid #ccd6df;padding:7px}tr:first-child{background:#e5edf5}</style>'+''.join(body),encoding='utf-8')
print(json.dumps({k:v for k,v in comparison.items() if k not in ('horizons','geometry_differences')},indent=2))
print('REPORT_DIRECTION_PASS')
