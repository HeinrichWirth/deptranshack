"""Post-inference qualification of decimated raw input; no reference enters replay."""
import ast, collections, hashlib, html, json, sqlite3, sys, tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'.runtime'))
import numpy as np

def load(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def lines(p): return [json.loads(s) for s in p.read_text().splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def stat(a):
    return dict(n=len(a),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(max(a)),mean=float(np.mean(a))) if len(a) else dict(n=0)

def main():
    dest=HERE/'results/INPUT5HZ_SOLVE2_REPLAY'; out=dest/'payload'
    if not (out/'COMPLETE.json').exists():
        with tarfile.open(dest/'results.tar.gz') as f:f.extractall(dest,filter='data')
    s=load(out/'SUMMARY.json'); arrivals=lines(out/'arrivals.jsonl'); poses=lines(out/'poses.jsonl')
    jobs=load(out/'solves.json'); audits=load(out/'causality_audit.json')['solves']
    original=HERE/'results/FULL_COPY_MAIN/payload'; previous=HERE/'results/INPUT10HZ_SOLVE5_REPLAY/payload'
    origarr=lines(original/'arrivals.jsonl'); origposes=lines(original/'poses.jsonl')
    oldjobs={r['frame']:r for r in load(original/'solves.json')}
    source_map=load(out/'source_index_map.json')['frames']
    assert len(arrivals)==len(poses)==len(source_map)==755
    assert s['original_messages']==1510 and s['skipped_messages']==755
    assert [r['source_frame_index'] for r in arrivals]==list(range(0,1510,2))
    assert s['copy_main']['input_stride']==2 and s['copy_main']['solve_stride']==2 and s['copy_main']['solve_offset']==1
    for i,(a,p,m) in enumerate(zip(arrivals,poses,source_map)):
        assert a['frame']==p['frame']==m['frame']==i
        assert a['source_frame_index']==p['source_frame_index']==m['source_frame_index']==i*2
        for k in ('message_id','bag_time_ns'): assert a[k]==m[k]==origarr[i*2][k]
        assert a['source_points']==origarr[i*2]['source_points'] and a['eligible_points']==origarr[i*2]['eligible_points']
        assert not p['uses_future_pose'] and p['available_after_frame']==i and p['reference_index']<=i
    # Verify skipped payloads did not enter the replay's streaming checksum.
    db=ROOT/'datas/test_synthetic/cloud_with_fake_obj/cloud_with_fake_obj_0.db3'
    h=hashlib.sha256()
    with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as conn:
        for row in source_map:h.update(conn.execute('SELECT data FROM messages WHERE id=?',(row['message_id'],)).fetchone()[0])
    assert h.hexdigest()==s['source_message_payload_sha256']
    f=HERE/'app/MVP/stages/05_1_contact_marching_temporal_fusion/src/fusion.py'
    node=next(n for n in ast.parse(f.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='valid_edge')
    scope={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(f),'exec'),scope)
    records=[dict(header_time_ns=p['header_time_ns'],pose_status=p['status'],pose_uses_future=False,lidar_pose_in_folder=p['pose'] if p['pose'] is not None else np.full((4,4),np.nan)) for p in poses]
    def edge(i):return scope['valid_edge'](records[i],records[i+1])
    for a in audits:
        i=a['frame']; assert i%2==1 and a['source_frame_index']==i*2
        assert a['max_pose_frame']<=i+1 and a['max_original_pose_frame']<=2*i+2
        assert a['dispatched_after_frame']>=i+1
        expected=[i]
        for k in range(i-1,max(-1,i-12),-1):
            if not edge(k):break
            expected.append(k)
        assert a['cloud_frames']==sorted(expected)
        assert a['original_cloud_frames']==[k*2 for k in sorted(expected)]
    for r in jobs:
        i=r['frame'];assert i%2==1
        folder=out/'solves'/f'{i:06d}';done=load(folder/'PREDICTION_COMPLETE.json');summary=load(folder/'summary.json')
        assert summary['source_frame_index']==i*2 and summary['input_stride']==2
        assert not done['uses_annotation_labels'] and not done['uses_future_clouds']
        for name,digest in done['files'].items():assert sha(folder/name)==digest
        allowed={m['frame']:m for m in summary['source_identity_map']}
        with np.load(folder/'point_provenance.npz') as z:
            rows=z['rows']; assert not len(rows) or (rows['source_frame'].min()>=max(0,i-11) and rows['source_frame'].max()<=i)
            for k in np.unique(rows['source_frame']):
                assert allowed[int(k)]['source_frame_index']==int(k)*2
                assert allowed[int(k)]['message_id']==arrivals[int(k)]['message_id']
                assert rows['source_point_index'][rows['source_frame']==k].max()<arrivals[int(k)]['source_points']
    frozen=0
    for group in ('runtime_files','build_files','release_artifacts'):
        for name,item in load(HERE/'ORIGINAL_RELEASE_MANIFEST.json')[group].items():
            if (ROOT/name).exists():assert sha(ROOT/name)==item['sha256'];frozen+=1
    for name,digest in load(HERE/'MANIFEST.json').items():
        if name.startswith(('app/','registration/')):assert sha(HERE/name)==digest
    assert not list(out.rglob('*.las'))
    selected=list(range(1,755,2)); solved={r['frame'] for r in jobs}
    missing=[dict(frame=i,source_frame_index=i*2,reason='NO_T_PLUS_1' if i+1>=755 else 'INVALID_POSE_EDGE' if not edge(i) else 'MOVEMENT_GATE_OR_SCHEDULER') for i in selected if i not in solved]
    position=[];angle=[];newinvalid=[];recovered=[]
    for i,p in enumerate(poses):
        old=origposes[i*2]
        if p['pose'] is None and old['pose'] is not None:newinvalid.append(i*2)
        if p['pose'] is not None and old['pose'] is None:recovered.append(i*2)
        if p['pose'] is None or old['pose'] is None:continue
        A,B=np.asarray(p['pose']),np.asarray(old['pose'])
        position.append(float(np.linalg.norm(A[:3,3]-B[:3,3])))
        angle.append(float(np.degrees(np.arccos(np.clip((np.trace(A[:3,:3].T@B[:3,:3])-1)/2,-1,1)))))
    local_motion={}
    for span in (1,11):
        translations=[];rotations=[]
        for i in range(span,len(poses)):
            j=i-span
            if not all(edge(k) for k in range(j,i)):continue
            if any(origposes[k]['pose'] is None for k in range(j*2,i*2+1)):continue
            A,Z=np.asarray(poses[i]['pose']),np.asarray(poses[j]['pose'])
            B,Y=np.asarray(origposes[i*2]['pose']),np.asarray(origposes[j*2]['pose'])
            ta=A[:3,:3].T@(Z[:3,3]-A[:3,3]);tb=B[:3,:3].T@(Y[:3,3]-B[:3,3])
            ra=A[:3,:3].T@Z[:3,:3];rb=B[:3,:3].T@Y[:3,:3]
            translations.append(float(np.linalg.norm(ta-tb)*100))
            rotations.append(float(np.degrees(np.arccos(np.clip((np.trace(ra.T@rb)-1)/2,-1,1)))))
        local_motion[str(span)]=dict(interval_seconds=span*.2,translation_cm=stat(translations),rotation_deg=stat(rotations))
    bands=[(0,8),(8,30),(30,50),(50,75),(75,100)];differences=[];availability=[];horizon=[]
    for r in jobs:
        i=r['frame'];src=i*2
        if src not in oldjobs:continue
        old=oldjobs[src]
        availability.append(dict(source_frame=src,old=old['final_geometry_available'],new=r['final_geometry_available']))
        if not (old['final_geometry_available'] and r['final_geometry_available']):continue
        with np.load(out/'solves'/f'{i:06d}'/'prediction.npz') as a,np.load(original/'solves'/f'{src:06d}'/'prediction.npz') as b:
            hi=min(float(a['s'].max()),float(b['s'].max()));lo=max(float(a['s'].min()),float(b['s'].min()))
            horizon.append(dict(source_frame=src,old=float(b['s'].max()),new=float(a['s'].max())))
            for low,high in bands:
                grid=np.arange(max(low,lo),min(high,hi)+1e-8,1.)
                if not len(grid):continue
                # Same raw sensor frame; compare at equal model arc length, no map alignment.
                values={}
                for key in ('C','pair'):
                    aa=np.asarray(a[key]);bb=np.asarray(b[key]);shape=aa.shape[1:]
                    av=np.column_stack([np.interp(grid,a['s'],v) for v in aa.reshape(len(aa),-1).T]).reshape((len(grid),)+shape)
                    bv=np.column_stack([np.interp(grid,b['s'],v) for v in bb.reshape(len(bb),-1).T]).reshape((len(grid),)+shape)
                    if key=='pair':
                        direct=np.linalg.norm(av-bv,axis=-1);swapped=np.linalg.norm(av-bv[:,::-1],axis=-1)
                        dist=swapped if np.mean(swapped)<np.mean(direct) else direct
                    else:dist=np.linalg.norm(av-bv,axis=-1)
                    values[key]=dict(median_cm=float(np.median(dist)*100),max_cm=float(np.max(dist)*100))
                differences.append(dict(source_frame=src,band=f'{low}-{high}',samples=len(grid),**values))
    pubs=load(out/'publications.json'); intervals=[(b['clock']-a['clock'])*1000 for a,b in zip(pubs,pubs[1:])]
    comparison=dict(pass_=True,frames=755,skipped=755,causal_jobs=len(audits),source_payload_hash_verified=True,source_identity_verified=True,history_leases_verified=True,frozen_files_checked=frozen,las_count=0,missing=missing,
        source_invalid_frames=[p['source_frame_index'] for p in poses if p['pose'] is None],new_invalid_vs_10hz=newinvalid,new_valid_vs_10hz=recovered,
        pose_position_difference_m=stat(position),pose_rotation_difference_deg=stat(angle),common_availability=availability,
        geometry_difference=differences,horizon_pairs=horizon,producer_ms=stat([r['producer_ms'] for r in arrivals]),
        final_lag_ms=arrivals[-1]['deadline_ms'],publication_intervals_ms=stat(intervals),arrival_lag_ms=stat([r['deadline_ms'] for r in arrivals]),local_motion_difference=local_motion)
    (dest/'VERIFICATION.json').write_text(json.dumps(comparison,indent=2)+'\n')
    prevs=load(previous/'SUMMARY.json'); bas=load(original/'SUMMARY.json')
    oldst=load(previous/'STAGES_MS.json')['timings'];newst=load(out/'STAGES_MS.json')['timings']
    mapping=[('Чтение','read_ms'),('Декодирование','decode_ms'),('Сшивка','registration_ms'),('Воксели','registration_voxel_sample_ms'),('ICP','registration_icp_ms'),('Нормали','registration_normals_ms'),('FrameData','frame_data_ms'),('Входной кадр целиком','producer_ms'),('Индекс / планировщик','core_arrival_ms'),('STEP1','T_STEP1_ms'),('STEP2','T_STEP2_ms'),('C4 с историей','T_C4_MARCHING_INCLUSIVE_HISTORY_IO_ms'),('Контракт C4','T_GEOMETRY_CONTRACT_ms'),('Seed adapter','T_SEED_ADAPTER_ms'),('Сглаживание C4','T_C4_SMOOTH_ms'),('STEP6','T_STEP6_RAIL_RECONSTRUCTION_ms'),('Provenance','T_PROVENANCE_ms'),('Сериализация','T_SERIALIZATION_ms'),('Геометрия целиком','solve_wall_ms')]
    table='\n'.join('| '+label+' | '+' | '.join(f'{v:.2f}' for v in (oldst[k]['median_ms'],oldst[k]['p95_ms'],newst[k]['median_ms'],newst[k]['p95_ms']))+' |' for label,k in mapping if k in newst and k in oldst)
    dt=[]
    for low,high in bands:
        rows=[r for r in differences if r['band']==f'{low}-{high}']
        if rows:dt.append(f"| {low}–{high} | {len(rows)} | "+' | '.join(f'{v:.2f}' for key in ('C','pair') for v in (np.median([r[key]['median_cm'] for r in rows]),np.quantile([r[key]['median_cm'] for r in rows],.95),max(r[key]['max_cm'] for r in rows)))+' |')
    geometry_table='\n'.join(dt)
    lost=[r['source_frame'] for r in availability if r['old'] and not r['new']]
    gained=[r['source_frame'] for r in availability if not r['old'] and r['new']]
    horizon_old=stat([r['old'] for r in horizon]);horizon_new=stat([r['new'] for r in horizon])
    shorter=sum(r['new']<r['old']-10 for r in horizon);longer=sum(r['new']>r['old']+10 for r in horizon)
    worst=sorted(differences,key=lambda r:r['pair']['max_cm'],reverse=True)[:8]
    worst_table='\n'.join(f"| {r['source_frame']} | {r['band']} | {r['C']['max_cm']:.2f} | {r['pair']['max_cm']:.2f} |" for r in worst)
    motion_table='\n'.join(f"| {v['interval_seconds']:.1f} | {v['translation_cm']['median']:.2f} | {v['translation_cm']['p95']:.2f} | {v['translation_cm']['max']:.2f} | {v['rotation_deg']['p95']:.3f} |" for v in local_motion.values())
    text=f'''# COPY_MAIN — вход 5 Гц, геометрия на каждом втором кадре

Из 1510 исходных кадров прочитаны и сшиты 755: исходные индексы 0, 2, 4…1508. Геометрия для индексов потока 1, 3, 5… (исходные 2, 6, 10…). Результат: **{s['wall_seconds']*1000:.0f} мс** на запись {s['original_recording_seconds']*1000:.0f} мс; последний оставленный кадр имеет время {s['recording_seconds']*1000:.0f} мс. Последний входной кадр завершён через {comparison['final_lag_ms']:.0f} мс после своего срока.

## Полный прогон

| Показатель | COPY_MAIN исходный | 10 Гц / каждый 5-й | 5 Гц / каждый 2-й |
|---|---:|---:|---:|
| Время, мс | {bas['wall_seconds']*1000:.0f} | {prevs['wall_seconds']*1000:.0f} | {s['wall_seconds']*1000:.0f} |
| Прочитано кадров | {bas['frames']} | {prevs['frames']} | {s['frames']} |
| Расчётов геометрии | {bas['completed_solves']} | {prevs['completed_solves']} | {s['completed_solves']} |
| Опубликовано решений | {bas['published']} | {prevs['published']} | {s['published']} |
| Публикации, Гц по времени прогона | {bas['publish_hz_wall']:.3f} | {prevs['publish_hz_wall']:.3f} | {s['publish_hz_wall']:.3f} |
| Задержка публикации p95, мс | {bas['publication_latency_ms']['p95']:.0f} | {prevs['publication_latency_ms']['p95']:.0f} | {s['publication_latency_ms']['p95']:.0f} |
| Задержка публикации max, мс | {bas['publication_latency_ms']['max']:.0f} | {prevs['publication_latency_ms']['max']:.0f} | {s['publication_latency_ms']['max']:.0f} |

Ритм входа задаётся исходными временными метками, поэтому длительность успешного прогона ограничена длительностью записи: ожидание следующих кадров входит в замер. Средняя активная обработка входного кадра — {comparison['producer_ms']['mean']:.2f} мс при интервале около 200 мс. Задержка приёма p95/max — {comparison['arrival_lag_ms']['p95']:.0f}/{comparison['arrival_lag_ms']['max']:.0f} мс. Интервал публикаций median/p95/max — {comparison['publication_intervals_ms']['median']:.0f}/{comparison['publication_intervals_ms']['p95']:.0f}/{comparison['publication_intervals_ms']['max']:.0f} мс. Это разные показатели: отсутствие растущей очереди не означает, что каждый прогноз готов за 200 мс.

## Все этапы, мс

| Этап | 10 Гц / 5-й median | 10 Гц / 5-й p95 | 5 Гц / 2-й median | 5 Гц / 2-й p95 |
|---|---:|---:|---:|---:|
{table}

Сшивка теперь выполняется вдвое реже; один вызов не обязан стать быстрее. Дочерние и параллельные таймеры не складываются. Нормали учитываются только в кадрах, где рассчитывались. Набор входных облаков изменился: это изменение режима данных, а не доказательство численно эквивалентной оптимизации. Оба режима замерены по одному разу.

## Причинность и идентичность источников

История — до 12 оставленных кадров (до 2,2 с вместо 1,1 с); пропущенные облака не читаются и не используются. Для T разрешена поза следующего оставленного кадра: исходный T+2, около 200 мс. Его облако не попадает в геометрию T. После разрыва окно короче; интерполяции через разрывы нет. Никакая сохранённая поза или геометрия не входит в inference.

Проверены все {len(audits)} запуска, точный состав истории, checksum только оставленных сообщений, исходные timestamps/message_id и provenance. Внутренний frame/source_frame — индекс потока 5 Гц; исходный индекс сохранён в source_frame_index, source_index_map.json и source_identity_map каждой summary. Номер точки внутри исходного облака сохранён. {frozen} frozen-файлов и геометрический код COPY_MAIN неизменны. Во время измеряемого replay LAS не создаются; экспорт для просмотра выполняется отдельно после него.

## Изменения результатов — сравнение, не независимый GT

Недоступные позы, исходные индексы: {comparison['source_invalid_frames']}. Дополнительные отказы относительно 10 Гц: {newinvalid}; восстановленные: {recovered}. Пропущенные выбранные расчёты: {missing}.

Сравнение накопленных поз на общих исходных кадрах в одной начальной системе координат: смещение median/p95/max = {comparison['pose_position_difference_m']['median']:.3f}/{comparison['pose_position_difference_m']['p95']:.3f}/{comparison['pose_position_difference_m']['max']:.3f} м; поворот median/p95/max = {comparison['pose_rotation_difference_deg']['median']:.3f}/{comparison['pose_rotation_difference_deg']['p95']:.3f}/{comparison['pose_rotation_difference_deg']['max']:.3f}°. Это расхождение двух оценок регистрации, не ошибка относительно истинного движения.

Чтобы отделить накопленное расхождение общей траектории от локального совмещения истории, дополнительно сравнено движение на одинаковых физических интервалах, выраженное в системе текущего датчика. Интервалы с недоступной позой исключены. Малые локальные различия могут накапливаться на длинном пути; следующий замер не отменяет необходимости проверки регистрации.

| Интервал, с | Локальное смещение median, см | p95, см | max, см | Локальный поворот p95, ° |
|---|---:|---:|---:|---:|
{motion_table}

Общих рассчитанных исходных кадров с полным COPY_MAIN: {len(availability)}. Было решение, стало недоступно: {lost}; появилось решение: {gained}. На кадрах, где доступны оба прогноза, сравниваются координаты в системе текущего датчика при одинаковой длине вдоль модельной кривой, только на общей дальности. Для пары выбирается соответствие двух рельсов с меньшим суммарным расстоянием. Каждая строка сначала агрегируется внутри кадра, затем между кадрами; max — наибольшее отдельное расхождение. Различие может включать сдвиг начальной точки параметризации. Старый прогноз не считается истинным.

На {len(horizon)} общих доступных прогнозах дальность модели: median {horizon_old['median']:.1f} → {horizon_new['median']:.1f} м, mean {horizon_old['mean']:.1f} → {horizon_new['mean']:.1f} м. Сокращение больше 10 м: {shorter} кадров; увеличение больше 10 м: {longer}. Ускорение геометрии нельзя целиком приписать вычислительной оптимизации: объём найденного пути тоже меняется.

| Дальность, м | Общих кадров | CR median, см | CR p95, см | CR max, см | Рельсы median, см | Рельсы p95, см | Рельсы max, см |
|---|---:|---:|---:|---:|---:|---:|---:|
{geometry_table}

Крупнейшие расхождения для дополнительной проверки:

| Исходный кадр | Дальность | CR max, см | Рельсы max, см |
|---|---|---:|---:|
{worst_table}

Полные пары дальностей и изменения по кадрам сохранены в VERIFICATION.json. Частота и история изменились; прежнюю точность автоматически переносить на новый режим нельзя.

## Воспроизведение

`./COPY_MAIN/run.ps1 -RunName my_5hz -Count 1510 -InputStride 2 -SolveStride 2 -SolveOffset 1`

Два geometry worker × два candidate thread; ICP query worker = 1. Исходный DB3 неизменен. Копирование DB3 внутрь Linux и его SHA256-проверка выполняются до замера и прогревают файловый кэш. Прогнозы не предвычисляются. JSON/NPZ пишутся локально внутри Linux, архив переносится после измерения. Временная копия DB3 удаляется вместе с контейнером (--rm); подтверждение в CONTAINER_CLEANUP.json. Это симуляция ROS2 PointCloud2 из DB3 без DDS-транспорта.
'''
    (dest/'REPORT.md').write_text(text,encoding='utf-8')
    body=[];inside=False
    for line in text.splitlines():
        if line.startswith('|'):
            if not inside:body.append('<table>');inside=True
            if not line.startswith('|---'):body.append('<tr>'+''.join('<td>'+html.escape(c.strip())+'</td>' for c in line.strip('|').split('|'))+'</tr>')
        else:
            if inside:body.append('</table>');inside=False
            if line.startswith('# '):body.append('<h1>'+html.escape(line[2:])+'</h1>')
            elif line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
            elif line:body.append('<p>'+html.escape(line).replace('**','')+'</p>')
    (dest/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>COPY_MAIN · Вход 5 Гц</title><style>body{font:16px/1.55 system-ui;max-width:1250px;margin:36px auto;padding:0 24px;color:#23374d;background:#f8fafc}table{border-collapse:collapse;width:100%;background:white}td{border:1px solid #d7e0e8;padding:8px}tr:first-child{font-weight:bold;background:#e5edf5}</style>'+''.join(body),encoding='utf-8')
    print(json.dumps({k:v for k,v in comparison.items() if k not in ('common_availability','geometry_difference','horizon_pairs')},indent=2))
    print('availability_lost',lost,'gained',gained)
    print('worst',json.dumps(worst));print('REPORT_DONE')

if __name__=='__main__':main()

