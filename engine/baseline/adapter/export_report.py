"""Post-inference report/LAS exports only. Never a source for the predictor."""
import sys,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
import bootstrap
import argparse,json,sqlite3,time,hashlib,collections,html,shutil
import numpy as np
import laspy
sys.path.insert(0,str(ROOT/'results_rail2d/_packages'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cdr_cloud import decode

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def sample_polyline(a,spacing=.1):
    if len(a)<2:return a
    out=[]
    for start,end in zip(a[:-1],a[1:]):
        n=max(1,int(np.ceil(np.linalg.norm(end-start)/spacing)));out.append(start+(end-start)*np.arange(n)[:,None]/n)
    return np.vstack([*out,a[-1:]])
def header(overlay=False):
    h=laspy.LasHeader(point_format=2,version='1.4');h.scales=np.full(3,.0001);h.offsets=np.zeros(3)
    specs=[('frame_index','u4'),('point_index','i8' if overlay else 'u4'),('message_id','u4'),('intensity_raw','f4')]
    if overlay:specs += [('geometry_generated','u1'),('geometry_component','u1')]
    for name,dtype in specs:h.add_extra_dim(laspy.ExtraBytesParams(name=name,type=dtype))
    h.vlrs.append(laspy.VLR(user_id='DEPTRANS',record_id=1,description='CAUSAL ROS2 REPLAY',record_data=json.dumps(dict(original_db='cloud_with_fake_obj_0.db3',coordinates='sensor T' if overlay else 'first lidar frame / causal ICP map',poses='poses.jsonl; no future interpolation',classification='unassigned; prediction samples are separate geometry',missing=['ring','point timestamp'],timestamps='frame_index -> poses.jsonl / arrivals.jsonl',rgb='raw grayscale; generated left red/right green/contact cyan',generated_component={'1':'left rail','2':'right rail','3':'contact reference'})).encode()))
    return h
def raw_las(h,xyz,p,ids,frame,message):
    a=laspy.ScaleAwarePointRecord.zeros(len(xyz),header=h);a.x=xyz[:,0];a.y=xyz[:,1];a.z=xyz[:,2]
    a.frame_index[:]=frame;a.point_index=ids;a.message_id[:]=message;a.intensity_raw=p['intensity'][ids]
    a.intensity=np.rint(np.clip(a.intensity_raw,0,65535)).astype(np.uint16)
    gray=(np.clip(a.intensity_raw,0,255)*257).astype(np.uint16);a.red=gray;a.green=gray;a.blue=gray
    return a

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--result',type=Path,required=True);ap.add_argument('--db',type=Path,required=True);a=ap.parse_args();out=a.result
    assert (out/'COMPLETE.json').exists()
    summary=load(out/'SUMMARY.json');arrivals=[json.loads(x) for x in (out/'arrivals.jsonl').read_text().splitlines()];poses=[json.loads(x) for x in (out/'poses.jsonl').read_text().splitlines()];solves=load(out/'solves.json');pubs=load(out/'publications.json')
    statuses=collections.Counter(r['status'] for r in solves);reasons=collections.Counter(r['original_reason'] for r in solves)
    valid=[r for r in solves if r['final_geometry_available']];ranked=sorted(valid,key=lambda r:r['available_c4_observed_range'])
    selected=[]
    def add(frame,kind):
        if frame not in [i for i,_ in selected]:selected.append((frame,kind))
    if ranked:
        for row in ranked[-2:]:add(row['frame'],'Лучший горизонт')
        for row in ranked[max(0,len(ranked)//2-1):len(ranked)//2+1]:add(row['frame'],'Средний горизонт')
        for row in ranked[:2]:add(row['frame'],'Короткий горизонт')
    refusals=[r for r in solves if not r['final_geometry_available']]
    grouped={}
    for r in refusals:grouped.setdefault(r['original_reason'],r)
    for r in list(grouped.values())[:2]:add(r['frame'],'Отказ поиска')
    for i in summary['invalid_pose_frames'][:1]+summary['invalid_pose_frames'][-1:]:add(i,'Нет причинной позы')
    selected_ids=set(i for i,_ in selected);sample_data={};gallery=out/'examples';gallery.mkdir(exist_ok=True)
    conn=sqlite3.connect(a.db.resolve().as_uri()+'?mode=ro',uri=True);conn.execute('PRAGMA query_only=ON')
    # Report export is not timed as train inference. Full source rows are preserved.
    export_begin=time.perf_counter();map_path=out/'stitched_cloud.las';h=header();total=0
    with laspy.open(map_path,mode='w',header=h) as writer:
        for i,pose in enumerate(poses):
            row=arrivals[i];blob=conn.execute('SELECT data FROM messages WHERE id=?',(row['message_id'],)).fetchone()[0];meta,p=decode(blob)
            xyz=np.column_stack([p[k] for k in ('x','y','z')]);ids=np.flatnonzero(np.isfinite(xyz).all(axis=1)&np.any(xyz!=0,axis=1)).astype(np.uint32)
            if i in selected_ids:sample_data[i]=(xyz[ids].copy(),p.copy(),ids,meta)
            if pose['pose'] is not None:
                P=np.asarray(pose['pose']);world=xyz[ids].astype(np.float64)@P[:3,:3].T+P[:3,3]
                writer.write_points(raw_las(h,world,p,ids,i,row['message_id']));total+=len(ids)
            if i%100==0:print('MAP_EXPORT',i+1,'/',len(poses),'points',total,flush=True)
    conn.close();map_seconds=time.perf_counter()-export_begin
    # All successfully published final model curves in map space, separately tagged.
    gh=header(True);geometry_path=out/'predicted_geometry.las';geometry_points=0
    colors={1:(65535,10000,10000),2:(5000,65535,15000),3:(0,50000,65535)}
    def generated(writer,pred,frame,message):
        nonlocal geometry_points
        q=int(pred['q']);left=pred['pair'][:,1 if q>0 else 0];right=pred['pair'][:,0 if q>0 else 1]
        for component,xyz in [(1,left),(2,right),(3,pred['C'])]:
            xyz=sample_polyline(xyz);rows=laspy.ScaleAwarePointRecord.zeros(len(xyz),header=gh);rows.x=xyz[:,0];rows.y=xyz[:,1];rows.z=xyz[:,2]
            rows.frame_index[:]=frame;rows.point_index[:]=-1;rows.message_id[:]=message;rows.intensity_raw[:]=np.nan;rows.geometry_generated[:]=1;rows.geometry_component[:]=component
            for channel,value in zip(('red','green','blue'),colors[component]):rows[channel][:]=value
            writer.write_points(rows);geometry_points+=len(rows)
    with laspy.open(geometry_path,mode='w',header=gh) as writer:
        for file in sorted((out/'published').glob('*.npz')):
            i=int(file.stem)
            with np.load(file) as z:generated(writer,{k:z[k] for k in z.files},i,arrivals[i]['message_id'])
    cards=[];by_frame={r['frame']:r for r in solves}
    for i,kind in selected:
        xyz,points,ids,meta=sample_data[i];row=by_frame.get(i);pred=None;path=out/'solves'/f'{i:06d}'/'prediction.npz'
        if path.exists():
            with np.load(path) as z:pred={k:z[k] for k in z.files}
        name=f'{i:06d}';laspath=gallery/(name+'.las')
        with laspy.open(laspath,mode='w',header=gh) as writer:
            writer.write_points(raw_las(gh,xyz,points,ids,i,arrivals[i]['message_id']))
            if pred is not None:generated(writer,pred,i,arrivals[i]['message_id'])
        # Display-only downsampling; neither registration nor detector reads these views.
        mask=(np.linalg.norm(xyz,axis=1)<140);display=xyz[mask];display=display[::max(1,len(display)//35000)]
        fig,axes=plt.subplots(1,2,figsize=(15,6));axes[0].scatter(-display[:,1],display[:,0],s=.3,c='#637a90',alpha=.28,rasterized=True);axes[1].scatter(-display[:,1],display[:,2],s=.3,c='#637a90',alpha=.28,rasterized=True)
        if pred is not None:
            q=int(pred['q']);ls=pred['pair'][:,1 if q>0 else 0];rs=pred['pair'][:,0 if q>0 else 1]
            for curve,color,label in [(ls,'#e54545','Левый ходовой'),(rs,'#19a052','Правый ходовой'),(pred['C'],'#09aac8','Контактный')]:
                axes[0].plot(-curve[:,1],curve[:,0],color=color,lw=2,label=label);axes[1].plot(-curve[:,1],curve[:,2],color=color,lw=2)
            axes[0].legend(loc='best');extent=max(12,min(130,float(np.max(-pred['C'][:,1]))+8))
        else:extent=50
        axes[0].set(xlim=(-5,extent),ylim=(-8,8),xlabel='−Y датчика, м',ylabel='X датчика, м',title='Вид сверху');axes[1].set(xlim=(-5,extent),ylim=(-6,6),xlabel='−Y датчика, м',ylabel='Z датчика, м',title='Высота')
        for ax in axes:ax.grid(alpha=.2)
        status=poses[i]['status'] if row is None else row['status'];reason=poses[i].get('reason','') if row is None else row['original_reason'];horizon=None if row is None else row['available_c4_observed_range']
        horizon_label=f'{horizon:.2f} м' if horizon is not None else 'нет позы'
        fig.suptitle(f'{kind} · T={i} · {status}\n{reason}; C4 horizon: {horizon_label}',fontsize=12);fig.tight_layout();fig.savefig(gallery/(name+'.png'),dpi=130);plt.close(fig)
        cards.append(dict(frame=i,kind=kind,status=status,reason=reason,horizon=horizon,png='examples/'+name+'.png',las='examples/'+name+'.las'))
    # Timing and availability across the entire source, not only solved frames.
    t=np.array([(r['bag_time_ns']-arrivals[0]['bag_time_ns'])/1e9 for r in arrivals]);fig,ax=plt.subplots(3,1,figsize=(14,10),sharex=True)
    ax[0].plot(t,[r['registration_ms'] for r in arrivals],label='ICP + preparation',lw=.7);ax[0].plot(t,[r['read_ms'] for r in arrivals],label='SQLite / bind read',lw=.7);ax[0].axhline(100,c='r',ls='--',label='100 ms budget');ax[0].set(ylabel='мс');ax[0].legend()
    ax[1].plot(t,[r['deadline_ms']/1000 for r in arrivals]);ax[1].set(ylabel='Накопленная задержка, с')
    ax[2].plot(t,[r['snapshot']['remaining_horizon'] if r.get('snapshot') else np.nan for r in arrivals],lw=.8,label='Оставшийся горизонт');ax[2].scatter(t[summary['invalid_pose_frames']],np.zeros(len(summary['invalid_pose_frames'])),c='r',s=14,label='Нет позы');ax[2].set(ylabel='м',xlabel='Время записи, с');ax[2].legend()
    for x in ax:x.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'timeline.png',dpi=120);plt.close(fig)
    export=dict(map_file=map_path.name,map_points=total,map_bytes=map_path.stat().st_size,map_export_seconds=map_seconds,geometry_file=geometry_path.name,geometry_bytes=geometry_path.stat().st_size,excluded_pose_frames=summary['invalid_pose_frames'],raw_points_rule='all finite nonzero source points with accepted causal pose; no thinning; no labels fabricated',examples=cards)
    save(out/'EXPORTS.json',export)
    timing=summary['timing'];f=lambda x:f'{x:.2f}'
    text=f'''# ROS2 replay: cloud_with_fake_obj

Все {summary['frames']} сообщений прочитаны последовательно. Сшивка заново рассчитана по исходным sensor-frame облакам; готовые позы, карта и GT не передавались в inference. Frozen geometry_pipeline_v1.0.0 не изменён.

## Итог

- Длительность записи: {f(summary['recording_seconds'])} с. Фактическая обработка с дренированием: {f(summary['wall_seconds'])} с.
- Причинная поза доступна: {summary['valid_poses']} / {summary['frames']}. Недоступны индексы: {summary['invalid_pose_frames']} (нумерация с 0).
- Завершено полных попыток геометрии: {summary['completed_solves']}; доступная геометрия: {summary['available_solves']}; наблюдалось публикаций: {summary['published']}.
- Доля входных кадров с положительным оставшимся горизонтом: {100*summary['positive_horizon_fraction']:.2f}%.
- Полный путь выдерживает 10 Hz: **{'PASS' if summary['realtime_10hz_deadline_pass'] else 'FAIL'}**. Входные кадры не выбрасывались; при недостатке скорости очередь отстаёт от времени записи. Это честный результат замера, а не ускоренный playback.

## Задержки

| Операция | median, мс | p95, мс |
|---|---:|---:|
'''
    for key,title in [('read_ms','Чтение SQLite из Windows bind mount'),('decode_ms','Декодирование PointCloud2'),('registration_ms','Сшивка ICP + её подготовка'),('frame_data_ms','Raw → map FrameData'),('core_arrival_ms','Обслуживание arrival ядром'),('deadline_ms','От планового прихода до подачи ядру')]:
        s=timing[key];text+=f"| {title} | {f(s['median'])} | {f(s['p95'])} |\n"
    for key,title in [('geometry_solve_ms','Полное решение геометрии, включая отказы'),('c4_ms','C4, включая отказы'),('publication_latency_ms','От планового прихода исходного облака T до публикации')]:
        s=summary[key];text+=f"| {title} | {f(s['median'])} | {f(s['p95'])} |\n" if s['n'] else ''
    text+='''
Сшивка и геометрические workers выполнялись одновременно; export результатов вынесен в отдельный поток. Большой LAS экспорт выполнен после inference и в приведённые задержки не входит. SQLite из файловой системы Windows не имитирует стоимость доставки живого ROS-сообщения; строка чтения выделена отдельно. Но пересчёт ICP уже входит в полный замер. Однокадровое ожидание T+1 — только необходимая задержка данных; к ней прибавляются очередь и вычисления.

Положительный оставшийся горизонт измерен относительно последнего ОБРАБОТАННОГО кадра. При накопленной очереди это не означает доступность актуальной геометрии для реального положения поезда по настенным часам. Конец записи не ускорялся, устаревшие входные кадры не выбрасывались. Публикации учитывают также завершение последних активных задач при закрытии сессии; при пропуске позы наружу для такого кадра выдаётся unavailable.

## Причинность и сшивка

Метод тот же, что в прежней обработке этой записи: keyframe point-to-plane ICP, 0.22 м voxel / 1.2–140 м только для registration, robust residuals, проверка качества, восстановление связи после затенения с motion prior и несколькими начальными приближениями. Численные ICP-функции прочитаны непосредственно из неизменённых register.py / register_occlusion.py. В полном облаке для rail detection прореживания нет.

Удалена только несовместимая с online-режимом ретроспективная интерполяция: плохая поза остаётся недоступной навсегда. Текущая восстановленная поза использует текущий scan и прошлые references; предыдущие пропуски не заполняются. При отказе сшивки геометрическая сессия закрывается, последняя кривая не переносится через неизвестное движение. После восстановления начинается новая сессия. Последний кадр не решается без T+1.

На каждом solve сохранён аудит: cloud indices <=T и >=T−11, pose prefix <=T+1, dispatch после появления T+1. Готовые регистрации не монтировались в контейнер. Изначальная база открыта mode=ro; dataset и geometry core не изменены. Сохранение контрольных результатов не является входом алгоритма.

## Отказы и ограничения

'''
    text+='Статусы решений: '+json.dumps(statuses,ensure_ascii=False)+'.\n\nПричины завершения/отказа:\n\n'
    for reason,n in reasons.most_common():text+=f'- `{reason}`: {n}.\n'
    text+='''
Причина остановки C4 не равна отказу всей геометрии. `CANDIDATES_REJECTED` означает, что дальнейшие кандидаты не прошли замороженные проверки формы/поддержки/согласованности. `NO_POINTS` и `TOO_FEW_SUPPORT` означают недостаточную поддержку в следующем окне. `LOCAL_FRAME_DEGENERATE` — недостаточную пространственную протяжённость/число подтверждённых точек для нового локального направления; на этом месте продолжение прекращается. `OVERLAP_FIRST` — несогласованность перекрытия, `TENTATIVE_UNCONFIRMED` — неподтверждённое продолжение, `ORIENTATION_JUMP` — слишком резкий поворот локальной рамки. Уже построенная часть может остаться AVAILABLE.

Начальные `CAUSAL_POSE_BASELINE_BELOW_50CM` относятся к недостаточному накопленному движению; будущий горизонт ожидания не расширялся. `STEP2_NOT_FOUND:no_supported_candidate` означает отсутствие принятого контактного кандидата в bootstrap. Синтетическая запись содержит перекрытия обзора; отказ ICP на них не исправлялся предположением о безошибочной траектории.

Здесь нет независимого GT ни для траектории, ни для головок рельсов. Поэтому «лучший / средний / худший» в галерее означает наблюдаемый горизонт и наличие результата, а не подтверждённую точность. ICP residual не является абсолютной ошибкой положения. Короткий C4 horizon может оставлять seed-supported final geometry; её нельзя считать подтверждённой на всём протяжении.

## Сохранённые файлы

- `stitched_cloud.las`: полное сшитое облако из кадров с принятой причинной позой. Исходные point_index, frame_index, message_id и float32 intensity сохранены. XYZ кодированы 0.1 мм, как в прежних LAS-экспортах; это не точность измерения.
- `predicted_geometry.las`: опубликованные модельные точки в общей системе; левый ходовой красный, правый зелёный, контактный голубой. `geometry_generated=1`, это модель, а не новые измерения.
- `examples/*.las`: исходный кадр и цветная геометрия в системе датчика T, пригодны для 3D-просмотра. Без позы сохранено только исходное облако.
- `solves/`: результат каждого завершённого решения, prediction NPZ, provenance и summary; `published/`: наблюдавшиеся опубликованные map-space curves.
- `poses.jsonl`, `arrivals.jsonl`, `solves.json`, `events.json`, `publications.json`, `causality_audit.json`, `SUMMARY.json`, `EXPORTS.json`.

Отсутствующие ring и per-point timestamps не придуманы. Время сообщения и sensor header связываются с точками через frame_index / message_id и sidecar-файлы. Нулевые placeholders и кадры без принятой позы остаются в исходном DB3, но не подмешиваются в зарегистрированную карту. Pipeline не является детектором синтетических препятствий.
'''
    (out/'REPORT.md').write_text(text,encoding='utf-8')
    cards_html=''.join(f'<article><h2>{html.escape(c["kind"])} · T={c["frame"]}</h2><p>{html.escape(c["status"])} · {html.escape(c["reason"])}</p><img loading="lazy" src="{c["png"]}"><p><a href="{c["las"]}">LAS с цветной геометрией</a></p></article>' for c in cards)
    page=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>ROS2 causal replay</title><style>body{{font:16px system-ui;background:#eef3f8;color:#19334a;max-width:1450px;margin:35px auto;padding:0 20px}}article,section{{background:white;padding:25px;border-radius:15px;margin:24px 0}}img{{max-width:100%}}a{{color:#087aa4}}.big{{font-size:22px}}</style><h1>ROS2 → причинная сшивка → frozen geometry</h1><section class="big">{summary['frames']} кадров · {summary['valid_poses']} принятых поз · {summary['published']} публикаций<br>Запись {summary['recording_seconds']:.1f} с → обработка {summary['wall_seconds']:.1f} с<br>10 Hz end-to-end: {'PASS' if summary['realtime_10hz_deadline_pass'] else 'FAIL — задержка накапливается'}<p><a href="REPORT.md">Полный отчёт</a> · <a href="SUMMARY.json">Численные результаты</a></p><p>Облака только до T, позы только до T+1. Готовая карта и разметка не используются. Сравнение примеров — по горизонту, без GT-оценки точности.</p></section><img src="timeline.png">{cards_html}</html>'''
    (out/'gallery.html').write_text(page,encoding='utf-8')
    with laspy.open(map_path) as check:assert check.header.point_count==total
    save(out/'EXPORT_COMPLETE.json',dict(map_points=total,examples=len(cards),source_unchanged=True,report='REPORT.md',gallery='gallery.html'))
    print('EXPORT_COMPLETE',total,'map points',len(cards),'examples',flush=True)

if __name__=='__main__':main()
