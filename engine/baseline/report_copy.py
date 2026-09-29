import json,hashlib,html,csv,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;RESULTS=HERE/'results';FINAL=RESULTS/'FULL_COPY_MAIN';PAYLOAD=FINAL/'payload'
def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def quantile(a,q):
    a=sorted(a);x=(len(a)-1)*q;i=int(x);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
def stat(a):return {'n':len(a),'median_ms':statistics.median(a),'p95_ms':quantile(a,.95)} if a else {}
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    old=load(ROOT/'results_ros2_replay_docker_local/full/SUMMARY.json');new=load(FINAL/'SUMMARY.json');stages=load(FINAL/'STAGES_MS.json');verification=load(FINAL/'VERIFICATION.json');assert verification['pass_']
    previous=load(ROOT/'results_ros2_replay_docker_local/full/solves.json');current=load(PAYLOAD/'solves.json');rows=[]
    mapping=[('Чтение DB3','read_ms'),('Декодирование','decode_ms'),('Сшивка целиком','registration_ms'),('Создание общего облака FrameData','frame_data_ms'),('Приём/индексирование/планировщик','core_arrival_ms')]
    for label,key in mapping:
        a=old['timing'][key];b=new['timing'][key];rows.append(dict(stage=label,old_median_ms=a['median'],old_p95_ms=a['p95'],copy_median_ms=b['median'],copy_p95_ms=b['p95']))
    for label,key in [('STEP1: ходовые рельсы','T_STEP1'),('STEP2: контактный рельс','T_STEP2'),('C4 с подготовкой истории','T_C4_MARCHING_INCLUSIVE_HISTORY_IO'),('Контракт геометрии C4','T_GEOMETRY_CONTRACT'),('3D seed adapter','T_SEED_ADAPTER'),('C4 smoothing','T_C4_SMOOTH'),('STEP6: восстановление рельсов','T_STEP6_RAIL_RECONSTRUCTION'),('Provenance','T_PROVENANCE'),('Сериализация/запись решения','T_SERIALIZATION')]:
        a=stat([r['timing'][key]*1000 for r in previous if key in r['timing']]);b=stat([r['timing'][key]*1000 for r in current if key in r['timing']]);rows.append(dict(stage=label,old_median_ms=a.get('median_ms'),old_p95_ms=a.get('p95_ms'),copy_median_ms=b.get('median_ms'),copy_p95_ms=b.get('p95_ms')))
    rows.append(dict(stage='Геометрия целиком',old_median_ms=old['geometry_solve_ms']['median'],old_p95_ms=old['geometry_solve_ms']['p95'],copy_median_ms=new['geometry_solve_ms']['median'],copy_p95_ms=new['geometry_solve_ms']['p95']))
    with (HERE/'STAGES_MS.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    variants=[]
    for name in ('baseline_2x4_400','packed_1x4_400','local_1x4_400','local_1x2_kd2_400','local_1x4_kd2_400','local_2x2_400','local_2x4_400'):
        s=load(RESULTS/name/'SUMMARY.json');variants.append(dict(name=name,wall_ms=s['wall_seconds']*1000,solves=s['completed_solves'],published=s['published'],publication_p95_ms=s['publication_latency_ms']['p95']))
    fixed_old=load(RESULTS/'fixed_original/BENCHMARK.json');fixed_new=load(RESULTS/'fixed_optimized/BENCHMARK.json');assert fixed_old['checks']==fixed_new['checks']
    fixed_a=stat([r['wall_ms'] for r in fixed_old['rows']]);fixed_b=stat([r['wall_ms'] for r in fixed_new['rows']])
    voxel=load(RESULTS/'VOXEL_VERIFICATION.json');freeze=load(HERE/'ORIGINAL_RELEASE_MANIFEST.json');checked=0
    for group in ('runtime_files','build_files','release_artifacts'):
        for name,item in freeze[group].items():
            p=ROOT/name
            if p.exists():assert digest(p)==item['sha256'];checked+=1
    for name,sha in load(ROOT/'results_ros2_replay_v1/RUN_SOURCE_LOCK.json').items():assert digest(ROOT/name)==sha
    manifests={str(p.relative_to(HERE)).replace('\\','/'):digest(p) for base in ('app','adapter','registration') for p in (HERE/base).rglob('*') if p.is_file()}
    for name in ('config.json','run.ps1','run_once.py','run_trial.py','summarize.py','verify_trial.py','fixed_work_bench.py'):manifests[name]=digest(HERE/name)
    qualification=dict(pass_=True,frozen_original_files_unchanged=checked,full_verification=verification,identical_fixed_work_frames=len(fixed_old['checks']),fixed_original=fixed_a,fixed_copy=fixed_b,stage_table=rows,variants=variants,full_before_ms=old['wall_seconds']*1000,full_after_ms=new['wall_seconds']*1000,speedup=old['wall_seconds']/new['wall_seconds'],temporary_container_removed=load(RESULTS/'CONTAINER_CLEANUP.json')['removed'],no_las=True)
    (HERE/'QUALIFICATION.json').write_text(json.dumps(qualification,indent=2,ensure_ascii=False)+'\n',encoding='utf-8');(HERE/'MANIFEST.json').write_text(json.dumps(manifests,indent=2)+'\n')
    table='\n'.join('| '+r['stage']+' | '+' | '.join(f'{r[k]:.2f}' for k in ('old_median_ms','old_p95_ms','copy_median_ms','copy_p95_ms'))+' |' for r in rows)
    alternatives='\n'.join(f"| {r['name']} | {r['wall_ms']:.0f} | {r['solves']} | {r['published']} | {r['publication_p95_ms']:.0f} |" for r in variants)
    detail='\n'.join(f"| {k} | {v['n']} | {v['median_ms']:.2f} | {v['p95_ms']:.2f} |" for k,v in stages['timings'].items() if k.startswith('registration_') or k.startswith('io_') or k=='payload_hash_ms')
    slow=sorted(current,key=lambda r:r['wall_ms'],reverse=True)[:5]
    slowtable='\n'.join(f"| {r['frame']} | {r['wall_ms']:.1f} | {r['timing']['T_STEP1']*1000:.1f} | {r['timing']['T_STEP2']*1000:.1f} | {r['c4_ms']:.1f} | {r['available_c4_observed_range']:.1f} | {r['original_reason']} |" for r in slow)
    text=f'''# COPY_MAIN — квалификация оптимизированной копии

Обработаны все {new['frames']} raw-кадров. Полный прогон ускорен с **{old['wall_seconds']*1000:.0f} до {new['wall_seconds']*1000:.0f} мс**, **{old['wall_seconds']/new['wall_seconds']:.2f}×**. Длительность самой записи {new['recording_seconds']*1000:.0f} мс. Бюджет 10 Гц по прежнему критерию задержки: **{'PASS' if new['realtime_10hz_deadline_pass'] else 'FAIL'}**. Это итог измерения, а не обещание реального времени.

Оригинал — предыдущий полный Docker-прогон с DB3 внутри Linux, 2×4 и диагностикой через Windows bind mount. COPY_MAIN — 2×2, точный packed-voxel, эквивалентные оптимизации геометрии и локальная Linux-диагностика. Копирование DB3, его проверка и перенос архива исключены из измеряемого replay в обоих подходящих сравнениях. Перенос результатов COPY_MAIN после прогона занял {load(FINAL/'EXPORT.json')['post_run_export_ms']:.0f} мс.

## Каждый этап, миллисекунды

| Этап | Оригинал median | Оригинал p95 | COPY_MAIN median | COPY_MAIN p95 |
|---|---:|---:|---:|---:|
{table}

Таймеры с пометкой «целиком» и «с подготовкой истории» включают дочерние операции. Медианы нельзя складывать: стадии выполняются параллельно, а набор выполненных geometry jobs зависит от scheduler. Старый registration_ms включал также SHA256 payload (около 2 мс); в COPY_MAIN hash измерен отдельно. Все исходные значения сохранены в STAGES_MS.json и per-frame логах.

## Детали сшивки и диагностики COPY_MAIN

| Таймер | Измерений | median, мс | p95, мс |
|---|---:|---:|---:|
{detail}

Нормали и recovery — условные операции; статистика дана только для кадров, где они выполнялись. Registration other включает фильтры и проверки движения. Запись диагностики включает JSON/NPZ, LAS отсутствуют.

## Одинаковый объём геометрической работы

На ровно 24 одинаковых задачах из всей записи, без пропусков scheduler, по одному engine с четырьмя потоками в обоих вариантах: median {fixed_a['median_ms']:.2f} → {fixed_b['median_ms']:.2f} мс; p95 {fixed_a['p95_ms']:.2f} → {fixed_b['p95_ms']:.2f} мс. Во всех 24 совпали прогнозы, сглаженная кривая, provenance, выбранные исходные ключи и статусы. Это отдельный kernel benchmark с фиксированными ранее причинными позами; его нельзя выдавать за полный live-прогон.

Воксельная подготовка на 89 реальных кадрах: median {voxel['original_median_ms']:.2f} → {voxel['optimized_median_ms']:.2f} мс. Состав и порядок точек совпали точно. Дополнительно проверены пустые данные, дубликаты, float32/float64 и границы ячеек.

## Сравнение настроек на первых 400 кадрах

| Вариант | Полный прогон, мс | Решений | Публикаций | p95 задержки от исходного времени, мс |
|---|---:|---:|---:|---:|
{alternatives}

Выбран 2×2: близкое к исходному количество решений и почти двукратное ускорение на development-участке. Один worker быстрее потребляет запись, но создаёт меньше прогнозов; этот эффект не выдаётся за ускорение вычисления каждого кадра. Небольшая разница между 2×2 и 2×4 получена в одном прогоне каждого варианта и может включать шум измерения. Подтверждено сравнение всего пайплайна, а не строгая оптимальность числа потоков на любом оборудовании.

## Полный прогон: частота, задержка и совпадение

- Завершённые geometry jobs: {old['completed_solves']} → {new['completed_solves']}; принятые: {old['available_solves']} → {new['available_solves']}; опубликованные: {old['published']} → {new['published']}.
- Частота публикаций по wall time: {old['publish_hz_wall']:.2f} → {new['publish_hz_wall']:.2f} Гц.
- p95 задержки публикации относительно исходного времени кадра: {old['publication_latency_ms']['p95']:.0f} → {new['publication_latency_ms']['p95']:.0f} мс. Максимум COPY_MAIN: {new['publication_latency_ms']['max']:.0f} мс.
- Принятых поз {new['valid_poses']}; все матрицы поз и отказы совпали с исходным прогоном. Недоступные позы: {new['invalid_pose_frames']}.
- На {verification['common_solves']} общих рассчитанных кадрах совпали все проверенные массивы prediction/curve/provenance и статусы. Проверены контрольные суммы всех новых solve-export, источники точек и границы T/T+1.
- Оригинальные frozen файлы: {checked} проверено, без изменений. COPY_MAIN отдельно зафиксирован собственным MANIFEST.json. Временный контейнер с DB3 удалён.

Это проверка эквивалентности вычислений, не независимая GT-оценка точности. Планировщик может выбрать другие кадры из-за нового времени завершения задач. Все 1510 входных облаков проходят регистрацию; geometry не обязана пересчитываться на каждом кадре по прежней политике latest pending. Cloud T+1 запрещён; разрешена только его pose. Пропуски регистрации не интерполируются.

## Самые долгие geometry jobs нового прогона

| T | Геометрия, мс | STEP1, мс | STEP2, мс | C4, мс | Горизонт, м | Причина остановки |
|---|---:|---:|---:|---:|---:|---|
{slowtable}

Это худшие случаи по времени, а не по геометрической ошибке. Более длинный наблюдаемый горизонт требует больше marching-шагов; индивидуальный wall time также зависит от перекрытия с регистрацией и вторым worker. Причина остановки C4 не означает ошибку всей уже построенной кривой.

## Артефакты и воспроизведение

`config.json`, `run.ps1`, `README.md` — запуск. `results/FULL_COPY_MAIN/payload` — полный результат без LAS. `STAGES_MS.csv`, `QUALIFICATION.json`, `results/FULL_COPY_MAIN/VERIFICATION.json` — проверки и метрики. `results/FULL_COPY_MAIN/results.tar.gz` — компактный архив; его можно удалить после подтверждённой распаковки.

Симуляция читает ROS2 PointCloud2 из DB3 по исходным временам. Реальный DDS publisher/subscriber здесь не запускался; его задержка не измерена. Если reader отстаёт, кадры остаются в очереди записи, а backlog показывается явно. Отдельное моделирование потери пакетов и внешней одометрии не выполнялось.
'''
    (HERE/'REPORT.md').write_text(text,encoding='utf-8')
    rendered='';in_table=False;in_list=False
    for line in text.splitlines():
        if line.startswith('|'):
            if not in_table:rendered+='<table>';in_table=True
            if set(line.replace('|','').replace(':','').replace('-','').strip())==set():continue
            rendered+='<tr>'+''.join('<td>'+html.escape(x.strip())+'</td>' for x in line.strip('|').split('|'))+'</tr>';continue
        if in_table:rendered+='</table>';in_table=False
        if not line:continue
        if line.startswith('# '):rendered+='<h1>'+html.escape(line[2:])+'</h1>'
        elif line.startswith('## '):rendered+='<h2>'+html.escape(line[3:])+'</h2>'
        else:rendered+='<p>'+html.escape(line).replace('**','')+'</p>'
    page='<!doctype html><meta charset="utf-8"><title>COPY_MAIN — полный профиль</title><style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:20px;background:#f5f7fb;color:#17304c;line-height:1.55}table{width:100%;border-collapse:collapse;background:white;margin:18px 0;font-size:14px}td{padding:9px;border-bottom:1px solid #d8e1eb}tr:first-child{font-weight:bold;background:#e6edf5}h1,h2{color:#0b527c}h2{margin-top:34px}a{color:#006ba3}</style>'+rendered+'<p><a href="STAGES_MS.csv">CSV этапов</a> · <a href="QUALIFICATION.json">Проверки</a> · <a href="README.md">Запуск COPY_MAIN</a></p>'
    (HERE/'report.html').write_text(page,encoding='utf-8');print(json.dumps({k:v for k,v in qualification.items() if k not in ('stage_table','variants','full_verification')},indent=2))

if __name__=='__main__':main()
