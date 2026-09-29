"""Post-run audit and timing comparison for the every-fifth-frame experiment."""
import ast
import hashlib
import html
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / '.runtime'))
import numpy as np
from verify_trial import verify, equal


def load(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def lines(p):
    return [json.loads(s) for s in p.read_text().splitlines()]


def stats(a):
    return dict(n=len(a), median_ms=float(np.median(a)), p95_ms=float(np.quantile(a, .95)),
                max_ms=float(max(a)), mean_ms=float(np.mean(a))) if a else {}


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    folder = HERE / 'results' / (sys.argv[1] if len(sys.argv)>1 else 'EVERY_FIFTH')
    verification = verify(folder)
    out = folder / 'payload'
    baseline = HERE / 'results/FULL_COPY_MAIN/payload'
    new, old = load(out / 'SUMMARY.json'), load(baseline / 'SUMMARY.json')
    stages, previous = load(folder / 'STAGES_MS.json'), load(baseline / 'STAGES_MS.json')
    poses, arrivals = lines(out / 'poses.jsonl'), lines(out / 'arrivals.jsonl')
    jobs = load(out / 'solves.json')
    oldjobs = {r['frame']: r for r in load(baseline / 'solves.json')}
    audit = load(out / 'causality_audit.json')['solves']
    assert new['copy_main']['solve_stride'] == 5 and new['copy_main']['solve_offset'] == 4
    assert [r['frame'] for r in arrivals] == list(range(1510))
    assert new['source_message_payload_sha256'] == old['source_message_payload_sha256']
    assert all(r['frame'] % 5 == 4 for r in jobs)
    # Reuse the actual frozen edge rule to verify every intermediate cloud lease.
    source = HERE / 'app/MVP/stages/05_1_contact_marching_temporal_fusion/src/fusion.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'valid_edge')
    scope = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    records = [dict(header_time_ns=p['header_time_ns'], pose_status=p['status'],
                    pose_uses_future=p['uses_future_pose'],
                    lidar_pose_in_folder=p['pose'] if p['pose'] is not None else np.full((4, 4), np.nan)) for p in poses]
    edge = lambda i: scope['valid_edge'](records[i], records[i+1])
    for job in audit:
        i = job['frame']; expected = [i]
        for k in range(i-1, max(-1, i-12), -1):
            if not edge(k):
                break
            expected.append(k)
        assert sorted(expected) == job['cloud_frames'], (i, expected, job)
    selected = list(range(4, len(poses), 5))
    solved = {r['frame'] for r in jobs}
    omissions = []
    for i in selected:
        if i in solved:
            continue
        reason = ('NO_T_PLUS_1' if i+1 >= len(poses) else
                  'NO_VALID_CAUSAL_EDGE_T_T_PLUS_1' if not edge(i) else
                  'EXISTING_MOVEMENT_GATE_OR_SCHEDULER')
        omissions.append(dict(frame=i, reason=reason))
    common = sorted(solved & oldjobs.keys())
    # Equal selected-frame numerical output, including structured NaN fields.
    for i in common:
        a, b = out / 'solves' / f'{i:06d}', baseline / 'solves' / f'{i:06d}'
        sa, sb = load(a / 'summary.json'), load(b / 'summary.json')
        for key in ('status', 'original_reason', 'final_geometry_available'):
            assert sa[key] == sb[key], (i, key)
        for name in ('prediction.npz', 'curve.npz', 'point_provenance.npz'):
            assert (a/name).exists() == (b/name).exists()
            if not (a/name).exists():
                continue
            with np.load(a/name) as x, np.load(b/name) as y:
                assert set(x.files) == set(y.files)
                for key in x.files:
                    xx, yy = x[key], y[key]
                    if xx.dtype.names:
                        assert xx.shape == yy.shape and xx.dtype == yy.dtype
                        assert all(equal(xx[n], yy[n]) for n in xx.dtype.names), (i, name, key)
                    else:
                        assert equal(xx, yy), (i, name, key)
    freeze_count = 0
    manifest = load(HERE / 'ORIGINAL_RELEASE_MANIFEST.json')
    for group in ('runtime_files', 'build_files', 'release_artifacts'):
        for name, item in manifest[group].items():
            p = ROOT/name
            if p.exists():
                assert sha(p) == item['sha256']; freeze_count += 1
    locked = load(HERE/'MANIFEST.json')
    for name, digest in locked.items():
        if name.startswith(('app/', 'registration/')):
            assert sha(HERE/name) == digest, name
    publications = load(out/'publications.json')
    intervals = [(b['clock']-a['clock'])*1000 for a, b in zip(publications, publications[1:])]
    new_by_frame = {r['frame']: r for r in jobs}
    comparison = dict(pass_=True, all_1510_inputs_preserved=True, input_payload_hash_equal=True,
                      all_history_cloud_leases_exact=True, selected_frames=selected, omissions=omissions,
                      common_with_full_copy_exact=len(common), original_verification=verification,
                      frozen_files_checked=freeze_count, copied_geometry_unchanged=True,
                      speedup=old['wall_seconds']/new['wall_seconds'],
                      producer=stats([r['producer_ms'] for r in arrivals]),
                      registration=stats([r['registration_ms'] for r in arrivals]),
                      publication_intervals=stats(intervals),
                      matched_geometry_old=stats([oldjobs[i]['wall_ms'] for i in common]),
                      matched_geometry_new=stats([new_by_frame[i]['wall_ms'] for i in common]),
                      final_arrival_lag_ms=arrivals[-1]['deadline_ms'])
    (folder/'COMPARISON.json').write_text(json.dumps(comparison, indent=2)+'\n')
    mapping = [('Чтение DB3','read_ms'), ('Декодирование','decode_ms'), ('Контрольная сумма','payload_hash_ms'),
               ('Сшивка целиком','registration_ms'), ('Воксели сшивки','registration_voxel_sample_ms'),
               ('ICP','registration_icp_ms'), ('Нормали (когда нужны)','registration_normals_ms'),
               ('Подготовка FrameData','frame_data_ms'), ('Обработка входного кадра целиком','producer_ms'),
               ('Приём / индекс / планировщик','core_arrival_ms'), ('STEP1','T_STEP1_ms'), ('STEP2','T_STEP2_ms'),
               ('C4 с подготовкой истории','T_C4_MARCHING_INCLUSIVE_HISTORY_IO_ms'),
               ('Контракт геометрии','T_GEOMETRY_CONTRACT_ms'), ('3D seed adapter','T_SEED_ADAPTER_ms'),
               ('C4 smoothing','T_C4_SMOOTH_ms'), ('STEP6','T_STEP6_RAIL_RECONSTRUCTION_ms'),
               ('Provenance','T_PROVENANCE_ms'), ('Сериализация','T_SERIALIZATION_ms'),
               ('Геометрия целиком','solve_wall_ms')]
    table = []
    for label, key in mapping:
        if key not in stages['timings'] or key not in previous['timings']:
            continue
        a, b = previous['timings'][key], stages['timings'][key]
        table.append([label, *[f'{v:.2f}' for v in (a['median_ms'], a['p95_ms'], b['median_ms'], b['p95_ms'])]])
    headers = ['Этап', 'Прежде median, мс', 'Прежде p95, мс', 'Каждый 5-й median, мс', 'Каждый 5-й p95, мс']
    mdtable = '| '+' | '.join(headers)+' |\n|---|---:|---:|---:|---:|\n'+'\n'.join('| '+' | '.join(r)+' |' for r in table)
    matched_old, matched_new = comparison['matched_geometry_old'], comparison['matched_geometry_new']
    omitted = ', '.join(f"{r['frame']} ({r['reason']})" for r in omissions) or 'нет'
    text = f'''# COPY_MAIN — геометрия на каждом пятом кадре

Обработаны все {new['frames']} кадров. Время полного прогона: **{old['wall_seconds']*1000:.0f} → {new['wall_seconds']*1000:.0f} мс**, ускорение **{comparison['speedup']:.2f}×**. Длительность входной записи — {new['recording_seconds']*1000:.0f} мс.

## Режим

Чтение, свежая причинная сшивка и индексирование выполняются для каждого кадра. Запуск геометрии: 5-й, 10-й, 15-й и далее, индексы с нуля 4, 9, 14… Окно T−11…T содержит все промежуточные доступные кадры; на старте и после разрыва окно короче. Разрешена только поза T+1; облако T+1 запрещено. Сохраняются прежние проверки движения, разрывов, отказов и правило latest pending. Между расчётами используется последнее принятое решение; при разрыве оно сбрасывается, как прежде.

Два geometry worker × два candidate thread, один ICP query worker. Изменён только допуск запуска по номеру кадра. Геометрический код, сшивка, пороги, точки и окно сохранены.

## Полный прогон

| Показатель | Прежде COPY_MAIN | Каждый пятый |
|---|---:|---:|
| Обработанные кадры | {old['frames']} | {new['frames']} |
| Принятые позы | {old['valid_poses']} | {new['valid_poses']} |
| Завершённые расчёты | {old['completed_solves']} | {new['completed_solves']} |
| Доступные решения | {old['available_solves']} | {new['available_solves']} |
| Опубликованные решения | {old['published']} | {new['published']} |
| Публикаций за секунду реального времени прогона | {old['publish_hz_wall']:.3f} | {new['publish_hz_wall']:.3f} |
| Задержка публикации p95, мс | {old['publication_latency_ms']['p95']:.0f} | {new['publication_latency_ms']['p95']:.0f} |
| Задержка публикации max, мс | {old['publication_latency_ms']['max']:.0f} | {new['publication_latency_ms']['max']:.0f} |

Последний входной кадр обработан с отставанием {comparison['final_arrival_lag_ms']:.0f} мс от исходного расписания. Интервал между публикациями: median {comparison['publication_intervals']['median_ms']:.0f} мс, p95 {comparison['publication_intervals']['p95_ms']:.0f} мс. 2 Гц — заданная частота расчётов по времени источника, а не гарантия выдачи без задержки.

## Этапы

{mdtable}

Медианы вложенных и параллельных таймеров нельзя складывать. Нормали и восстановление регистрации считаются только там, где запускались. Сравнение всех geometry jobs включает разные наборы кадров. На {len(common)} одинаковых кадрах время геометрии: median {matched_old['median_ms']:.2f} → {matched_new['median_ms']:.2f} мс; p95 {matched_old['p95_ms']:.2f} → {matched_new['p95_ms']:.2f} мс. Это сравнение времени в двух полноценных прогонах, не отдельный тест одного ядра.

## Отставание и ограничение скорости

Средняя обработка одного входного кадра в producer: {comparison['producer']['mean_ms']:.2f} мс; средняя сшивка: {comparison['registration']['mean_ms']:.2f} мс. Входной интервал около 100 мс. Снижение числа расчётов геометрии разгружает процессор, но чтение и сшивку всех кадров не отменяет. {'В этом прогоне накопилось отставание; режим ещё не обеспечивает непрерывную обработку 10 Гц.' if comparison['final_arrival_lag_ms'] > 1000 else 'Достигнутую задержку следует оценивать по таблице, включая хвосты распределения.'}

Сопоставление сделано с предыдущим полным прогоном на том же файле и той же конфигурации потоков. Это по одному полному замеру каждого режима; фоновые нагрузки и вариативность времени не изолированы повторениями. Изменение времени отдельных этапов согласуется со снижением конкуренции за процессор, но не является отдельным доказательством причины.

## Проверки

Все 1510 входных сообщений прочитаны; контрольная сумма payload совпала. Все позы точно совпали с исходным прогоном. Проверено {len(audit)} запусков: только каждый пятый T, точная полная доступная история T−11…T и отсутствие будущих облаков. На {len(common)} общих кадрах с предыдущим COPY_MAIN прогнозы, кривые и provenance совпали точно. С исходным замороженным прогоном также проверены {verification['common_solves']} общих кадров. Frozen-файлы ({freeze_count}) и геометрический код COPY_MAIN не менялись.

Из {len(selected)} выбранных номеров не были рассчитаны: {omitted}. Последнему кадру записи недоступна поза T+1. Исходные разрывы регистрации не интерполируются. Новая точность по независимой разметке не измерялась; совпадение выбранных прогнозов не доказывает одинаковую точность удерживаемого результата между более редкими обновлениями.

LAS не создавались. Копирование и SHA256 файла DB3 до запуска прогревают файловый кэш и исключены из replay; предварительного вычисления поз или геометрии нет. JSON/NPZ пишутся внутри Linux; экспорт архива после прогона исключён. Это replay сериализованных ROS2 PointCloud2 из DB3 без DDS publisher/subscriber.

## Повторение

`./COPY_MAIN/run.ps1 -RunName my_every_fifth -Count 1510 -SolveStride 5 -SolveOffset 4`

Без этих параметров сохраняется прежний режим. Машиночитаемые результаты: SUMMARY.json, STAGES_MS.json, COMPARISON.json, VERIFICATION.json, payload/causality_audit.json. Временная копия DB3 удаляется вместе с контейнером через --rm; состояние удаления зафиксировано отдельно в CONTAINER_CLEANUP.json.
'''
    (folder/'REPORT.md').write_text(text, encoding='utf-8')
    # Self-contained static report with the same headings and comparison tables.
    output = []
    in_table = False
    for line in text.splitlines():
        if line.startswith('|'):
            if not in_table:
                output.append('<table>'); in_table = True
            if line.startswith('|---'):
                continue
            output.append('<tr>'+''.join('<td>'+html.escape(c.strip())+'</td>' for c in line.strip('|').split('|'))+'</tr>')
        else:
            if in_table:
                output.append('</table>'); in_table = False
            if line.startswith('# '): output.append('<h1>'+html.escape(line[2:])+'</h1>')
            elif line.startswith('## '): output.append('<h2>'+html.escape(line[3:])+'</h2>')
            elif line: output.append('<p>'+html.escape(line).replace('**', '')+'</p>')
    page = '<!doctype html><meta charset="utf-8"><title>COPY_MAIN · Каждый пятый кадр</title><style>body{font:16px/1.55 system-ui;max-width:1200px;margin:36px auto;padding:0 24px;color:#23374d;background:#f8fafc}table{border-collapse:collapse;width:100%;background:white}td{border:1px solid #d7e0e8;padding:8px}tr:first-child{font-weight:bold;background:#e5edf5}h1,h2{color:#143456}</style>'+''.join(output)
    (folder/'report.html').write_text(page, encoding='utf-8')
    print(json.dumps({k:v for k,v in comparison.items() if k not in ('selected_frames',)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
