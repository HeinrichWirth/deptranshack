from . import ROOT,OUT
from .run_full import RUNS
import long_common as lc
import numpy as np
import csv,html,json,time,pstats,platform
from collections import Counter
from fusion import valid_edge


def stats(values):
    x=np.asarray(values,float);x=x[np.isfinite(x)]
    return {f'p{p}':float(np.percentile(x,p)) if len(x) else None for p in (50,90,95,99,100)}


def main():
    runrows=[];framerows=[];availability=[];spacing=[];evaluations=[];errorrows=[];motion=[]
    for run in RUNS:
        modes={mode:lc.load(OUT/'runs'/variant/run/'INFERENCE_COMPLETE.json') for mode,variant in
            [('OLD','every_frame_python'),('NEW','scheduler_spatial')]}
        records=lc.frames(run);duration=(int(records[-1]['header_time_ns'])-int(records[0]['header_time_ns']))/1e9
        r=modes['NEW']['rows'];N=len(r);fresh=[x for x in r if x['fresh']];attempt=[x for x in r if x['solve_attempted']]
        reused=[x for x in r if x['low_motion_reuse']];stale=[x for x in r if x['stale']];missing=[x for x in r if not x['valid_geometry_state']]
        av=dict(run=run,raw_frames=N,fresh=len(fresh),reuse_low_motion=len(reused),stale=len(stale),true_no_geometry=len(missing),
            fresh_update_fraction=len(fresh)/N,reuse_fraction=len(reused)/N,stale_fraction=len(stale)/N,true_no_geometry_fraction=len(missing)/N,
            STATEFUL_GEOMETRY_AVAILABILITY=(len(fresh)+len(reused)+len(stale))/N,
            fresh_or_low_motion_fraction=(len(fresh)+len(reused))/N,
            FRESH_FRAME_BOOTSTRAP_AVAILABILITY=sum(x['fresh'] for x in modes['OLD']['rows'])/N,
            max_stale_m=max([x['distance_since_geometry_update'] for x in stale] or [0]),
            max_age_frames=max([x['geometry_age_frames'] for x in r if x['geometry_age_frames'] is not None] or [0]),
            max_stale_age_seconds=max([x['geometry_age_seconds'] for x in stale] or [0]),
            state_but_nonpositive_remaining_horizon=sum(x['valid_geometry_state'] and x['remaining_horizon_m'] is not None and x['remaining_horizon_m']<=0 for x in r))
        availability.append(av)
        for i in range(1,N):
            if not valid_edge(records[i-1],records[i]):continue
            dt=(int(records[i]['header_time_ns'])-int(records[i-1]['header_time_ns']))/1e9
            ds=float(np.linalg.norm(np.asarray(records[i]['lidar_pose_in_folder'])[:3,3]-np.asarray(records[i-1]['lidar_pose_in_folder'])[:3,3]))
            motion.append(dict(run=run,frame=i,speed_m_s=ds/dt,seconds=dt,fresh=r[i]['fresh'],solve_attempted=r[i]['solve_attempted'],
                state=r[i]['valid_geometry_state'],reuse=r[i]['low_motion_reuse']))
        previous=None
        for x in fresh:
            if previous is not None:
                dt=(int(records[x['frame']]['header_time_ns'])-int(records[previous['frame']]['header_time_ns']))/1e9
                spacing.append(dict(run=run,frame=x['frame'],meters=x['trigger_distance_m'],frames=x['frame']-previous['frame'],
                    seconds=dt,speed_m_s=(x['station']-previous['station'])/dt,updates_hz=1/dt))
            previous=x
        for mode,b in modes.items():
            rr=b['rows'];ts=[x['wall_seconds']*1000 for x in rr];cp=sum(x['cpu_seconds'] for x in rr)
            runrows.append(dict(run=run,mode=mode,frames=len(rr),duration_seconds=duration,wall_seconds=b['wall_seconds'],
                process_CPU_seconds=b['cpu_seconds'],core_CPU_seconds=cp,sum_core_wall_seconds=sum(ts)/1000,
                serialization_seconds=b['serialization_seconds'],geometry_solve_attempts=sum(x['solve_attempted'] for x in rr),
                fresh_updates=sum(x['fresh'] for x in rr),reuse_count=sum(x['low_motion_reuse'] for x in rr),
                effective_CPU_ms_raw_frame=cp/len(rr)*1000,effective_wall_ms_raw_frame=sum(ts)/len(rr),
                raw_frame_latency_stats_ms=stats(ts),fresh_solves_per_source_second=sum(x['fresh'] for x in rr)/duration,
                peak_RSS_bytes=max(x['peak_rss_bytes'] for x in rr),wall_note='3 independent processes concurrent; clean ablation separately'))
        framerows.extend(dict(run=run,**{k:v for k,v in x.items() if k not in ('run','timing','direction')},direction_mode=x.get('direction',{}).get('mode')) for x in r)
        p=OUT/'evaluation'/run/'SUMMARY.json'
        if p.exists():
            ev=lc.load(p);evaluations.extend(ev['rows']);errorrows.extend(ev['errors'])
    lc.csv_write(OUT/'scheduler_runs.csv',runrows);lc.csv_write(OUT/'scheduler_stats.csv',framerows)
    lc.csv_write(OUT/'stateful_availability.csv',availability);lc.csv_write(OUT/'update_spacing.csv',spacing)
    empirical=[]
    for lo,hi in [(0,1),(1,2),(2,3),(3,5),(5,10),(10,20),(20,50.000001)]:
        rr=[r for r in motion if lo<=r['speed_m_s']<hi]
        if not rr:continue
        seconds=sum(r['seconds'] for r in rr)
        empirical.append(dict(speed_lo_m_s=lo,speed_hi_m_s=min(hi,50),raw_frames=len(rr),source_seconds=seconds,
            fresh_updates=sum(r['fresh'] for r in rr),fresh_updates_per_second=sum(r['fresh'] for r in rr)/seconds,
            solve_attempts_per_second=sum(r['solve_attempted'] for r in rr)/seconds,reuse_frames=sum(r['reuse'] for r in rr),
            state_fraction=sum(r['state'] for r in rr)/len(rr)))
    lc.csv_write(OUT/'empirical_speed_load.csv',empirical)
    bins=[]
    for lo,hi in [(0,.1),(.1,.2),(.2,.3),(.3,.4),(.4,.5),(.5,float('inf'))]:
        rr=[x for x in errorrows if lo<=x['distance']<hi]
        bins.append(dict(lo=lo,hi=None if not np.isfinite(hi) else hi,frames=len(set((x['run'],x['frame']) for x in rr)),stations=len(rr),
            far_error_m=stats([x['far_error'] for x in rr]),near_error_m=stats([x['near_error'] for x in rr]),
            scope='8-100m current sensor range; near/far relative to contact rail; post-inference approximate future reference; >0.5 includes stale'))
    lc.csv_write(OUT/'error_vs_reuse_distance.csv',bins);lc.csv_write(OUT/'reuse_evaluation_frames.csv',evaluations)
    abl=[]
    for name in ('python','optimized','spatial','native','native_spatial'):
        rr=lc.load(OUT/'ablation'/f'{name}.json');lat=[r['T_TOTAL']*1000 for r in rr];c4=[r['T_C4_MARCHING']*1000 for r in rr]
        abl.append(dict(backend=name,frames=len(rr),median_ms=float(np.median(lat)),p95_ms=float(np.percentile(lat,95)),
            C4_median_ms=float(np.median(c4)),C4_p95_ms=float(np.percentile(c4,95)),
            bitwise_exact=name=='python' or lc.load(OUT/'ablation'/f'{name}_equivalence.json')['bitwise_exact']))
    lc.csv_write(OUT/'python_optimization_ablation.csv',abl)
    opt=lc.csv_read(OUT/'c4_optimizer_stats.csv');nt=lc.csv_read(OUT/'native_benchmark.csv')
    native_speed=np.median([float(r['median_us']) for r in nt if r['backend']=='python'])/np.median([float(r['median_us']) for r in nt if r['backend']=='native'])
    py=next(r for r in abl if r['backend']=='spatial');native=next(r for r in abl if r['backend']=='native_spatial')
    original=abl[0];baseline_rows=lc.load(OUT/'ablation/python.json')
    c4_share=sum(r['T_C4_MARCHING'] for r in baseline_rows)/sum(r['T_TOTAL'] for r in baseline_rows)
    native_reduction=1-native['median_ms']/py['median_ms']
    # Keep conservative production choice: native is not >=2x C4 and was only
    # validated on the ablation cohort, not all full-run update stations.
    decision='B. Scheduler + Python optimizations'
    native_report=f'''# Native feasibility\n\nC++17 / pybind11 3.0.1 prototype compiled and tested on Windows x64.
Residual distance kernel keeps point subtraction order, float64, sqrt and exact
nearest-template distance. Solver remains frozen SciPy; no Jacobian/loss/gates changed.
Kernel median speedup across captured fits: {native_speed:.3f}x.
Additional full-update median reduction versus spatial Python: {native_reduction*100:.2f}%.
Native C4 median: {native['C4_median_ms']:.2f} ms vs Python {py['C4_median_ms']:.2f} ms.
144 captured kernels and 40 fits were tested; full C4 decisions checked on 32 starts.
See native_equivalence.csv, native_fit_benchmark.csv and ablation/*_equivalence.json.
Production default remains Python: >=2x C4 was not achieved. No full solver/Ceres
port, no OpenMP or host-specific CPU flags. Ceres would change termination/order
and is outside the safe measured gain. Eigen is not needed for this two-coordinate kernel.
Native loading performs ABI/import and arithmetic self-check; explicit failure,
never silent fallback during a benchmark. Linux build status is separate.
'''
    (OUT/'native_feasibility.md').write_text(native_report,encoding='utf-8')
    theory=[]
    reuse_median=float(np.median([r['wall_seconds'] for r in framerows if r['low_motion_reuse']]))*1000
    for speed in (0,1,2,3,5,10):
        n=None if speed==0 else int(np.ceil(.5/(speed/10)-1e-12))
        hz=0. if n is None else 10/n
        theory.append(dict(speed_m_s=speed,frames_per_update=n,updates_hz=hz,
            geometry_wall_estimate_ms_raw_frame=py['median_ms']*hz/10,
            total_wall_estimate_ms_raw_frame=py['median_ms']*hz/10+reuse_median*(1-hz/10),
            note='constant straight speed, successful solves, 10Hz input; median wall proxy incl reuse, not measured CPU'))
    lc.csv_write(OUT/'speed_dependent_load.csv',theory)
    # Add frozen distance/fit counters from the existing full cProfile artifacts.
    profile_rows=[r for r in lc.csv_read(OUT/'c4_profile.csv') if r['operation'] not in ('distances','fit')]
    for operation in ('distances','fit'):
        hits=[v for path in (OUT/'profiles').glob('*.prof') for (file,line,name),v in pstats.Stats(str(path)).stats.items()
              if name==operation and file.endswith('tracker.py')]
        profile_rows.append(dict(operation=operation,calls=sum(v[1] for v in hits),sum_seconds=sum(v[3] for v in hits),
            self_seconds=sum(v[2] for v in hits),median_ms=None,p95_ms=None,overlapping='cProfile cumulative; per-call durations unavailable'))
    lc.csv_write(OUT/'c4_profile.csv',profile_rows)
    allocations=[]
    for path in (OUT/'profiles').glob('*.prof'):
        ps=pstats.Stats(str(path))
        for (file,line,name),(cc,nc,tt,ct,callers) in ps.stats.items():
            if any(k in name for k in ('concatenate','stack','array','copy','unique','reshape','approx_derivative')):
                allocations.append(dict(profile=path.stem,function=name,file=file,line=line,calls=nc,self_seconds=tt,
                    cumulative_seconds=ct,allocated_bytes=None,note='call counts; transient allocated byte volume not observable from cProfile'))
    lc.csv_write(OUT/'allocation_audit.csv',allocations)
    total=sum(r['raw_frames'] for r in availability);fresh=sum(r['fresh'] for r in availability);reuse=sum(r['reuse_low_motion'] for r in availability)
    stale=sum(r['stale'] for r in availability);missing=sum(r['true_no_geometry'] for r in availability)
    oldcpu=sum(r['core_CPU_seconds'] for r in runrows if r['mode']=='OLD');newcpu=sum(r['core_CPU_seconds'] for r in runrows if r['mode']=='NEW')
    md=['# FINAL PERFORMANCE OPTIMIZATION\n'];parts=[]
    def text(x):md.append(x+'\n');parts.append('<p>'+html.escape(x).replace('\n','<br>')+'</p>')
    def heading(x):md.append('\n## '+x+'\n');parts.append('<h2>'+html.escape(x)+'</h2>')
    def table(headers,rows):
        md.append('| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in rows)+'\n')
        parts.append('<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>')
    text(f'{decision}. Frozen geometry/seed/configs unchanged. Всего {total} последовательных кадров, пять полных runs.\nFresh: {fresh/total:.2%}; low-motion reuse: {reuse/total:.2%}; stale после неудачного update: {stale/total:.2%}; без геометрии: {missing/total:.2%}.\nCPU на raw frame: OLD {oldcpu/total*1000:.1f} мс → NEW {newcpu/total*1000:.1f} мс.\nPython spatial median / p95 solve: {py["median_ms"]:.1f} / {py["p95_ms"]:.1f} мс. Real-time 10 Hz не достигнут на всей выборке.')
    table(['Режим, одинаковые 32 старта','Solve median / p95, ms','C4 median / p95, ms'],[
        ['Исходные вычисления',f'{original["median_ms"]:.1f} / {original["p95_ms"]:.1f}',f'{original["C4_median_ms"]:.1f} / {original["C4_p95_ms"]:.1f}'],
        ['Python optimized + spatial cache',f'{py["median_ms"]:.1f} / {py["p95_ms"]:.1f}',f'{py["C4_median_ms"]:.1f} / {py["C4_p95_ms"]:.1f}'],
        ['Python spatial + C++ kernel',f'{native["median_ms"]:.1f} / {native["p95_ms"]:.1f}',f'{native["C4_median_ms"]:.1f} / {native["C4_p95_ms"]:.1f}']])
    text(f'Доля C4 в суммарном времени исходного 32-start benchmark: {c4_share:.1%}. State имеется на {(total-missing)/total:.2%} кадров, включая {stale} stale; fresh + low-motion reuse без stale: {(fresh+reuse)/total:.2%}. Цель C4 median <500 ms не достигнута.')
    env=lc.load(OUT/'environment.json')
    text(f'Стенд: {env["CPU"]}, {env["platform"]}, Python 3.12.14, NumPy {env["numpy"]}, SciPy {env["scipy"]}. BLAS/OMP/MKL threads = 1. Чистый benchmark выполнен после завершения всех full-run процессов: один последовательный проход на backend, одинаковые 32 starts, прогретый файловый cache. Это latency на данном CPU, не гарантия на целевом устройстве.')
    heading('1. Scheduler и direction contract')
    text('Порог 0.5 м — евклидово смещение от последнего успешного обновления. При меньшем движении сохраняется та же map-space geometry, преобразуются только coordinates/bases/covariances в текущий sensor frame. Старые s и source frame сохранены; нет нового prediction или экстраполяции. Полные raw frames читаются и передаются в obstacle interface также на reuse; obstacle detector здесь не реализован.')
    text('Orientation adapter документирован до экспериментов: прежний T→T+1 basis используется без изменений, если допустим. Иначе только при малом шаге/последнем T ищется ближайшая прошлая поза с накопленными ≥0.5 м, через непрерывные valid_edge. Полученный вектор проходит ту же basis(). Это явно новый источник направления на ранее отказавших кадрах; его нельзя называть битовым воспроизведением старого отказа. STEP1 detector и RAW STEP1→STEP6 seed adapter неизменны. Облако T+1 запрещено.')
    text('Разрыв pose/time сбрасывает state. Неудачный solve после порога оставляет UPDATE_FAILED_USING_STALE_GEOMETRY с возрастом/дистанцией и повторной попыткой на следующем кадре. Новый hard cutoff не введён. State availability включает явно отмеченный stale; это не сертификация пригодности obstacle corridor. Remaining horizon хранится отдельно, в том числе отрицательный после проезда конца прежней геометрии.')
    heading('2. Полная доступность и частота обновлений')
    table(['Run','Frames','Old fresh %','Fresh','Reuse','Stale','No geometry','State %','Max stale m'],[[r['run'],r['raw_frames'],f'{100*r["FRESH_FRAME_BOOTSTRAP_AVAILABILITY"]:.2f}',r['fresh'],r['reuse_low_motion'],r['stale'],r['true_no_geometry'],f'{100*r["STATEFUL_GEOMETRY_AVAILABILITY"]:.2f}',f'{r["max_stale_m"]:.2f}'] for r in availability])
    text(f'Кадров с сохранённым state, но исчерпанной прогнозной дальностью: {sum(r["state_but_nonpositive_remaining_horizon"] for r in availability)}. Максимальное время stale после отказа update: {max(r["max_stale_age_seconds"] for r in availability):.2f} с. Эти случаи нельзя трактовать как полное покрытие текущего obstacle corridor.')
    text('Исторические 61.11% на 1422 benchmark starts теперь называются FRESH_FRAME_BOOTSTRAP_AVAILABILITY. Новая STATEFUL_GEOMETRY_AVAILABILITY рассчитана на полных последовательностях. Два development runs выбраны заранее из 1000-frame записей с большой долей движения <0.5 м: 01a2 и 01d1.')
    text('Все 443 кадра NO_GEOMETRY_STATE относятся к началу new_data_part_01d1, кадры 0–442: ещё нет достаточного перемещения для первичного направления/bootstrap. Первый state — кадр 443, затем до конца записи пропусков состояния нет. Scheduler сохраняет существующую геометрию, но не создаёт её при отсутствии первого допустимого seed.')
    reasons=Counter(r['reason'] for r in framerows if r['stale'])
    table(['Причина неудачного refresh','Stale frames'],sorted(reasons.items()))
    worst_stale=sorted([r for r in framerows if r['stale']],key=lambda r:r['distance_since_geometry_update'],reverse=True)[:5]
    table(['Stale run / frame','Последний успех','Age s','Пройдено m','Осталось geometry m'],[
        [f'{r["run"]} / {r["frame"]}',r['source_geometry_frame'],f'{r["geometry_age_seconds"]:.3f}',
         f'{r["distance_since_geometry_update"]:.3f}',f'{r["remaining_horizon_m"]:.3f}'] for r in worst_stale])
    text('Самый длинный stale — switch frame 56 после успеха на 51: пять последовательных STEP2_NOT_FOUND, 0.50 с и 6.73 м без успешного refresh. У исходной геометрии горизонт только 8 м; остаётся около 1.27 м. Наличие state в этом случае не означает дальний obstacle corridor. Stale не прекращается искусственным новым порогом, но явно помечен для потребителя.')
    coverage=[]
    for run in RUNS:
        rr=[r for r in framerows if r['run']==run]
        valid=[r['remaining_horizon_m'] for r in rr if r['valid_geometry_state'] and r['remaining_horizon_m'] is not None]
        coverage.append(dict(run=run,median_remaining_horizon_m=float(np.median(valid)),minimum_remaining_horizon_m=min(valid),
            **{f'raw_fraction_remaining_ge_{h}m':sum(r['valid_geometry_state'] and r['remaining_horizon_m'] is not None and r['remaining_horizon_m']>=h for r in rr)/len(rr) for h in (30,50,75,100)}))
    lc.csv_write(OUT/'remaining_geometry_coverage.csv',coverage)
    table(['Run','Median remaining m','Minimum m','Raw frames with ≥30 m / ≥50 m / ≥75 m'],[
        [r['run'],f'{r["median_remaining_horizon_m"]:.2f}',f'{r["minimum_remaining_horizon_m"]:.2f}',
         ' / '.join(f'{r[f"raw_fraction_remaining_ge_{h}m"]:.1%}' for h in (30,50,75))] for r in coverage])
    table(['Run','Median m/update','p90 m/update','Median frames/update','Fresh updates/source sec'],[[run,
        f'{np.median([x["meters"] for x in spacing if x["run"]==run and x["meters"] is not None]):.3f}',
        f'{np.percentile([x["meters"] for x in spacing if x["run"]==run and x["meters"] is not None],90):.3f}',
        f'{np.median([x["frames"] for x in spacing if x["run"]==run]):.2f}',
        f'{next(x["fresh_solves_per_source_second"] for x in runrows if x["run"]==run and x["mode"]=="NEW"):.2f}'] for run in RUNS])
    heading('3. Реальная стоимость на raw frame')
    table(['Run','Mode','Wall/run s','CPU/run s','Solves','CPU ms/raw','Core wall ms/raw'],[[r['run'],r['mode'],f'{r["wall_seconds"]:.1f}',f'{r["process_CPU_seconds"]:.1f}',r['geometry_solve_attempts'],f'{r["effective_CPU_ms_raw_frame"]:.1f}',f'{r["effective_wall_ms_raw_frame"]:.1f}'] for r in runrows])
    text('Полные runs исполнялись максимум тремя независимыми процессами: wall/run содержит конкуренцию за ресурсы и сериализацию. Core CPU измерен в своём процессе; CSV отдельно сохраняет оба времени и serialization. Чистое сравнение latency сделано отдельным последовательным benchmark. У OLD adjacent-pose отказы дешёвые; NEW действительно решает многие из этих кадров, поэтому рост availability может увеличить CPU. Не выдаём стоимость отказа за стоимость полезного solve.')
    text(f'Пиковый RSS одного процесса NEW: {max(r["peak_RSS_bytes"] for r in runrows if r["mode"]=="NEW")/2**20:.1f} MiB; OLD: {max(r["peak_RSS_bytes"] for r in runrows if r["mode"]=="OLD")/2**20:.1f} MiB. Это измеренный peak процесса, включая сохранённые метаданные исследования; не оценка суммы одновременных процессов.')
    text('State availability здесь — последовательный offline replay по source timestamps, не доля выполненных дедлайнов 100 мс. CLI решает geometry синхронно и при быстром входном потоке накапливал бы очередь. Отдельный every-cloud consumer может получать последний полностью опубликованный state через geometry_for_pose без нового solve; асинхронная очередь и obstacle detector не реализованы. Возраст такого snapshot считается относительно текущего raw frame, включая задержку завершения solve.')
    for label,rr in [('low-motion reuse',[x for x in framerows if x['low_motion_reuse']]),('stale',[x for x in framerows if x['stale']])]:
        if rr:text(f'{label}: n={len(rr)}, median/p95 wall {np.median([x["wall_seconds"] for x in rr])*1000:.2f}/{np.percentile([x["wall_seconds"] for x in rr],95)*1000:.2f} мс. Время reuse не приравнивается к нулю.')
    table(['Speed m/s','Frames/update','Updates/s','Solve contribution ms/raw','Solve + reuse ms/raw'],[[r['speed_m_s'],r['frames_per_update'] or '∞',f'{r["updates_hz"]:.2f}',f'{r["geometry_wall_estimate_ms_raw_frame"]:.1f}',f'{r["total_wall_estimate_ms_raw_frame"]:.1f}'] for r in theory])
    text('Теоретическая таблица предполагает постоянную прямолинейную скорость и успешный solve. Начальный bootstrap/повторные неудачи в ней не моделируются. При 3 м/с и 10 Hz шаг 0.3 м: обновление через 2 кадра (0.6 м), а не непрерывные 6 updates/sec. При ≥5 м/с scheduler почти не сокращает частоту.')
    text('Solve + reuse — оценка по измеренным медианам wall time, не измерение CPU. При остановке после bootstrap остаётся около 20 мс на чтение raw frame, перенос геометрии и interface; нулевая частота solve не означает нулевую стоимость кадра.')
    table(['Measured speed m/s','Raw frames','Fresh updates/source sec','Attempts/source sec','Reuse frames'],[
        [f'{r["speed_lo_m_s"]}–{r["speed_hi_m_s"]}',r['raw_frames'],f'{r["fresh_updates_per_second"]:.2f}',f'{r["solve_attempts_per_second"]:.2f}',r['reuse_frames']] for r in empirical])
    text('Empirical speed bins используют только допустимые соседние pose/time интервалы и реальное source elapsed time. В отличие от теории они включают initial bootstrap и неудачные попытки; invalid intervals исключены, их скорость не превращена в ноль.')
    heading('4. Проверка reused geometry')
    table(['Distance since update','Frames with GT','Stations','Near p95 cm','Far p95 cm','Far max cm'],[[f'{r["lo"]}–{r["hi"]}',r['frames'],r['stations'],
        'N/A' if r['near_error_m']['p95'] is None else f'{r["near_error_m"]["p95"]*100:.2f}',
        'N/A' if r['far_error_m']['p95'] is None else f'{r["far_error_m"]["p95"]*100:.2f}',
        'N/A' if r['far_error_m']['p100'] is None else f'{r["far_error_m"]["p100"]*100:.2f}'] for r in bins])
    text(f'Оценка запускается после INFERENCE_COMPLETE каждого run. Reused prediction действительно перенесён в текущую позу и сравнивается на поперечных плоскостях с прежним независимым future-class1 reference. Оценённых reused/stale кадров: {sum(r["evaluated_stations"]>0 for r in evaluations)} из {len(evaluations)}. Отсутствующий GT остаётся N/A; разметка/регистрация приблизительны. Различия bins также отражают состав случаев и не являются причинной оценкой деградации от возраста.')
    worst=sorted([r for r in evaluations if r['far_max'] is not None],key=lambda r:r['far_max'],reverse=True)[:6]
    table(['Run / current frame','Source geometry','Age s / displacement m','GT stations','Max error cm'],[
        [f'{r["run"]} / {r["frame"]}',r['source_frame'],f'{r["age_seconds"]:.3f} / {r["distance"]:.3f}',r['evaluated_stations'],f'{100*r["far_max"]:.2f}'] for r in worst])
    text('Near/far обозначают ходовой рельс, ближайший/дальний относительно контактного, а не диапазон дальности. Оценка — поперечная lateral/vertical ошибка в метрах; точки за sensor исключены, текущая дальность 8–100 м. В bins секции объединены по доступному GT, поэтому многократно reused stationary state даёт коррелированные наблюдения.')
    text('Худший reused случай для дальнего от CR ходового рельса — одна и та же сохранённая геометрия switch frame 675, просмотренная из кадров 676–678: 17.48–17.59 см на единственной доступной GT-секции. Смещение за эти кадры 0.04–0.14 м, заметного монотонного роста ошибки нет. Это не доказательство точности полного горизонта: в этом месте GT-покрытие очень ограничено. Алгоритм и метрики под эти случаи не корректировались.')
    heading('5. C4 profile и optimizer audit')
    totalopt=sum(float(r['seconds']) for r in opt);residual=sum(float(r['residual_seconds']) for r in opt)
    text(f'Четыре профилированных обновления, {len(opt)} least_squares вызовов; median nfev {np.median([int(r["nfev"]) for r in opt]):.1f}, p95 {np.percentile([int(r["nfev"]) for r in opt],95):.1f}. Optimizer inclusive {totalopt:.3f} с; Python residual callback inclusive {residual:.3f} с ({residual/totalopt:.1%}). Остаток {totalopt-residual:.3f} с включает Python SciPy orchestration, finite differences и native BLAS, а не только native solver. Точное число iterations SciPy result не раскрывает; nfev/njev сохранены без подмены.')
    text('c4_profile.csv содержит history/ROI, KDTree build/query, candidates, template, state/beam helpers. Времена inclusive перекрываются. c4_optimizer_stats.csv — каждый fit и residual calls; c4_spatial_stats.csv — каждое дерево/query; allocation_audit.csv — наблюдаемые вызовы concatenate/stack/copy/unique/array и numdiff. Объём всех transient allocations cProfile не измеряет, поэтому bytes не выдуманы. line_profiler/py-spy/Scalene не устанавливались.')
    profiles=lc.load(OUT/'profiles/summary.json');spatial=lc.csv_read(OUT/'c4_spatial_stats.csv')
    table(['Profile run / frame','Fits','least_squares','Trees','query / radius','Mean query size'],[
        [f'{p["run"]} / {p["frame"]}',p['fits'],p['optimizers'],
         sum(x['operation']=='KDTree_build' for x in spatial if x['run']==p['run']),
         str(sum(x['operation']=='KDTree_query' for x in spatial if x['run']==p['run']))+' / '+str(sum(x['operation']=='KDTree_radius' for x in spatial if x['run']==p['run'])),
         f'{np.mean([float(x["size"]) for x in spatial if x["run"]==p["run"] and x["operation"]=="KDTree_query" and x["size"]]):.1f}'] for p in profiles])
    text(f'least_squares calls/update: median {np.median([p["optimizers"] for p in profiles]):.1f}, p95 {np.percentile([p["optimizers"] for p in profiles],95):.1f}. Эти четыре кадра профилировались с instrumentation, поэтому их время не подменяет чистые latency ablations.')
    table(['Operation','Calls','Inclusive seconds (overlap)'],[[r['operation'],r['calls'],f'{float(r["sum_seconds"]):.4f}'] for r in lc.csv_read(OUT/'c4_profile.csv')])
    history=lc.csv_read(OUT/'history_transform_audit.csv')
    lc.csv_write(OUT/'history_transform_summary.csv',[dict(run=run,reads=len(rr),source_frames=len(set(r['source_frame'] for r in rr)),
        transformed_points=sum(int(r['points']) for r in rr),source_metadata_cache_misses=sum(r['metadata_cached']=='False' for r in rr))
        for run in [p['run'] for p in profiles] if (rr:=[x for x in history if x['run']==run])])
    text('history_transform_audit.csv учитывает каждый source-frame read и объём transformed points. Внутри одного solve RoiSource.parts уже повторно использует current-coordinate массив каждого кадра; повторные marching ROI не трансформируют весь cloud заново. Между T изменяется sensor pose, поэтому current-coordinate XYZ пересчитываются; source-only metadata остаются в 12-frame cache. Новый spatial cache переиспользует map-space tree, а не устаревшие sensor XYZ.')
    text('Residual уже vectorized через SciPy cKDTree. В каждом callback создаётся p−anchor и массив расстояний; C++ удаляет эти промежуточные массивы внутри distance kernel, но возвращаемый residual остаётся отдельным массивом. Общий mutable buffer не введён: solver может удерживать старые residual values. Шаблон/tree уже immutable в tracker. Reverse tree по тем же candidate points вынесен из seed loop; повторные идентичные fits memoized. Query batching между marching шагами не введён: следующий центр зависит от выбранного предыдущего кандидата.')
    heading('6. Python / native ablation и equivalence')
    table(['Backend','Median ms','p95 ms','C4 median ms','Bitwise'],[[r['backend'],f'{r["median_ms"]:.1f}',f'{r["p95_ms"]:.1f}',f'{r["C4_median_ms"]:.1f}',r['bitwise_exact']] for r in abl])
    text('Python: неизменяемое дерево для reverse coverage строится один раз на fit; одинаковые fit inputs memoized внутри одного C4 вызова. Spatial cache хранит tree исходного map XYZ; консервативная broad-phase область затем уточняется тем же SciPy KDTree по тем же float64 текущим координатам и исходному радиусу. Broad-phase guard не меняет финальный ROI/acceptance. Trees ограничены разрешённой историей; старые transforms не переиспользуются между T.')
    text(f'C++ kernel ускоряет только point-to-template distance: {native_speed:.2f}× на captured inputs. Полный solver не переносился. Добавочный выигрыш к spatial Python: {native_reduction:.1%} median; C4 не ускорен вдвое. Production default — Python; native — явно включаемый проверочный prototype. Численные различия kernel/32-start outputs равны нулю в выполненных Windows проверках. Тест ThreadPool находится в threading_benchmark.csv: порядок результатов сохранён, убедительного ускорения нет, parallel candidate solver не включён.')
    orientation=lc.load(OUT/'orientation_equivalence.json')
    text(f'Отдельно проверены {orientation["cases"]} равномерно выбранных обновления с accumulated-pose direction: unoptimized frozen stages при том же адаптированном направлении и spatial implementation совпали побитово. Для прежних допустимых направлений полные runs также сверены с сохранёнными frozen results. Native tolerance 1e-10 задана заранее, discrete decisions tolerance 0; фактически измеренные расхождения равны нулю.')
    table(['ThreadPool workers','40 fits, seconds','Exact ordering/results'],[[r['workers'],r['seconds'],r.get('bitwise_identical','not recorded')] for r in lc.csv_read(OUT/'threading_benchmark.csv')])
    text('ThreadPool сохраняет порядок выдачи map(), но не ускоряет достаточно короткие независимые fits. ProcessPool в production не вводился: отдельная передача candidate arrays и lifecycle IPC не обоснованы полученным небольшим временем fit; его ускорение не измерено и не заявляется. Worker count / BLAS / OMP / MKL = 1 в production. Full-run независимые процессы использованы только для сокращения длительности аудита.')
    heading('7. Docker и воспроизводимость')
    build=(OUT/'docker_build_report.md').read_text(encoding='utf-8') if (OUT/'docker_build_report.md').exists() else 'Docker status pending'
    text(build)
    heading('8. Итог')
    text(decision+'. Scheduler устраняет ложную потерю уже полученного состояния при малом движении, но не обеспечивает 10 Hz geometry refresh на высоких скоростях. После успешного bootstrap состояние можно передавать в obstacle interface на каждом кадре непрерывного pose-сегмента; до bootstrap оно отсутствует. Актуальность и покрытие учитываются отдельно. Solver и математика детекторов не изменены.')
    (OUT/'REPORT_PERFORMANCE_FINAL.md').write_text('\n'.join(md),encoding='utf-8')
    page='<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Final performance</title><style>body{font:16px/1.6 system-ui;background:#eef2f5;color:#182f43;margin:24px}main{max-width:1320px;margin:auto;background:white;padding:28px}table{border-collapse:collapse;font-size:13px;width:100%}td,th{padding:8px;border:1px solid #d5dfe5;text-align:left}th{background:#e5eef5}.scroll{overflow:auto}h2{margin-top:36px}img{width:100%}</style><main><h1>FINAL PERFORMANCE OPTIMIZATION</h1>'+''.join(parts)+'</main></html>'
    (OUT/'REPORT_PERFORMANCE_FINAL.html').write_text(page,encoding='utf-8')
    (OUT/'FINAL_RUNTIME_RECOMMENDATION.md').write_text('# '+decision+'\n\n'+md[1]+'\nDefault C4_BACKEND=python. Native is opt-in experimental.\nSee REPORT_PERFORMANCE_FINAL.md and freeze.json.\n',encoding='utf-8')
    lc.save(OUT/'report_summary.json',dict(decision=decision,total_frames=total,fresh=fresh,reuse=reuse,stale=stale,no_geometry=missing,
        stateful_availability=(total-missing)/total,OLD_CPU_ms_raw=oldcpu/total*1000,NEW_CPU_ms_raw=newcpu/total*1000,
        ablation=abl,native_extra_reduction=native_reduction))


if __name__=='__main__':main()
