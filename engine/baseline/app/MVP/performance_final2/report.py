"""Measured results only. Tables and narrative share one Markdown/HTML source."""
from . import ROOT,OUT
import long_common as lc
import numpy as np
import csv,html,json
from collections import Counter

def read(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def pct(v,p=50):
    a=[float(x) for x in v if x is not None and x!='' and np.isfinite(float(x))]
    return float(np.percentile(a,p)) if a else None
def fmt(v):
    if v is None:return '—'
    if isinstance(v,(float,np.floating)):return f'{v:.3f}'
    return str(v)
class Document:
    def __init__(self,title):self.md=['# '+title+'\n'];self.html=['<h1>'+html.escape(title)+'</h1>']
    def h(self,text):self.md.append('\n## '+text+'\n');self.html.append('<h2>'+html.escape(text)+'</h2>')
    def p(self,text):self.md.append(text+'\n');self.html.append('<p>'+html.escape(text).replace('\n','<br>')+'</p>')
    def table(self,heads,rows):
        rows=[[fmt(x) for x in r] for r in rows];self.md.extend(['| '+' | '.join(heads)+' |','| '+' | '.join(['---']*len(heads))+' |']+['| '+' | '.join(r)+' |' for r in rows]+[''])
        self.html.append('<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in heads)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>')
    def image(self,path,caption):self.md.append(f'![{caption}]({path})\n');self.html.append(f'<figure><img src="{path}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    def save(self,name):
        (OUT/(name+'.md')).write_text('\n'.join(self.md),encoding='utf-8')
        (OUT/(name+'.html')).write_text('<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>C4 performance sprint</title><style>body{font:16px/1.6 system-ui;background:#eef3f6;color:#203347;margin:0}main{max-width:1260px;margin:auto;padding:32px;background:white}h1{line-height:1.2}h2{margin-top:40px}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:9px;border:1px solid #d8e1e7;text-align:left}th{background:#e9f1f5}.scroll{overflow:auto}img{width:100%;height:auto}figure{margin:24px 0}figcaption{color:#567}</style><main>'+''.join(self.html)+'</main></html>',encoding='utf-8')

def main():
    selection=lc.load(OUT/'selection.json');best=selection['backend'];paired={p.stem:lc.load(p) for p in (OUT/'paired').glob('*.json')}
    def metric(name,rr):return dict(backend=name,starts=len(rr),median_ms=pct([r['T_TOTAL']*1000 for r in rr]),p95_ms=pct([r['T_TOTAL']*1000 for r in rr],95),
        C4_median_ms=pct([r['T_C4_MARCHING']*1000 for r in rr]),C4_p95_ms=pct([r['T_C4_MARCHING']*1000 for r in rr],95))
    metrics={m:metric(m,r) for m,r in paired.items()};base=metrics['python'];winner=metrics[best];previous=metrics['native_spatial']
    screen_winner=winner;confirmation=None
    if (OUT/'confirmation.json').exists():
        confirmation=lc.load(OUT/'confirmation.json');cr=confirmation['rows']
        final_metrics={m:metric(m,[r for r in cr if r['backend']==m]) for m in ('python',best)}
        base=final_metrics['python'];winner=final_metrics[best]
    reduction=1-winner['median_ms']/base['median_ms'];speedup=base['median_ms']/winner['median_ms'];c4speed=base['C4_median_ms']/winner['C4_median_ms']
    waterfall=[]
    for m in ('python','native_spatial','fd','batch','soa','grid','grid_native'):
        waterfall.append(dict(**metrics[m],source='counterbalanced same-frame 32; independent variants, not additive',reduction_percent=100*(1-metrics[m]['median_ms']/metrics['python']['median_ms'])))
    lc.csv_write(OUT/'paired_variants.csv',waterfall)
    breakdown=[]
    keys=('T_POSES','T_STEP1','T_STEP2','T_C4_MARCHING','T_GEOMETRY_CONTRACT','T_SEED_ADAPTER','T_C4_SMOOTH','T_STEP6_RAIL_RECONSTRUCTION','T_LAS_IO','T_TRANSFORMS','T_PROVENANCE','T_TOTAL')
    for key in keys:
        breakdown.append(dict(stage=key,before_median_ms=pct([r[key]*1000 for r in paired['python']]),after_median_ms=pct([r[key]*1000 for r in paired[best]]),
            before_sum_seconds=sum(r[key] for r in paired['python']),after_sum_seconds=sum(r[key] for r in paired[best]),note='Boundary timers; median components are not additive'))
    lc.csv_write(OUT/'c4_before_after.csv',breakdown)
    abl={p.stem:lc.load(p) for p in (OUT/'ablations').glob('*.json') if isinstance(lc.load(p),list)}
    chronological=[];last=None
    for label,mode,folder in [('Baseline','python','baseline'),('Spatial','spatial','baseline'),('H1','h1','ablations'),('H2','h2','ablations'),('H3','h3','ablations'),('H4','residual','ablations'),('H5','fd','ablations'),('H8','batch','ablations'),('H9 16 threads','threads_16','ablations'),('Final paired','batch','paired')]:
        row=metric(mode,lc.load(OUT/folder/(mode+'.json')));row.update(optimization=label,protocol=folder,observed_delta_ms=None if last is None else row['median_ms']-last,causal_delta=False)
        chronological.append(row);last=row['median_ms']
    lc.csv_write(OUT/'waterfall.csv',chronological)
    threads=[]
    for m,n in [('batch',1),('threads_2',2),('threads_4',4),('threads_8',8),('threads_16',16)]:
        rr=abl[m];threads.append(dict(workers=n,**metric(m,rr),peak_RSS_bytes=max(r['peak_RSS_bytes'] for r in rr),source='chronological 32 starts; drift confounded, paired confirmation separate'))
    lc.csv_write(OUT/'thread_scaling.csv',threads)
    candidate=[]
    for m in ('fd','batch'):
        for r in paired[m]:candidate.append(dict(backend=m,run=r['run'],frame=r['frame'],total_ms=r['T_TOTAL']*1000,C4_ms=r['T_C4_MARCHING']*1000,**r['stats']))
    lc.csv_write(OUT/'candidate_batch_benchmark.csv',candidate)
    profiles=read('profile_functions.csv');counts=[]
    patterns={'optimizer':'least_squares','finite_difference':'approx_derivative','array_conversion':'asarray','unique':'unique','concatenate':'concatenate','stack':'stack','norm':'norm','SVD':'svd','tree_build':'tree'}
    for mode in ('python','native_spatial',best):
        for label,name in patterns.items():
            rr=[r for r in profiles if r['backend']==mode and (r['function']==name or (name=='asarray' and r['function']=='<built-in method numpy.asarray>'))]
            counts.append(dict(backend=mode,operation=label,calls=sum(int(r['calls']) for r in rr),self_seconds=sum(float(r['self_seconds']) for r in rr),
                cumulative_seconds=sum(float(r['cumulative_seconds']) for r in rr),scope='four instrumented starts; allocation-related API calls, not exact heap allocations'))
    for mode in ('fd','batch'):
        rr=paired[mode]
        for key in ('fit_calls','memo_hits','optimizer_calls','native_fun_calls','native_jac_calls','native_distance_calls','batch_calls','tree_builds','tree_hits'):
            counts.append(dict(backend=mode,operation=key,calls=sum(r['stats'].get(key,0) for r in rr),scope='32 unprofiled starts; explicit counters'))
    lc.csv_write(OUT/'call_counts.csv',counts)
    trees=read('tree_reuse.csv');tree_summary=[]
    for mode in ('python','native_spatial',best):
        rr=[r for r in trees if r['backend']==mode];unique=len(set((r['run'],r['frame'],r['content_hash']) for r in rr))
        tree_summary.append([mode,len(rr),unique,len(rr)-unique,sum(float(r['hash_seconds']) for r in rr)*1000,sum(float(r['build_seconds']) for r in rr)*1000])
    nr=read('nn_benchmark.csv');res=read('residual_benchmark.csv');jac=read('jacobian_benchmark.csv');fits=read('fit_backends.csv');layout=read('layout_gil.csv');allocation=read('allocation_workspace.csv')
    eq=lc.load(OUT/'FINAL_EQ_COMPLETE.json');quality=read('quality_sanity.csv');solver=lc.load(OUT/'solver_budget.json')
    docker=[]
    for variant in ('docker_portable','docker_host'):
        complete=lc.load(OUT/variant/'DOCKER_COMPLETE.json');rr=read(variant+'/docker_timings.csv')
        for mode in dict.fromkeys(x['backend'] for x in rr):
            values=[{k:float(v) if k.startswith(('T_','CPU_')) else v for k,v in x.items()} for x in rr if x['backend']==mode]
            docker.append(dict(platform=variant,**metric(mode,values),equivalence_cases=complete['cases'],exact=complete['bitwise'],host_optimized=complete['host_optimized']))
    lc.csv_write(OUT/'docker_benchmark.csv',docker)
    async_rows=[];frames=[];memory=[];examples=[]
    for path in sorted((OUT/'async').glob('*/*/result.json')):
        b=lc.load(path);rr=b['frames'];events=b['events'];solves=b['solves'];mode=rr[0]['backend'];run=rr[0]['run'];duration=b['source_duration']
        published=[e for e in events if e['event']=='FRESH_GEOMETRY_PUBLISHED' and e['clock']<=duration]
        avail=[r for r in rr if r['state_available']];positive=[r for r in avail if r['remaining_horizon']>0];weights=np.diff([r['source_seconds'] for r in rr]).tolist()+[0.]
        source_ids=[r.get('geometry_source_frame') for r in rr];newly_observed=sum(x is not None and (j==0 or x!=source_ids[j-1]) for j,x in enumerate(source_ids))
        expired_on_first_view=sum(r['state_available'] and (j==0 or source_ids[j]!=source_ids[j-1]) and r['remaining_horizon']<=0 for j,r in enumerate(rr))
        row=dict(backend=mode,run=run,raw_frames=len(rr),source_seconds=duration,fresh_solves=len(solves),published=len(published),
            superseded=sum(e['event']=='GEOMETRY_REQUEST_SUPERSEDED' for e in events),reuse_frames=len(avail)-newly_observed,
            max_pending=b['max_pending'],state_frames=len(avail),state_fraction=len(avail)/len(rr),positive_horizon_frames=len(positive),
            expired_state_frames=len(avail)-len(positive),positive_horizon_fraction=len(positive)/len(rr),expired_on_first_view=expired_on_first_view,
            valid_state_seconds=sum(w for r,w in zip(rr,weights) if r['state_available']),positive_horizon_seconds=sum(w for r,w in zip(rr,weights) if r['state_available'] and r['remaining_horizon']>0),
            geometry_age_median_s=pct([r['geometry_age_seconds'] for r in avail]),geometry_age_p95_s=pct([r['geometry_age_seconds'] for r in avail],95),
            geometry_age_p90_s=pct([r['geometry_age_seconds'] for r in avail],90),distance_age_p95_m=pct([r['distance_since_geometry_source'] for r in avail],95),
            raw_frame_Hz=(len(rr)-1)/duration,solve_Hz=len(solves)/duration,time_to_horizon_p05_s=pct([r.get('time_to_horizon') for r in avail],5),
            min_remaining_horizon_m=min([r['remaining_horizon'] for r in avail] or [float('nan')]),
            solve_median_ms=pct([r['wall_seconds']*1000 for r in solves]),solve_p95_ms=pct([r['wall_seconds']*1000 for r in solves],95),
            raw_service_median_ms=pct([r['service_seconds']*1000 for r in rr]),raw_service_p95_ms=pct([r['service_seconds']*1000 for r in rr],95),
            raw_over_100ms=sum(r['service_seconds']>.1 for r in rr),publish_Hz=len(published)/duration,
            first_available_frame=next((r['frame'] for r in rr if r['state_available']),None),peak_RSS_bytes=b['peak_RSS_bytes'])
        async_rows.append(row);frames.extend(rr);memory.append(dict(scope='async sequential simulation process peak high-water',backend=mode,run=run,bytes=b['peak_RSS_bytes']))
        for kind,ordered in [('maximum_age',sorted(avail,key=lambda r:-r['geometry_age_seconds'])),('minimum_horizon',sorted(avail,key=lambda r:r['remaining_horizon']))]:
            if ordered:examples.append(dict(kind=kind,**{k:v for k,v in ordered[0].items() if k not in ('worker_busy','pending_depth')}))
    lc.csv_write(OUT/'async_runs.csv',async_rows)
    lc.csv_write(OUT/'geometry_age.csv',[{k:r.get(k) for k in ('backend','run','frame','state_available','geometry_source_frame','geometry_age_seconds','distance_since_geometry_source','speed_m_s')} for r in frames])
    lc.csv_write(OUT/'horizon_age.csv',[{k:r.get(k) for k in ('backend','run','frame','geometry_source_frame','remaining_horizon','time_to_horizon','speed_m_s')} for r in frames])
    lc.csv_write(OUT/'async_worst_cases.csv',examples)
    bins=[]
    for mode in dict.fromkeys(r['backend'] for r in frames):
        available=[r for r in frames if r['backend']==mode and r['state_available']]
        for lo,hi in [(-float('inf'),5),(5,10),(10,20),(20,30),(30,50),(50,float('inf'))]:
            rr=[r for r in available if lo<=r['remaining_horizon']<hi]
            bins.append(dict(backend=mode,lo_m=None if not np.isfinite(lo) else lo,hi_m=None if not np.isfinite(hi) else hi,frames=len(rr),fraction_available=len(rr)/len(available) if available else None))
    lc.csv_write(OUT/'horizon_bins.csv',bins)
    for row in threads:memory.append(dict(scope='single backend 32 starts process peak RSS',backend=row['backend'],bytes=row['peak_RSS_bytes']))
    for row in read('offline_throughput.csv'):memory.append(dict(scope='independent process upper estimate, not simultaneous measured RSS',backend=best,processes=row['processes'],bytes=row['concurrent_RSS_upper_estimate']))
    lc.csv_write(OUT/'memory.csv',memory)
    hypotheses=[
      ('H1','Принято','Small-N exact brute force: forward N<256, reverse N<1024; thresholds выбраны по captured timings, не по GT. Макроэффект смотрите вместе с H2/H3.','A: kernel work'),
      ('H2','Принято частично','Повторное дерево points вынесено из цикла seed; content cache ограничен 128. Источники: tree_reuse.csv. Identity-cache требовал удержания owner; первоначальный дефект исправлен и 32 старта повторены.','B: rebuild overhead'),
      ('H3','Принято','Reverse coverage сохраняет shifted-template-to-points семантику; маленькие наборы используют exact native NN.','A: kernel work'),
      ('H4','Принято','Полный residual distance/0.015 в C++; Cauchy остаётся в frozen SciPy. Никаких новых reverse residual terms.','C: Python callback overhead'),
      ('H5','Принято','Native 2-point finite differences воспроизводят rel_step, zero fallback, bounds, фактический dx и F layout. 108 residual/Jac cases и 150 fitted seeds совпали.','C: finite-difference orchestration'),
      ('H6','Отклонено','Analytic Jacobian меняет результат captured fits; NN assignment piecewise smooth и шаги solver отличаются. В final не включён.','D: equivalence failure'),
      ('H7','Spike отклонён','2 параметра. SciPy+native fun/jac принят. Полностью native projected damped Gauss–Newton не эквивалентен TRF: 0/150. Ceres не добавлялся.','D: solver semantics'),
      ('H8','Частично реализовано','Все независимые fit-запросы beam parents собраны по шагу, возвращены в исходные slots. Один outer pybind batch вызов; внутри сохраняются Python callbacks в SciPy и scoring. Полного устранения boundary crossings нет.','E: remaining Python/SciPy loop'),
      ('H9','Не выбран','2/4/8/16 потоков проверены с BLAS=1 и exact output. Есть GIL и очень мелкие native участки; хронологические замеры дрейфуют, сильного масштабирования не подтверждено.','F: GIL and scheduling'),
      ('H10','Реализовано','std::thread pool живёт на протяжении C4 tracker, не создаётся на каждом march step; backend default использует 1 поток.','F: pool lifetime'),
      ('H11','Не выбран','Native grid возвращает superset, исходный cKDTree точно фильтрует радиус. Sets и full pipeline совпали. Paired full timing хуже selected backend; broad-phase extra points/refinement/build cost.','G: broad-phase cost'),
      ('H12','Не выбран','AoS/SoA native сравнение и полный paired run. SoA не дал устойчивого выигрыша total; строгая arithmetic order ограничивает vector reduction.','H: layout and SIMD'),
      ('H13','Не выбран отдельно','Безопасный reusable scratch для Jacobian прошёл exact check; выигрыш мал относительно полного solver. API allocation counts из cProfile — не heap allocation counts; public arrays всегда владеют памятью.','I: allocation lower-order cost'),
      ('H14','Ограничено','Memo identical fits сохранён; реальные hits/calls приведены отдельно. Template tree и transformed source frame cache уже переиспользуются. Бессмысленный content hash не добавлен к большим source clouds на hot path.','B: cache economics'),
      ('H15','Без semantic changes','Bounds и immutable template hoisted. loss=cauchy, f_scale=1, diff_step=.001, max_nfev=24 и tolerances SciPy не изменены; уменьшать iterations/beam/history запрещено.','J: frozen settings'),
      ('H16','Принято','Native arithmetic освобождает GIL; Python/SciPy часть снова требует GIL. gil_scaling.csv отдельно от полного pipeline.','F: limited parallel fraction'),
      ('H17','Только offline','1/2/4/8 процессов на восьми разных runs: отдельная throughput метрика, не latency одного LiDAR stream.','F: independent-run parallelism')]
    hypothesis_rows=[]
    maps={'H1':'h1','H2':'h2','H3':'h3','H4':'residual','H5':'fd','H8':'batch','H9':'threads_16','H10':'threads_16','H11':'grid','H12':'soa','H14':'h2','H15':'fd','H16':'batch'}
    for h,s,t,c in hypotheses:
        mode=maps.get(h);measured=metrics.get(mode) or (metric(mode,abl[mode]) if mode in abl else None)
        micro_speed={'H4':pct([r['python_us'] for r in res])/pct([r['native_us'] for r in res]),'H5':pct([r['python_us'] for r in jac])/pct([r['native_us'] for r in jac]),
            'H12':pct([r['aos_us'] for r in layout])/pct([r['soa_us'] for r in layout]),'H13':pct([r['temporary_us'] for r in allocation])/pct([r['reused_us'] for r in allocation])}.get(h)
        hypothesis_rows.append(dict(hypothesis=h,status=s,explanation=t,cause=c,backend=mode,micro_speedup_ratio_of_medians=micro_speed,
            full_speedup=None if measured is None else metrics['python']['median_ms']/measured['median_ms'],C4_speedup=None if measured is None else metrics['python']['C4_median_ms']/measured['C4_median_ms'],
            attribution='Combined backend, not isolated causal contribution; micro omitted when not a comparable kernel'))
    lc.csv_write(OUT/'hypothesis_results.csv',hypothesis_rows)
    auditfits=read('solver_audit.csv');total=sum(r['full_seconds'] for r in solver);c4sum=sum(r['C4_seconds'] for r in solver)
    amdahl=[]
    for name,hs,seconds in [('native residual','H1/H4/H12/H16',sum(float(r['fun_seconds']) for r in auditfits)),
        ('native FD Jacobian','H5/H6/H13',sum(float(r['jac_seconds']) for r in auditfits)),
        ('solver outside callbacks','H7/H8/H9/H10/H15',sum(float(r['orchestration_seconds']) for r in auditfits)),
        ('all C4 conservative ceiling','H2/H3/H11/H14',c4sum)]:
        fraction=seconds/total;amdahl.append(dict(operation=name,hypotheses=hs,seconds=seconds,full_fraction=fraction,infinite_speedup_upper=1/(1-fraction),
            scope='Remaining best backend; subsets overlap. Whole C4 is a deliberately loose ceiling for unseparated ROI/cache paths'))
    lc.csv_write(OUT/'amdahl_remaining.csv',amdahl)
    doc=Document('FINAL C4 PERFORMANCE SPRINT — измерения и решение')
    doc.p(f'Лучший измеренный equivalent backend: native_batch_spatial (внутреннее имя {best}), 1 CPU worker, BLAS/OMP/MKL=1. Полный solve median/p95: {winner["median_ms"]:.1f}/{winner["p95_ms"]:.1f} ms; C4: {winner["C4_median_ms"]:.1f}/{winner["C4_p95_ms"]:.1f} ms. Относительно original Python в том же замере: {speedup:.2f}× full и {c4speed:.2f}× C4, снижение full latency {reduction*100:.1f}%. Вердикт D: C4 остаётся >500 ms. Ускорения в несколько раз и 10 Hz геометрии не достигнуто.')
    if confirmation:
        doc.p('Headline — заключительный изолированный замер FINAL binary на 32 прежних starts: только original и выбранный backend, порядок чередуется, повторного выбора backend нет. Он сделан после исправления cache в отдельном вызове Jacobian; Windows и Linux gates повторены. Screening таблицы ниже сохраняют прежние значения всех вариантов и не смешиваются с этой контрольной таблицей.')
        doc.table(['Final build','Full median ms','Full p95 ms','C4 median ms','C4 p95 ms'],[[m]+[final_metrics[m][k] for k in ('median_ms','p95_ms','C4_median_ms','C4_p95_ms')] for m in ('python',best)])
    doc.h('Что проверено и что заморожено')
    doc.p('Новые код и результаты находятся только в MVP/performance_final2 и results_performance_final2. STEP1/STEP2/C4/C4_SMOOTH/STEP6, прошлый performance sprint и source LAS не менялись. Frozen RAW seed adapter и разрешённый causal pose fallback сохранены. Pose T+1 допустима; cloud T+1 запрещено. Inference не использует class1/class2. Допуск float64 1e-10 задан до замеров, discrete tolerance=0; фактически наблюдаемые дельты равны 0.')
    doc.p(f'Проверка: Windows {eq["cases"]} заранее зафиксированных стартов, Linux portable 200 стартов, host-optimized Linux 32. Сравниваются STEP1, STEP2, C4, seed, smooth curve, итоговая геометрия и provenance; временные поля исключены. Это проверка эквивалентности frozen решений, а не подтверждение точности неоднозначной GT. Полные candidate fits дополнительно проверены на captured fixtures.')
    doc.p('Из 200 стартов 182 выдали итоговую геометрию, 18 — одинаковое отсутствие результата; эти отказы включены в проверку, а не удалены из выборки. Запуски на размеченном пуле ANNOTATED используют его как RAW: classification не является inference feature. Между Windows и Linux сравнивается original→optimized внутри каждой платформы; cross-OS побитовое равенство разных BLAS сборок не заявляется.')
    doc.h('Протокол производительности')
    doc.p('AMD Ryzen 9 7950X3D, 16 cores/32 logical; Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1, pybind11 3.0.1. Windows MSVC /O2 /fp:strict; Linux GCC -O3 -fno-fast-math -ffp-contract=off. Основная выборка — прежние 32 старта, 4 recordings × 8 последовательных frames. Финальные 200: по 40 равномерных ordinals на пяти полных recordings, зафиксированы до новых измерений.')
    doc.p('Baseline воспроизведён с hashes прошлого этапа. Первые последовательные ablations заметно дрейфовали, поэтому итоговая таблица получена чередованием backend на каждом одинаковом frame с вращением порядка. Эти 32 используются и для выбора победителя, поэтому малый перевес batch над fd нельзя считать независимым доказательством. 200-start correctness timings с конкурирующей нагрузкой не используются для speedup. Peak RSS в paired процессе включает несколько backend caches; память одиночного backend измерена отдельно.')
    doc.table(['Backend','Full median ms','Full p95 ms','C4 median ms','C4 p95 ms'],[[m,*[metrics[m][k] for k in ('median_ms','p95_ms','C4_median_ms','C4_p95_ms')]] for m in ('python','native_spatial','fd','batch','soa','grid','grid_native')])
    doc.image('figures/waterfall.png','Waterfall: последовательность измеренных equivalent вариантов; эффекты не складываются как независимые ускорения')
    doc.h('Разбор полного solve')
    doc.table(['Stage','Before median ms','After median ms','Before sum s','After sum s'],[[r[k] for k in ('stage','before_median_ms','after_median_ms','before_sum_seconds','after_sum_seconds')] for r in breakdown])
    doc.p('T_C4_MARCHING вычитает учтённые history I/O и transformations из inclusive C4 interval. Отдельные median не обязаны суммироваться в median TOTAL. ROI extraction, post-fit scoring, beam orchestration и неинструментированные мелкие операции остаются внутри C4. cProfile таблицы отдельно: cumulative times перекрываются и не суммируются.')
    doc.h('H1–H17: принятые и отклонённые гипотезы')
    doc.table(['ID','Решение','Причина'],[[h,s,t+' '+c] for h,s,t,c in hypotheses])
    doc.h('NN/KDTree audit')
    doc.table(['Backend','Builds','Unique frame/content','Repeated content','Hash ms','Build ms'],tree_summary)
    doc.p('tree_reuse.csv хранит frame, call ID, pointer, object identity, shape, SHA256 contents, доступный step/fit context и build/hash time. Audit hashing существует только в instrumented запуске. Одинаковые object pointers не доказывают одинаковые данные; allocator может переиспользовать адрес. В H1 это вызвало дефект lifetime: tree мог копировать non-contiguous input. Исправление удерживает original owner рядом с tree; исходный провал сохранён в diagnostics, исправленный H1 проверен повторно.')
    doc.table(['Direction','Cases','Native median µs','KD query median µs','KD build+query median µs'],[[d,len(rr),pct([float(r['native_ns'])/1000 for r in rr]),pct([float(r['kdtree_query_ns'])/1000 for r in rr]),pct([float(r['kdtree_build_query_ns'])/1000 for r in rr])] for d in ('forward','reverse') if (rr:=[r for r in nr if r['direction']==d])])
    doc.p('Distance-only NN paths сохраняют tie-distance; там, где используется индекс nearest template point, оставлен исходный cKDTree. Reverse query — template[::2]+anchor к points, не наоборот. Forward template tree уже был построен один раз. Большие деревья source frames переиспользуются предыдущим spatial cache; grid не выбран по end-to-end результату. spatial_benchmark.csv содержит также exact brute broad-phase на реальных clouds; native KDTree не реализовывался.')
    doc.h('Residual, Jacobian и solver')
    doc.table(['Operation','Python median µs','Native median µs','Fixtures'],[['Residual',pct([r['python_us'] for r in res]),pct([r['native_us'] for r in res]),len(res)],['Finite-difference Jacobian',pct([r['python_us'] for r in jac]),pct([r['native_us'] for r in jac]),len(jac)]])
    doc.table(['Backend','Fit count','Median ms','p95 ms','Max parameter Δ','Passed'],[[mode,len(rr),pct([r['ms'] for r in rr]),pct([r['ms'] for r in rr],95),max(float(r['max_parameter_delta']) for r in rr),sum(r['passed']=='True' for r in rr)] for mode in ('original','residual','fd','analytic') if (rr:=[r for r in fits if r['backend']==mode])])
    doc.table(['Run','C4 s','Solver without native fun/jac s','Fraction'],[[r['run'],r['C4_seconds'],r['solver_overhead_seconds'],r['fraction']] for r in solver])
    doc.p('Solver budget — отдельные wall wrappers: из времени least_squares вычтены native fun/jac calls. Остаток включает SciPy validation, bounds, loss, TRF orchestration, linear algebra и wrapper overhead; это не чистое время Python. Первичный inclusive cProfile показал повод проверить native fit spike; уточнённый budget оказался ниже 30% C4 на всех четырёх starts (около 4–27%). Поэтому условие для полного эквивалентного порта solver НЕ подтверждено. Маленький Experimental projected damped Gauss–Newton spike решает тот же residual/loss/bounds, но НЕ воспроизводит SciPy TRF step/termination. Он отвергнут на captured-fit gate (0/150), full C4 с ним не запускался. Полный эквивалентный порт SciPy TRF не выполнен и не заявлен.')
    doc.h('C++ analysis: граница реального выигрыша')
    doc.p('Исходные cKDTree.query, SVD/BLAS и NumPy arithmetic уже выполнялись native. Выигрыш дали small-N поиск без лишних KDTree calls и перенос подготовки residual/FD в C++. Последовательные march steps и выбор beam сохраняются. Native batch API пока вызывает Python fit/scoring и frozen SciPy; это частичный H8, а не один Python↔C++ переход на весь step. Счётчики ниже включают nested calls; нельзя выдавать число outer batch calls за общее число границ.')
    rr=paired[best];sums={k:sum(r['stats'].get(k,0) for r in rr) for k in ('optimizer_calls','native_fun_calls','native_jac_calls','native_distance_calls','batch_calls','fit_calls','memo_hits')}
    doc.table(['Counter, 32 starts','Count'],list(sums.items()))
    doc.p(f'Нижняя оценка вызовов Python→native arithmetic/batch: {sums["native_fun_calls"]+sums["native_jac_calls"]+sums["native_distance_calls"]+sums["batch_calls"]}; дополнительно native→Python callback на каждую задачу fit и index-sensitive SciPy NN calls. Memo hit rate: {sums["memo_hits"]}/{sums["fit_calls"]}. Полного allocator trace нет: call_counts.csv отражает API вызовы; allocation_workspace.csv — точное устранение одного temporary vector/Jac, публичные массивы не alias.')
    doc.p('SoA layout и reusable workspace прошли exact checks, но не обосновали отдельное включение по total latency. Compiler reports сохранены в native_build.log и docker_build_*.log. Отсутствие fast-math сохраняет порядок операций; заметки vectorizer не равны измерению memory bandwidth. Hardware counters bandwidth/cache misses не снимались, поэтому memory-bound причина не утверждается как доказанный факт.')
    doc.h('Потоки, GIL и независимые runs')
    doc.table(['Workers','Full median ms','C4 median ms','RSS MiB'],[[r['workers'],r['median_ms'],r['C4_median_ms'],r['peak_RSS_bytes']/2**20] for r in threads])
    doc.p('Это исходный chronological sweep: выигрыши относительно медленного первого batch-run нельзя целиком приписывать потокам. Контрольный paired 1-thread batch быстрее его первого замера. Сохраняются Python callbacks и короткие native kernels, pool overhead/GIL уменьшают пользу. Предопределённые output slots сохраняют порядок и ties. По этой причине default — 1 worker, без oversubscription. Candidate fit duration распределение и nfev/njev доступны в solver_audit.csv; iterations SciPy напрямую не экспонирует, их не подменяли nfev.')
    tp=read('thread_paired.csv');doc.table(['Counterbalanced workers','8 starts full median ms','C4 median ms'],[[n,pct([r['full_ms'] for r in tp if int(r['workers'])==n]),pct([r['C4_ms'] for r in tp if int(r['workers'])==n])] for n in (1,2,4,8,16)])
    offline=read('offline_throughput.csv');doc.table(['Processes','Solves/s','Wall s','Worker peak MiB','Exact'],[[r['processes'],float(r['solves_per_second']),float(r['wall_seconds']),float(r['peak_worker_RSS_bytes'])/2**20,r['exact']] for r in offline])
    doc.p('H17 использует 8 разных physical runs, middle four frames каждого. В wall включён запуск процессов. Это ускорение обработки независимых записей offline; оно не ускоряет причинную цепочку одного live stream. RAM upper estimate = workers × max worker peak, не измеренный одновременный суммарный RSS.')
    doc.h('Эквивалентность и downstream sanity')
    doc.table(['Platform','Starts','Exact'],[['Windows',eq['cases'],eq['bitwise']],['Linux portable',200,True],['Linux -march=native diagnostic',32,True]])
    doc.table(['Range m','Stations','Starts','Original p95 m','Best p95 m','Max Δ'],[[r['lo']+'–'+r['hi'],r['stations'],r['starts'],r['original_p95'],r['best_p95'],r['max_difference']] for r in quality])
    doc.p('GT читается только после FINAL_EQ_COMPLETE. Это прежний approximate reference, а не новый эталон. Downstream ошибки равны, потому что предсказания равны. Эта таблица не оценивает дополнительную ошибку при async reuse более старой геометрии на текущем frame; async age/horizon ниже — эксплуатационная диагностика, а не доказательство точности препятствий.')
    doc.h('Linux Docker и переносимость')
    doc.table(['Image','Backend','Full median ms','p95 ms','C4 median ms'],[[r[k] for k in ('platform','backend','median_ms','p95_ms','C4_median_ms')] for r in docker])
    doc.p('Portable runtime собран multi-stage, внутри нет gcc/g++/cmake/make. Original datasets и captures подключены read-only, runtime network=none. Проверены residual/Jacobian, bounds/zero, owning arrays, native pool order, grid exact refinement, latest-pending/gap semantics, затем полный pipeline. Host-optimized -march=native — диагностический image для этого CPU, не переносимый default. Linux Docker использует WSL2 и Windows bind mounts: разницу I/O/виртуализации нельзя выдавать за чистую разницу компилятора.')
    doc.h('Async latest-frame architecture')
    doc.p('GeometryWorker: максимум один активный solve и один pending_latest_frame. Новый допустимый request заменяет pending, фиксируется GEOMETRY_REQUEST_SUPERSEDED. Spatial trigger 0.5 m относительно последнего успешного source pose; после publish pending eligibility перепроверяется. Только полностью рассчитанная геометрия публикуется атомарно в map coordinates. Consumer читает каждый текущий raw frame и переводит published geometry в текущую sensor frame. Gap сбрасывает state и generation, поздний pre-gap solve не публикуется. При failure прежняя геометрия остаётся с явным event. Нового cutoff по age нет.')
    doc.table(['Backend/run','Frames','Solves','Publish Hz','Age p50/p95 s','Min horizon m','Positive horizon s','Raw p95 ms'],[[r['backend']+'/'+r['run'],r['raw_frames'],r['fresh_solves'],r['publish_Hz'],f'{r["geometry_age_median_s"]:.2f}/{r["geometry_age_p95_s"]:.2f}' if r['geometry_age_median_s'] is not None else '—',r['min_remaining_horizon_m'],r['positive_horizon_seconds'],r['raw_service_p95_ms']] for r in async_rows])
    doc.image('figures/async_age.png','Возраст опубликованной геометрии на полных runs; source timestamps и реально измеренная solve wall latency')
    doc.p('Основные full-run результаты — discrete-event simulation на реальных source timestamps, каждый selected solve действительно выполнен и его wall time задаёт момент completion. Consumer read/transform измерен отдельно, с независимым ресурсом в модели; это не end-to-end нагрузочный стенд препятствий. Дополнительно выполнен live paced replay с настоящим background thread: async_live.json фиксирует фактические arrival lateness и service latency. Поздние completions после конца записи не повышают on-stream availability; pending в конце не drained.')
    live=lc.load(OUT/'async_live.json');lr=live['frames'];doc.table(['Live frames','Max pending','Raw p50 ms','Raw p95 ms','Arrival lateness p95 ms','State frames'],[[len(lr),live['max_pending'],pct([r['service_seconds']*1000 for r in lr]),pct([r['service_seconds']*1000 for r in lr],95),pct([r['arrival_lateness_seconds']*1000 for r in lr],95),sum(r['state_available'] for r in lr)]])
    doc.p('Remaining horizon измеряется после преобразования в текущий frame и может стать отрицательным: наличие state object не означает актуальную полезную геометрию. time-to-horizon использует оценку скорости только по валидному последнему pose edge; при нулевой/неизвестной скорости остаётся missing. Pose/speed остаются оценками. Начальную задержку до первой успешной публикации и случаи исчерпания горизонта отчёт не скрывает.')
    doc.table(['Backend/run','State %','Positive horizon % of ALL raw frames','Expired state frames','New states already expired'],[[r['backend']+'/'+r['run'],r['state_fraction']*100,r['positive_horizon_fraction']*100,r['expired_state_frames'],r['expired_on_first_view']] for r in async_rows])
    diag=read('async_failure_diagnosis.csv');doc.table(['Case','Source/current frame','Source horizon m','Age s','Travel m','Remaining m','Frozen stop reason'],[[r['run'],r['geometry_source_frame']+'/'+r['current_frame'],r['source_geometry_horizon_m'],r['source_age_s'],r['distance_since_source_m'],r['current_remaining_horizon_m'],r['original_reason']] for r in diag])
    doc.p('Причина худших отрицательных horizons воспроизведена повторным inference: frozen C4 завершился с NO_POINTS, итоговый доступный участок остался 8 m. За 1.9–2.3 s поезд переместился на 23–34 m при оценке скорости 11–15 m/s. Ограниченная очередь устраняет backlog, но не восстанавливает отсутствующий дальний участок и не делает 8 m достаточными. На платформе большой возраст 38.1 s сопровождался только 2.17 m перемещения и ещё 27.19 m горизонта: возраст сам по себе не достаточный критерий. Новый порог rejection/priority не вводился.')
    doc.h('Лучшие, средние и худшие измеренные случаи')
    ordered=sorted(paired[best],key=lambda r:r['T_TOTAL']);doc.table(['Kind','Run/frame','Full ms','C4 ms'],[[name,r['run']+'/'+str(r['frame']),r['T_TOTAL']*1000,r['T_C4_MARCHING']*1000] for name,r in [('fastest',ordered[0]),('median',ordered[len(ordered)//2]),('slowest',ordered[-1])]])
    doc.table(['Kind','Backend/run/frame','Age s','Remaining horizon m','Time to horizon s'],[[r['kind'],r['backend']+'/'+r['run']+'/'+str(r['frame']),r.get('geometry_age_seconds'),r.get('remaining_horizon'),r.get('time_to_horizon')] for r in examples])
    doc.h('Amdahl limit и решение о продолжении')
    doc.table(['Remaining component','Hypotheses','Share of full','Infinite-speedup upper bound'],[[r['operation'],r['hypotheses'],r['full_fraction'],r['infinite_speedup_upper']] for r in amdahl])
    fraction=sum(r['T_C4_MARCHING'] for r in paired['python'])/sum(r['T_TOTAL'] for r in paired['python']);outside=pct([(r['T_TOTAL']-r['T_C4_MARCHING'])*1000 for r in paired['python']])
    doc.p(f'Доля C4 в сумме baseline времени = {fraction*100:.1f}%. Даже бесконечно быстрый C4 даёт верхнюю Amdahl-границу {1/(1-fraction):.2f}× для этого full solve; median вне C4 = {outside:.1f} ms. Это математический потолок, не обещание достижимого ускорения. Реальный остаток содержит ROI/trees, sequential marching и generic SciPy orchestration. Независимый native solver spike не эквивалентен; H8 пока не устранил внутренние границы. Продолжать полный переписанный solver перед сдачей не рекомендуется: выигрыш не доказан, regression risk высокий.')
    enabled=reduction>=.2 and eq['bitwise'] and all(r['exact'] for r in docker)
    doc.p(f'Production backend recommendation: native_batch_spatial, 1 worker, только с зафиксированным build/runtime; default gate={enabled}: full Docker equivalence, 0 discrete differences, >=20% total latency reduction относительно original Python. В screening дополнительное снижение относительно прежнего native_spatial = {(1-screen_winner["median_ms"]/previous["median_ms"])*100:.1f}%, поэтому оно отдельно от порога к original. Если ABI/import self-check не проходит, запуск завершается явно; автоматического незаметного fallback нет. Async latest-frame worker предотвращает рост очереди геометрии. Возможность raw service 10 Hz оценивается отдельно по live/p95; obstacle detector в этом репозитории не реализован.')
    doc.p('Итог по второму вопросу: raw consumer в live replay обслуживал 10 Hz и очередь геометрии не росла. Непрерывный положительный горизонт НЕ обеспечен: на быстрых runs выбранный backend имел его примерно в 55–65% всех raw frames, на платформенном run — в 92%. Поэтому формулировка D относится к вычислительной тяжести; безусловное «async делает всю систему пригодной» эти данные не подтверждают. Рекомендуется остановить performance rewrite и передавать потребителю честный age/horizon/status, не скрывая недостаток горизонта.')
    doc.p('Основные файлы: hypothesis_results.csv, waterfall.csv, c4_before_after.csv, call_counts.csv, tree_reuse.csv, solver_audit.csv, native_equivalence.csv, full_equivalence.csv, async_runs.csv, geometry_age.csv, horizon_age.csv, docker_benchmark.csv, memory.csv, freeze.json, MANIFEST.json. Скрипты воспроизведения и команды — MVP/performance_final2/README.md.')
    doc.save('REPORT_C4_OPTIMIZATION_FINAL')
    ad=Document('ASYNC GEOMETRY REPORT')
    ad.p('Один active solve + один newest pending. Полные source runs; никаких средних 100 ms вместо измеренной wall latency. Raw service и geometry solve приведены отдельно. State presence не приравнивается к положительному remaining horizon.')
    ad.table(['Backend/run','Raw','Solves','Superseded','Published','Reuse','Age median s','Age p95 s','Min horizon m','Valid-state s','Positive-horizon s'],[[r[k] for k in ('backend','raw_frames','fresh_solves','superseded','published','reuse_frames','geometry_age_median_s','geometry_age_p95_s','min_remaining_horizon_m','valid_state_seconds','positive_horizon_seconds')] if False else [r['backend']+'/'+r['run']]+[r[k] for k in ('raw_frames','fresh_solves','superseded','published','reuse_frames','geometry_age_median_s','geometry_age_p95_s','min_remaining_horizon_m','valid_state_seconds','positive_horizon_seconds')] for r in async_rows])
    ad.table(['Backend/run','Solve median ms','Solve p95 ms','Raw median ms','Raw p95 ms','Raw >100ms','Publish Hz','First state frame'],[[r['backend']+'/'+r['run']]+[r[k] for k in ('solve_median_ms','solve_p95_ms','raw_service_median_ms','raw_service_p95_ms','raw_over_100ms','publish_Hz','first_available_frame')] for r in async_rows])
    ad.p(f'Live replay: {len(lr)} frames, max pending={live["max_pending"]}, raw service p95={pct([r["service_seconds"]*1000 for r in lr],95):.2f} ms, arrival lateness p95={pct([r["arrival_lateness_seconds"]*1000 for r in lr],95):.2f} ms. Полные runs — independent-consumer discrete-event модель; только этот дополнительный replay использует фактическую одновременную работу worker и consumer. Geometry queue bounded архитектурно, это не гарантия дедлайнов самого raw consumer или точности старой геометрии.')
    ad.image('figures/async_age.png','Geometry age');ad.save('ASYNC_GEOMETRY_REPORT')
    rec=Document('FINAL PERFORMANCE RECOMMENDATION');rec.p(f'D. Выбрать native_batch_spatial, 1 worker. Full {winner["median_ms"]:.1f}/{winner["p95_ms"]:.1f} ms median/p95; C4 {winner["C4_median_ms"]:.1f}/{winner["C4_p95_ms"]:.1f} ms. {speedup:.2f}× full относительно Python, {c4speed:.2f}× C4. Windows 200 и Linux 200 стартов exact. Нет 10 Hz геометрии.');rec.p('Остановить полный solver rewrite перед сдачей. Latest-frame async worker удерживает очередь и raw service: live p95 около 27 ms. Но непрерывный положительный горизонт НЕ обеспечен: 55–65% raw frames на быстрых runs, около 92% на платформенном. Худшие случаи — NO_POINTS, 8 m исходного горизонта против 23–34 m движения за время старения. Не скрывать это фразой «async всё решил»; age/distance/horizon остаются обязательными входами потребителя. Frozen fallback доступен явно.');rec.save('FINAL_PERFORMANCE_RECOMMENDATION')
    lc.save(OUT/'report_summary.json',dict(best=best,production_backend='native_batch_spatial',verdict='D',metrics=metrics,speedup=speedup,C4_speedup=c4speed,reduction=reduction,default_native_gate=enabled,async_runs=async_rows,
        live_raw_p95_ms=pct([r['service_seconds']*1000 for r in lr],95),Amdahl_fraction=fraction,Amdahl_upper=1/(1-fraction),final_confirmation=final_metrics if confirmation else None))
    print('REPORT_WRITTEN',winner,flush=True)

if __name__=='__main__':main()
