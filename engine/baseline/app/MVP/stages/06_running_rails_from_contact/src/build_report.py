"""Read-only aggregation and Russian report; never changes frozen predictions."""
from rr_common import *
from run_inference import check_freeze
from collections import Counter
import html,re,shutil

def read_csv(path):
    out=[]
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):
            row={}
            for k,v in r.items():
                if not v:row[k]=None
                elif v in ('True','False'):row[k]=v=='True'
                else:
                    try:row[k]=float(v)
                    except ValueError:row[k]=v
            out.append(row)
    return out
def fmt(x,scale=1,digits=2):return 'нет данных' if x is None or not np.isfinite(float(x)) else f'{float(x)*scale:.{digits}f}'
def table(headers,rows):return '\n| '+' | '.join(headers)+' |\n| '+' | '.join('---' for _ in headers)+' |\n'+''.join('| '+' | '.join(str(v) for v in r)+' |\n' for r in rows)+'\n'
def metric(metrics,method,lo,hi):return next((r for r in metrics if r['method']==method and r['lo']==lo and r['hi']==hi),{})
def errors_table(metrics,method):
    rows=[]
    for lo,hi in ((30,50),(50,75),(75,100),(100,125),(125,150)):
        r=metric(metrics,method,lo,hi);rows.append([f'{lo}–{hi}',r.get('starts',0),r.get('stations',0),fmt(r.get('near_error_p95'),100),fmt(r.get('far_error_p95'),100),fmt(r.get('center_error_p95'),100),fmt(r.get('alpha_error_deg_p95')),fmt(r.get('both_coverage95'),100)])
    return table(['Range, м','Стартов','Сечений','Near p95, см','Far p95, см','Center p95, см','α p95, °','Оба в 95%, %'],rows)
def inline(s):
    s=html.escape(s);s=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',r'<a href="\2">\1</a>',s);s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s);s=re.sub(r'`([^`]+)`',r'<code>\1</code>',s);return s
def render(md):
    result=[];lines=md.splitlines();i=0
    while i<len(lines):
        s=lines[i]
        if not s.strip():i+=1;continue
        if s.startswith('```'):
            block=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):block.append(lines[i]);i+=1
            result.append('<pre>'+html.escape('\n'.join(block))+'</pre>');i+=1;continue
        if s.startswith('| '):
            rows=[]
            while i<len(lines) and lines[i].startswith('| '):
                cells=[c.strip() for c in lines[i].strip('| ').split('|')]
                if not all(c=='---' for c in cells):rows.append(cells)
                i+=1
            result.append('<div style="overflow:auto"><table>'+''.join('<tr>'+''.join(('<th>' if j==0 else '<td>')+inline(c)+('</th>' if j==0 else '</td>') for c in r)+'</tr>' for j,r in enumerate(rows))+'</table></div>');continue
        if s.startswith('#'):
            level=min(6,len(s)-len(s.lstrip('#')));result.append(f'<h{level}>'+inline(s[level:].strip())+f'</h{level}>');i+=1;continue
        if s.startswith('- '):
            result.append('<ul>')
            while i<len(lines) and lines[i].startswith('- '):result.append('<li>'+inline(lines[i][2:])+'</li>');i+=1
            result.append('</ul>');continue
        result.append('<p>'+inline(s)+'</p>');i+=1
    css='body{font:16px/1.6 system-ui,sans-serif;color:#183044;background:#f3f5f7;margin:0}main{max-width:1180px;margin:auto;background:white;padding:36px}h1{font-size:32px}h2{margin-top:40px;border-top:1px solid #dce3e8;padding-top:18px}a{color:#006b9e}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0}th,td{border:1px solid #cdd6dd;padding:7px;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#eaf0f5}code,pre{background:#eff3f6;padding:2px 5px}pre{overflow:auto;padding:15px}p{max-width:1100px}'
    return '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>STEP6 — полный отчёт</title><style>'+css+'</style><main>'+''.join(result)+'</main></html>'

def aggregate_deliverables(lock):
    names=lock['methods'];stations=[];starts=[];ranges=[];decomp=[];aux={k:[] for k in ('availability','offsets','overlap','curves','side')}
    for group in ('phase_b','phase_e'):
        stations.extend(dict(r,cohort=group) for r in read_csv(OUT/group/'per_station.csv') if r['method'] in names)
        starts.extend(dict(r,cohort=group) for r in load(OUT/group/'per_start.json') if r['method'] in names)
        ranges.extend(dict(r,cohort=group) for r in load(OUT/group/'range_metrics.json') if r['method'] in names)
        decomp.extend(dict(r,cohort=group) for r in load(OUT/'oracle_decomposition'/group/'metrics.json') if r['method'] in names)
        for k in aux:aux[k].extend(dict(r,cohort=group) for r in read_csv(OUT/'audit'/group/(k+'.csv')))
    write_csv(OUT/'per_station.csv',stations);write_csv(OUT/'per_start.csv',starts);write_csv(OUT/'range_metrics.csv',ranges);write_csv(OUT/'oracle_decomposition.csv',decomp)
    write_csv(OUT/'alpha_gt_pred.csv',[{k:r[k] for k in ('run','frame','cohort','method','station','range_m','gt_available','alpha_pred','alpha_gt','alpha_error_deg','alpha_sigma')} for r in stations]);write_csv(OUT/'offset_stability.csv',aux['offsets']);write_csv(OUT/'side_switch.csv',aux['side'])
    for k in ('availability','overlap','curves'):write_csv(OUT/'audit'/(k+'.csv'),aux[k])
    methods=[];corridors=[];thresholds=[]
    for group in ('phase_b','phase_e'):
        for name in names:
            rr=[r for r in stations if r['cohort']==group and r['method']==name and r['gt_available'] and r['station']>=8];ss=[r for r in starts if r['cohort']==group and r['method']==name]
            methods.append(dict(cohort=group,method=name,starts=len(ss),seed_available=sum(r['status']=='AVAILABLE' for r in ss),evaluated_stations=len(rr),near_p95=stats([r['near_error'] for r in rr]).get('p95'),far_p95=stats([r['far_error'] for r in rr]).get('p95'),center_p95=stats([r['center_error'] for r in rr]).get('p95'),alpha_p95_deg=stats([abs(r['alpha_error_deg']) for r in rr]).get('p95')))
            for lo,hi in ((0,10),(10,20),(20,30),(30,40),(40,50),(50,60),(60,75),(75,100),(100,125),(125,150),(30,50),(50,75)):
                bb=[r for r in rr if lo<=r['range_m']<hi]
                if not bb:continue
                row=dict(cohort=group,method=name,lo=lo,hi=hi,n=len(bb),coverage95=np.mean([r['both_inside95'] for r in bb]),coverage99=np.mean([r['both_inside99'] for r in bb]),development_scale_applied=group=='phase_e')
                for level,chi in ((95,5.991464547),(99,9.210340372)):
                    scale=np.quantile([np.sqrt(max(r['near_mahal2'],r['far_mahal2'])/chi) for r in bb],level/100);row[f'diagnostic_extra_scale_for_{level}']=float(scale)
                    for side in ('near','far'):
                        row[f'{side}_coverage{level}']=np.mean([r[side+f'_inside{level}'] for r in bb])
                        for axis in ('lateral','vertical'):
                            width=np.array([2*np.sqrt(chi)*r[side+'_sigma_'+axis] for r in bb]);row[f'{side}_width{level}_{axis}_p50']=float(np.median(width));row[f'{side}_width{level}_{axis}_p95']=float(np.quantile(width,.95));row[f'diagnostic_{side}_width{level}_{axis}']=float(np.median(width)*scale)
                corridors.append(row)
                for side in ('near','far','center'):
                    for cm in (5,10,20,50,100):thresholds.append(dict(cohort=group,method=name,lo=lo,hi=hi,rail=side,threshold_cm=cm,exceedances=sum(r[side+'_error']>cm/100 for r in bb),stations=len(bb)))
    write_csv(OUT/'per_method.csv',methods);write_csv(OUT/'corridor_metrics.csv',corridors);write_csv(OUT/'failures/thresholds.csv',thresholds)
    return stations,starts,ranges,decomp,aux,methods,corridors,thresholds

def main():
    lock=check_freeze();name=lock['best_development'];stations,starts,ranges,decomp,aux,methods,corridors,thresholds=aggregate_deliverables(lock);dev=load(OUT/'phase_b/range_metrics.json');bench=load(OUT/'phase_e/range_metrics.json');bm=[r for r in starts if r['cohort']=='phase_e' and r['method']==name];ds=[r for r in starts if r['cohort']=='phase_b' and r['method']==name];gates=load(OUT/'models/phase_b_signal_gates.json');runtime=load(OUT/'runtime/summary.json');cases=load(OUT/'gallery/cases.json');selection=load(OUT/'gallery/selection.json');las=load(OUT/'las/index.json');auditb=load(OUT/'audit/phase_b/geometry_audit.json');audite=load(OUT/'audit/phase_e/geometry_audit.json');loro=[r for r in read_csv(OUT/'uncertainty/corridor_calibration_loro.csv') if r['method']==name];maxeval=max(r.get('max_evaluated_range') or 0 for r in bm);maxpred=max(r.get('max_predicted_range') or 0 for r in bm);br=[r for r in stations if r['cohort']=='phase_e' and r['method']==name and r['gt_available'] and r['station']>=8];badcases=[r for r in cases if r['group']=='phase_e' and r['far_max']>.2];cfg=load(OUT/'configs.json')[name]
    md=f'''# STEP 6. Восстановление ходовых рельсов из контактного рельса

**Лучший метод по development: {name}. CR-вход: зафиксированный C4 STEP5.2. Seed: первые 8 м текущего кадра. Режим: pure prediction после seed.**

**Вердикт C: CR-only is insufficient without roll observations.** Кривая CR и начальная пара рельсов позволяют строить геометрическую гипотезу и коридоры, но не определяют будущий поворот сечения. Проверка не подтверждает точные дальние линии и переносимую гарантию 95% покрытия. Код C4, его параметры и сохранённые решения не изменялись. **У нового адаптера STEP6 также найден spline overshoot до 23,6 см: это собственный дефект STEP6, сохранённый в итоговой статистике, а не ошибка, автоматически приписанная C4 или GT.**

[Галерея всех категорий](gallery.html) · [Худшие случаи](gallery.html#worst_far) · [Все случаи >20 см](gallery.html#all_gt20cm) · [CSV](per_station.csv) · [LAS](las_exports.csv) · [Research freeze](research_freeze.json)

## Первая страница: итоговый reused research benchmark

Две ранее использованные записи, {len(bm)} стартов. Seed доступен в {sum(r['status']=='AVAILABLE' for r in bm)}. В 553 стартах нет начального решения C4, ещё в 4 недостаточна пара ходовых рельсов в seed. Это **reused research benchmark, не новая слепая валидация**. Значения ниже — по доступным общим сечениям; старты и соседние сечения зависимы.
'''+errors_table(bench,name)
    md+=f'''Максимальная дальность сохранённого прогноза: **{maxpred:.2f} м**; максимальная дальность с доступным GT: **{maxeval:.2f} м**. Это максимум отдельного старта, а не дальность гарантированной работы. Дальше окончания C4 основной режим рельсы не дорисовывает.

После seed расхождение дальнего рельса >20 см встретилось в **{len(badcases)} стартах** benchmark; **{sum(r['far_error']>.2 for r in br)} из {len(br)}** проверенных сечений. Все такие случаи находятся в галерее. Их нельзя автоматически приписать только алгоритму: GT получен из реальной разметки и оценённой регистрации.

Номинальные коридоры после зафиксированной development-калибровки:
'''
    cb=sorted([r for r in corridors if r['cohort']=='phase_e' and r['method']==name and (r['lo'],r['hi']) in ((30,50),(50,75),(75,100),(100,125))],key=lambda r:r['lo'])
    md+=table(['Range, м','Far ширина 95% lat, см','Far ширина 95% vert, см','Оба 95%, %','Оба 99%, %','Эмпирический множитель для 95%'],[[f"{r['lo']}–{r['hi']}",fmt(r['far_width95_lateral_p50'],100),fmt(r['far_width95_vertical_p50'],100),fmt(r['coverage95'],100),fmt(r['coverage99'],100),fmt(r['diagnostic_extra_scale_for_95'])] for r in cb])
    md+='Ширины полные, медианные; lat/vert заданы поперёк прогнозного пути. Основной коридор — двумерный эллипс с коррелированными осями. Последний столбец — **послефактум диагностический** коэффициент на этой же выборке, он не применён к прогнозам и не доказывает переносимость. В `corridor_metrics.csv` также есть p95 ширины, 99%, отдельное покрытие near/far и эмпирические ширины для 95/99% одновременного покрытия.\n'
    md+='''
## Что реализовано и откуда параметры

Новый код находится в `MVP/stages/06_running_rails_from_contact/`, результаты — в `results_running_rails_from_contact/`. Предыдущие stages импортируются только для чтения. Использованы сохранённые причинные решения C4: 108 development + 1422 benchmark. SHA закреплены в `dependency_lock.json` и `c4_input_index.json`; STEP5.3 не является зависимостью.

C4 не запускается заново. STEP6 получает его исходные anchors, выбранные точки CR с происхождением и оценённые local frames. Через сохранённые anchors проводится естественный C2 cubic spline. Это адаптер непрерывной геометрии STEP6; сами anchors C4 не перезаписываются. В коде предусмотрено пропускание соседних опор ближе 5 см для параметризации; аудит показал, что во всех доступных входах с ≥2 опорами их количество сохранено. В 81 старте с единственной C4 опорой создаётся только seed-линия 0–8 м, без заявления о дальнем CR-прогнозе. Bishop frame переносится с внутренним шагом 0,25 м. Научное сравнение выполняется в общих поперечных плоскостях с шагом station 1 м и конечной точкой; это дискретизация непрерывной модели, не метрика по плотности LAS.

В seed текущие координаты отбираются в 0–8 м до чтения их class1. Замороженный STEP1 локализует пару; внутри каждой метровой секции вычисляются головы размеченных компонентов (верхние 30% высоты, медиана, удаление точных/миллиметровых дублей). Требуются минимум две подходящие секции. Из них оцениваются α, d, h_near, g, h_far, median/MAD, тренд и covariance. Seed GT разрешён постановкой: это исследование при известной начальной паре, не полностью label-free старт.

После seed predictor получает только числовую CR-геометрию, seed и CR-profile оценки. Ни class1, ни сырые точки ходовых рельсов дальней зоны в него не поступают. Все результаты inference записываются до оценки будущего class1. Формальные тесты запрещают поля raw/future/labels и импорты evaluator в predictor. Проверено происхождение C4: source_frame ≤ T.

Фиксированные гипотезы неопределённости: стартовый floor α 0,1°, offsets 4/4/6/4 мм; CR position σ — max(1 см, медианный template residual) плюс 0,5 см при ≥5 уникальных точках или 1,5 см при слабой поддержке. Tangent σ — минимум 0,1° или четверть локального изменения касательной. Это инженерные priors, не измеренная точность LiDAR. Их достаточность проверяется coverage, а не объявляется заранее.

Статистики прежнего исследования 0,68/2,20/1,56 м не использованы как нормативная истина. Основная геометрия калибруется отдельно для каждого старта. O2 использует prior только из development seed с исключением собственной записи; O0/O1 работают от своего seed.

## Наблюдаемость и чувствительность

Из C(s) известны положение и касательная. Для любого гладкого α(s), совпадающего на первых 8 м и отличающегося дальше, можно построить другую пару N/F при той же C(s). Поэтому α не наблюдаем из одной центральной линии. Bishop выбирает систему отсчёта с минимальным вращением, но не измеряет физический cant.

N = C − q·d·b − h_near·n; F = C − q·(d+g)·b − h_far·n. b = cos(α)e1 + sin(α)e2; n = −sin(α)e1 + cos(α)e2. Знаки q берутся из C4 seed. h_near и h_far хранятся раздельно. При точном определении b по паре головок их высоты в собственной плоскости пары совпадают алгебраически; это не предположение о равенстве их Z в sensor/world frame.

Ошибка поворота δα даёт ровно 2r·|sin(δα/2)| ≈ r·|δα|. Для иллюстративных радиусов anchors:
'''
    md+=table(['δα, °','Near r=0,68 м, см','Far r=2,20 м, см'],[[deg,fmt(2*.68*np.sin(np.deg2rad(deg)/2),100),fmt(2*2.2*np.sin(np.deg2rad(deg)/2),100)] for deg in (.1,.25,.5,.75,1,1.5,2)])
    md+='Чтобы одна только roll-составляющая ошибки дальнего рельса не превышала 2 см, нужен угол около 0,52°; для 5 см — около 1,30°. Остальные источники ошибки добавляются. Численный контрпример с одной CR и разными дальними рельсами: `alpha_observability/identical_cr_different_rails.npz`.\n'
    md+='''
## Методы и честный порядок выбора

Phase A: 36 стратифицированных стартов, 22 конфигурации всех основных семейств. Phase B: 8 α-конфигураций + 16 offset-контролей на всех 108 development-стартах. Phase C: leave-one-development-run-out для ранжирования и калибровки, с сохранением лучшего метода без каждой записи. Phase D: три разные семьи и два baseline зафиксированы. Phase E: один полный reused benchmark, без перенастройки после его просмотра.

Основная функция выбора: p95 far, 0,5·p95 center, доля >20 см, штраф за отклонение покрытия от 95% и ширину коридора; отдельно в диапазонах 30–50, 50–75, 75–100. Записи имеют равный вес, добавлен штраф за худшую запись. Общая цель: 70% CR1 и 30% диагностического CR0. Короткие записи без дальнего GT не превращаются в нулевую ошибку. Точная формула и результаты каждого fold находятся в `src/select_methods.py`, `phase_b/ranking.csv`, `cross_run.csv`.
'''
    modelrows=[('B0','Постоянные offsets в сохранённом local frame C4; baseline, возможна дискретность frame.'),('B1','Bishop, постоянный средний α; covariance растёт.'),('B2','Sensor-up/world-up proxy с сохранением начального cant; ось не считается gravity truth.'),('B3','Random walk α без новых наблюдений: среднее постоянно, растёт variance. Это не новые измерения α.'),('B4','Гладкое продолжение seed-тренда α с затуханием, L=10/25/50 м; C2 стык на границе seed.'),('B5','β профиля CR, прямое использование или robust Kalman update; проверено в A, не продвинуто после провала signal gate.'),('B6','Frenet normal, диагностическая гипотеза; не продвинута.'),('B7','Кривизна→roll: LORO диагностирован, самостоятельный predictor не продвинут, поскольку условный gate не пройден.'),('B8','Seed cant + затухание тренда на 25 м, коэффициенты тренда 0,25/0,5/1.'),('B9','Вероятностное состояние α и offsets: тренд α из seed и его неопределённость распространяются аналитически; offsets random walk. Отдельный последовательный posterior α′ не оценивается без наблюдений.'),('O0/O1/O2','Свой постоянный seed / random walk offsets / мягкое стягивание к development seed prior.')]
    md+=table(['Семейство','Реализация'],modelrows)
    md+=f'''Выбран {name}; остальные финалисты: {', '.join(lock['top3'][1:])}. Для лучшего метода L={cfg['damping_m']:.0f} м; α′ берётся из seed. При d=max(s−8,0) используется τ=d³/(d²+4), z=min(τ/L,1), α=α0+α′·L·(z−z²+z³/3). На границе seed совпадают значение и первые две производные. O1 не меняет среднее offsets, а добавляет дисперсию {cfg['offset_q_m_sqrt_m']*1000:.1f} мм/√м. Поэтому совпадающие средние O0/O1 ожидаемы и не считаются независимыми подтверждениями.

## Development и сравнение финалистов
'''+errors_table(dev,name)
    md+=table(['Метод','Cohort','Far p95 30–50, см','Far p95 50–75, см','Far p95 75–100, см','Center p95 50–75, см'],[[n,g,fmt(metric(mm,n,30,50).get('far_error_p95'),100),fmt(metric(mm,n,50,75).get('far_error_p95'),100),fmt(metric(mm,n,75,100).get('far_error_p95'),100),fmt(metric(mm,n,50,75).get('center_error_p95'),100)] for g,mm in [('development',dev),('benchmark',bench)] for n in lock['methods']])
    screen=load(OUT/'phase_a/range_metrics.json');md+='Остальные гипотезы на одинаковом screening subset (до продвижения):\n'+table(['Метод','Far p95 30–50, см','α p95, °','Сечений'],[[n,fmt(metric(screen,n,30,50).get('far_error_p95'),100),fmt(metric(screen,n,30,50).get('alpha_error_deg_p95')),metric(screen,n,30,50).get('stations',0)] for n in ('B1_BISHOP','B2_SENSOR_UP','B2_WORLD_UP','B5_PROFILE','B5_FILTER_0.1','B6_FRENET')])
    md+=f'''## Проверка CR-profile roll и curvature prior

β оценивался по выбранным **только C4** CR-точкам в окне ±2 м, без rail GT. Параметры: вращение шаблона и два поперечных смещения; несколько стартов оптимизации, robust loss, проверка extent, Hessian/σ, competing minima и bounds. Затем, отдельно после prediction, изменения β относительно своего seed сравнивались с изменениями α_GT.

Медианная корреляция по development-записям: **{gates['profile_median_run_correlation']:.3f}**. Gate: ≥3 записи с ≥20 наблюдениями и median correlation ≥0,35. **CR profile orientation is not informative enough** в данном C4-представлении. B5 не используется в итоговом методе. Это отрицательный результат для отобранной C4 поддержки; выбор точек фиксированным профилем сам создаёт selection bias. Он не доказывает невозможность оценки roll по полному плотному CR-профилю другой модели.
'''
    profile=read_csv(OUT/'cr_profile_roll/phase_b_by_run.csv');md+=table(['Запись','N','corr','bias, °','median abs, °','p90, °','p95, °'],[[r['run'],int(r['n']),fmt(r['correlation'],digits=3),fmt(r['bias_deg']),fmt(r['median_abs_deg']),fmt(r['p90_abs_deg']),fmt(r['p95_abs_deg'])] for r in profile])
    curvature=read_csv(OUT/'curvature_roll/phase_b_cross_run.csv');md+='Curvature prior также не прошёл заранее записанный gate: median LORO улучшение p95 >10% и положительный эффект минимум в 75% записей. Детальные folds: `curvature_roll/phase_b_cross_run.csv`. Frenet normal не равен поперечной оси рельсов, особенно при слабой/шумной кривизне. World/sensor-up — proxy; физическая гравитационная система в этих данных не доказана.\n'
    md+='''
## Oracle-разложение: CR, α и offsets

CR0 — будущий class2 proxy, сглаженный только внутри непрерывных компонентов (Gaussian σ≈1 м, опоры≈2 м). Начальные 8 м те же. Это диагностический верхний ориентир, а не безошибочная физическая линия. Oracle α и offsets получены из будущих ходовых рельсов и запрещены в production-like inference.

Для сопоставления компонентов все варианты пересекают **те же поперечные плоскости C4**, на общем доступном горизонте. Никакого независимого nearest-point matching каждой нити. Ниже медиана per-start p95 дальнего рельса, а не pooled p95; старт с отсутствующим сопоставлением пропускается с явным учётом N.
'''
    variants=['production_CR_production_alpha','oracle_CR_production_alpha','production_CR_oracle_alpha','production_CR_oracle_offsets','oracle_CR_oracle_alpha','all_oracle'];orows=[]
    for v in variants:
        row=[v]
        for lo,hi in ((30,50),(50,75),(75,100)):
            rr=[r for r in decomp if r['cohort']=='phase_e' and r['method']==name and r['variant']==v and r['lo']==lo and r['hi']==hi];row.append(fmt(np.median([r['far_p95'] for r in rr]) if rr else None,100)+f' (N={len(rr)})')
        orows.append(row)
    md+=table(['Вариант','30–50, см','50–75, см','75–100, см'],orows)
    md+='Отдельно чистая inverse geometry на расширенном CR0; это другой доступный горизонт и не прирост дальности C4:\n'
    ext=load(OUT/'oracle_decomposition/extended_phase_e/range_metrics.json');ocr=load(OUT/'oracle_cr/phase_e/range_metrics.json');md+=table(['Вариант','50–75 far p95, см','75–100, см','100–125, см','125–150, см'],[[n]+[fmt(metric(mm,n,lo,hi).get('far_error_p95'),100) for lo,hi in ((50,75),(75,100),(100,125),(125,150))] for n,mm in [(name,ocr),('CR0_ORACLE_ALPHA',ext),('CR0_ORACLE_OFFSETS',ext),('CR0_ALL_ORACLE',ext)]])
    md+='Ноль all-oracle — алгебраическая проверка формулы и параметризации, не доказательство абсолютной точности. Улучшение oracle α показывает вклад неизвестного поворота; остаток oracle CR + oracle α с seed offsets показывает изменение anchors/offsets и неоднозначность GT. Ошибка CR может частично компенсировать roll на одной нити, поэтому исправление одного компонента не обязано улучшать каждый старт.\n'
    md+='''
## GT: две нити, доступность и ограничения

GT строится по class1 будущих кадров T+2… до непрерывного горизонта около 150 м. В каждом будущем кадре используются ближние 1–12 м, две независимые группы по lateral и верхняя поверхность. Широкий sanity check 0,8–2,5 м отделяет две компоненты; это не нормативный gauge. Движение для локальной оси берётся до накопления ≥0,5 м, с проверкой времени/регистрации. Через недопустимые разрывы путь не соединяется.

Группы хранят идентичность negative/positive lateral относительно движения, затем пересекают одну CR transverse plane. Пара пересечений должна иметь согласованный продольный progress. Это предотвращает произвольное независимое сопоставление нитей. Автоматическая кластеризация не гарантирует правильный выбор пути в каждой сложной развилке: это ограничение reference, а не скрыто исправленная разметка.

GT-отсутствие не считается failure. Все availability и причины недоступного seed сохранены. Гистограммы/метрики условны на доступность GT, а число доступных стартов показано в таблицах. Расстояние range — евклидово от LiDAR T до CR; station — монотонная длина опорного CR-полигона, не одинаковая длина дуги всех трёх линий.
'''
    md+=table(['Cohort','Seed доступен','Всего','Срезов current/future overlap','Near overlap p50/p95, см','Far overlap p50/p95, см'],[[g,sum(r['status']=='AVAILABLE' for r in ss),len(ss),au['overlap_near']['n'],fmt(au['overlap_near'].get('p50'),100)+' / '+fmt(au['overlap_near'].get('p95'),100),fmt(au['overlap_far'].get('p50'),100)+' / '+fmt(au['overlap_far'].get('p95'),100)] for g,ss,au in [('development',ds,auditb),('benchmark',bm,audite)]])
    md+='Overlap оценивает совокупность регистрации, разметки, видимости и выбора anchor; он не разделяет их ошибки. Это эмпирический масштаб разрешимости сравнения, не строгая нижняя граница sensor accuracy. Заявление «физическая точность 1 см» из этих данных не следует.\n'
    md+='''
## Неопределённость, коридоры и gaps

P_rail = P_CR + J_state P_state J_stateᵀ + J_tangent P_tangent J_tangentᵀ. Учтены CR position, две ошибки tangent, α, d, h_near, g, h_far. Для track center используется средний Jacobian с общей CR covariance; общая ошибка не делится на два как при независимых рельсах. Сохранены 3×3 covariance обоих рельсов и центра.

100000 Monte Carlo реализаций дали относительное Frobenius расхождение аналитической covariance 0,36% near и 0,20% far. Это проверка вычисления при заданных распределениях, а не доказательство правильной дисперсии на реальных данных. 1σ/2σ/3σ в CSV — радиусы Mahalanobis в 2D; их нельзя читать как одномерные 68/95/99,7%. Для 95/99% используются χ²₂=5,991/9,210.
'''
    md+=f"Для выбранного метода rail σ умножено на **{cfg['corridor_scale']:.3f}**, коэффициент получен только из development с равными весами записей и стартов. Исходная covariance заметно недооценивала расхождения. Переносимость отдельно проверена LORO:\n"
    md+=table(['Оставленная запись','Обученный scale','Сечений','Оба в 95%, %','Far ширина lat/vert, см'],[[r['held_run'],fmt(r['scale']),int(r['n']),fmt(r['coverage95'],100),fmt(r['median_fullwidth95_far_lateral'],100)+' / '+fmt(r['median_fullwidth95_far_vertical'],100)] for r in loro])
    md+='Во время tentative/gap к α variance добавляется 0,1°² на метр непрерывного gap, к CR σ — 4 мм на метр. Такие сечения имеют cr_state=1/3; cr_state=3 означает `RAILS_INFERRED_FROM_CR_STATE_GAP`. Это не наблюдаемый CR. Независимого наблюдения α там нет, confidence не сохраняется постоянным.\n'
    posthoc=load(OUT/'audit/posthoc_diagnostics.json')
    md+='Для центра пути общая ошибка CR не усредняется как независимый шум. Номинальные 95% коридоры центра:\n'+table(['Range, м','N','Покрытие 95%, %','Полная ширина lat, см','Полная ширина vert, см'],[[f"{r['lo']}–{r['hi']}",r['n'],fmt(r['coverage95'],100),fmt(r['median_fullwidth95_lateral'],100),fmt(r['median_fullwidth95_vertical'],100)] for r in posthoc['center_corridors']])
    md+='Неопределённость дальнего рельса возле 50/75/100 м (по benchmark, полная номинальная 95% ширина):\n'
    wr=[]
    for dist in (50,75,100):
        rr=[r for r in br if abs(r['range_m']-dist)<=2.5];wr.append([dist,len(rr),fmt(np.median([r['far_sigma_lateral'] for r in rr]) if rr else None,100),fmt(np.median([r['far_sigma_vertical'] for r in rr]) if rr else None,100),fmt(np.median([r['far_sigma_lateral'] for r in rr])*4.895 if rr else None,100),fmt(np.median([r['far_sigma_vertical'] for r in rr])*4.895 if rr else None,100)])
    md+=table(['Range ±2,5 м','N','σ lat, см','σ vert, см','95% lat, см','95% vert, см'],wr)
    md+='''
## Stress: seed, CR и длина калибровки

18 development-стартов, по два с длинным C4 горизонтом на запись. Синтетически смещены отдельные головы на ±1/2/5 см в двух поперечных направлениях, α0 на ±0,1/0,25/0,5/1°. Отдельно смещена CR-кривая на ±1/2/5 см и касательная на ±0,1/0,25/0,5/1°. Измерены 3D и transverse изменения около 30/50/75/100 м. Нет C4 на требуемой дистанции — `available=False`, без экстраполяции для красивой цифры.
'''
    for filename,title in [('seed_stress.csv','Изменение seed'),('cr_stress.csv','Изменение CR')]:
        data=read_csv(OUT/filename);rr=[]
        for typ in sorted(set(r['perturbation'] for r in data)):
            for magnitude in sorted(set(abs(r['magnitude']) for r in data if r['perturbation']==typ)):
                v=[r for r in data if r['perturbation']==typ and abs(r['magnitude'])==magnitude and r['available']];rr.append([typ,magnitude,len(v),fmt(np.quantile([r['near_transverse_cm'] for r in v],.95) if v else None),fmt(np.quantile([r['far_transverse_cm'] for r in v],.95) if v else None),fmt(np.quantile([r['far_3d_cm'] for r in v],.95) if v else None)])
        md+='### '+title+'\n'+table(['Возмущение','Амплитуда (м или °)','N','Near transverse Δ p95, см','Far transverse Δ p95, см','Far 3D Δ p95, см'],rr)
    length=read_csv(OUT/'seed_calibration/length_stress.csv');md+='### Seed 4/8/12/16 м\n\n12/16 м — диагностический режим с дополнительным class1 текущего кадра; основной seed остаётся 8 м. Записана фактическая последняя использованная секция, поэтому запрос 16 м не выдаётся за 16 м реальной поддержки.\n'
    lr=[]
    for L in (4,8,12,16):
        rr=[r for r in length if r['requested_seed_length']==L and r['status']=='AVAILABLE' and r.get('n',0)>0 and r.get('lo')==30];lr.append([L,len(rr),fmt(np.median([r['observed_seed_end'] for r in rr]) if rr else None),fmt(np.median([r['far_p95'] for r in rr]) if rr else None,100)])
    md+=table(['Запрошенный seed, м','Стартов с GT 30–50','Медианная фактическая station','Медиана per-start far p95, см'],lr)
    md+='''
## Смена стороны и геометрическая согласованность

Замороженный C4 хранит начальную сторону q; подтверждённых событий переключения стороны в его контракте нет. STEP6 не обнаруживает переключение по будущей разметке и не утверждает, что обработал переход. Будущий class2 используется только для поиска диагностических кандидатов противоположной стороны. Это кандидаты, включая возможную параллельную инфраструктуру/ошибки GT, не вручную подтверждённые switch events. Для production-перехода нужен явный causal side event и новая калибровка; этот интерфейс C4 не предоставляет.
'''
    ss=aux['side'];md+=f"Найдено {sum(r['evaluation_switch_candidate'] for r in ss)} диагностических 10-метровых bins противоположной стороны, из них {sum(r['evaluation_switch_candidate'] and r['c4_prediction_reaches_bin'] for r in ss)} пересекают численную дальность C4. Это не число независимых физических переключений: окна перекрываются. Все строки в `side_switch.csv`.\n"
    md+=f"В первоначальном численном аудите nonforward curves: {auditb['nonforward_curves']}/{audite['nonforward_curves']} (development/benchmark). Все шесть benchmark-флагов относятся к двум seed-only стартам: конечная station 8 м отличается от целочисленной сетки лишь на 1,78·10⁻¹⁵ м и фактически дублирует последнюю точку. Это дефект дедупликации endpoint в STEP6, а не обратный ход рельса; после freeze он не исправлялся и сохранён в `audit/near_duplicate_stations.csv`. Пар не соседних samples ближе 2 см: {auditb['nonlocal_close_curves']}/{audite['nonlocal_close_curves']}. Дискретная проверка на шаге 1 м не доказывает отсутствие любого самопересечения между samples. `audit/curves.csv` содержит каждую кривую.\n"
    md+=f"### Отдельный дефект STEP6: overshoot гладкой интерполяции\n\nC2-гладкость сама по себе не ограничивает уход кривой между опорами. У {posthoc['cases_C2_more_10cm']} стартов максимум отличия natural cubic spline от прямых сегментов через те же C4 anchors превышает 10 см. Все сравнения находятся в `audit/interpolation_diagnostic.csv`. В frame_000191.las записи squareT_platform_squareT_switch две опоры около 56 м разделены 10,8 см, а соседние интервалы — многими метрами. Сплайн переносит локальный большой наклон на соседние интервалы и создаёт ложную волну до 23,6 см. В коротком doubleT_platform/frame_000032.las есть резкий спад последней C4 опоры, плюс собственный overshoot STEP6 до 19,4 см, влияющий и на seed. На графиках исходные опоры и прямые сегменты показаны отдельно от STEP6 spline.\n\nЭто выявленный недостаток новой реализации. Он **не исправлен после benchmark**, результаты не переоценены на подобранной по ним интерполяции. Следующая версия должна проверять ограниченную интерполяцию/регуляризацию как отдельную заранее зафиксированную гипотезу, сохранив C4. Текущая версия — research artifact, не готовый production adapter.\n"
    md+='''
## Худшие, средние и лучшие случаи

Категории формируются после всех прогнозов. Best/median/worst используют ≥30 м проверяемого горизонта, чтобы короткая лёгкая зона не выдавала себя за хороший дальний результат. Worst ранжируется по максимальной far-ошибке; best/median — по per-start p95. High-roll означает большую ошибку α, curve — изменение касательной, grade — изменение направления в начальной плоскости высоты. Все far >20 см после seed добавляются независимо от горизонта.
'''
    md+=table(['Категория','Количество'],list(selection['category_counts'].items()))
    md+='### 20 худших по дальнему рельсу\n'
    worst=sorted([r for r in cases if 'worst_far' in r['tags']],key=lambda r:-r['far_max']);md+=table(['Пример','Far max, см','Range, м','Near в том же сечении, см','Far p95, см','Диагностические признаки'],[[f"[{r['run']} / {r['file']}](gallery/{r['id']}/index.html)",fmt(r['far_max'],100),fmt(r['worst_range']),fmt(r['worst_near'],100),fmt(r['far_p95'],100),', '.join(r['cause'])] for r in worst])
    md+='Признаки в таблице вычислены по улучшению сопоставимого oracle per-start p95 минимум на 30%; это объясняющие контрфактические проверки, не окончательная ручная экспертиза. `CR` означает существенное улучшение при oracle CR **включая возможную ошибку адаптера STEP6**, `roll` — при oracle α, `offsets` — при oracle d/h/g. Несколько причин могут взаимодействовать. Большой future GT spread отмечен отдельно; такие примеры не вычеркнуты из статистики.\n'
    md+='### Разбор всех 11 случаев >20 см после просмотра графиков\n'
    notes=load(STAGE/'MANUAL_REVIEW_RU.json')
    for r in sorted([c for c in cases if 'all_gt20cm' in c['tags']],key=lambda r:-r['far_max']):md+=f"\n**[{r['run']} / {r['file']}](gallery/{r['id']}/index.html)** — "+notes[r['id']]+'\n'
    med=[r for r in cases if 'median' in r['tags']];best=[r for r in cases if 'best' in r['tags']];md+='### Средние и лучшие\n'+table(['Категория','Пример','Far p95, см','Горизонт GT, м'],[[tag,f"[{r['run']} / {r['file']}](gallery/{r['id']}/index.html)",fmt(r['far_p95'],100),fmt(r['max_evaluated'])] for tag,rr in [('Средние',med),('Лучшие',best)] for r in rr])
    md+='''В каждом примере: план, профиль высоты, α_GT/α_pred с модельной полосой, d/h/g, ошибки near/far/center, ширины коридоров, сетка сечений 10…120 м и четыре oracle-сравнения. Отсутствующий C4/GT указан явно. CR-точки — реальные выбранные C4 returns; rail curves — синтетический прогноз.

## Скорость и память

Измерение отдельно от GT, отчёта и экспорта: один процесс, 12 стартов с разными длинами, 10 повторов чистой геометрии на каждый. Подготовка включает чтение C4, SHA, текущий seed и диагностический CR-profile fit. Последний выбранному методу не нужен, но его время не скрыто.
'''
    md+=table(['Операция','Медиана, мс','p95, мс'],[['Геометрия из готового входа',fmt(runtime['geometry_ms']['p50']),fmt(runtime['geometry_ms']['p95'])],['STEP6 с подготовкой входа',fmt(runtime['total_ms']['p50']),fmt(runtime['total_ms']['p95'])]])
    md+=f"Пиковый working set измеренного процесса: {runtime['peak_process_mb']:.1f} MiB. **Стоимость самого C4 не включена:** C4 читается из сохранённых решений. Его прежнее записанное время находится отдельным столбцом `C4_saved_original_ms`, оно не измерено этим запуском. Это не заявка на end-to-end real-time производительность; параллельная работа в проекте могла влиять на CPU/диск.\n"
    md+=f'''## LAS и воспроизводимость

Экспортировано **{las['cases']} примеров × 3 LAS**: `cr_input.las`, `rails_predicted.las`, `overlay.las`. Источники не перезаписывались. Реальные C4 returns скопированы с побитовой проверкой всех исходных полей после чтения экспортированного LAS. Synthetic rail samples идут через 0,1 м только для отображения, `predicted_geometry=1`, `observed_or_predicted=1`, `rail_id=1/2`. Реальные returns имеют `predicted_geometry=0`. Extra Bytes включают station_s, sigma_lateral, sigma_vertical, alpha, alpha_sigma, source_method, cr_state и provenance. Метры и радианы описаны в VLR. У synthetic points нет физического интенсивности/времени измерения; остальные унаследованные измерительные поля не интерпретируются.

Файлы находятся в системе координат исходной зарегистрированной папки записи. Source frame и source row сохранены; произвольное совмещение разных recordings не выполняется. CSV `las_exports.csv` содержит пути, количество точек, размеры, SHA и результат проверки полей.

13 численных/causality тестов пройдены. Freeze и SHA зависимостей проверены до и после benchmark. PNG просмотрены визуально; HTML и локальные ссылки проверены статически. Браузерный UI-тест не выполнялся. `MANIFEST.json` фиксирует завершённый набор артефактов.

## Ответы на 21 вопрос

1. **Можно ли восстановить рельсы только из CR?** Можно построить семейство гипотез после seed. Однозначно — нет: α(s) не определяется центральной линией.

2. **До какой дальности?** Основной режим ограничен концом C4, в benchmark максимум {maxpred:.2f} м, с GT {maxeval:.2f} м. Это отдельные максимумы. CR0 100+ м — отдельный oracle-эксперимент.

3. **Ошибка nearest?** P95 по каждому диапазону приведён на первой странице; ближний менее чувствителен к α, но разделяет ошибку положения CR.

4. **Ошибка far?** Отдельные p95, max и пороги приведены без усреднения с near. В benchmark {len(badcases)} стартов с post-seed far >20 см.

5. **Главный bottleneck?** Неизвестная эволюция roll; в конкретных C4 примерах добавляются дрейф CR-anchor и неоднозначность reference.

6. **CR curve error?** Да, переносится почти напрямую на оба рельса. Oracle CR показывает вклад; хорошее выделение точек CR не гарантирует одинаково точную anchor-линию на частично видимом профиле.

7. **Roll α?** Ошибка усиливается рычагом дальнего рельса. Один градус даёт около 3,84 см на радиусе 2,2 м, ещё до других ошибок.

8. **Offsets d/h/g?** Свой seed достаточно устойчив для короткого переноса; oracle offsets и length stress показывают остаток. O1 расширяет uncertainty, не исправляет неизвестный средний drift.

9. **Station parameterization?** Использована общая CR transverse plane для обеих нитей. Независимое равенство arc length и произвольные nearest points не используются.

10. **Нужен ли Bishop?** Это полезная гладкая система отсчёта без искусственного twist. Она не заменяет информацию о cant и сама не обеспечивает точность.

11. **Достаточно constant α?** Как baseline — да; как дальняя точная модель — проверка не подтверждает. Неизвестный roll может измениться после seed.

12. **Нужна smooth α model?** Она улучшила development-цель и даёт гладкие кривые, но продолжает prior. Рост неопределённости остаётся необходим.

13. **Можно ли получить α из CR profile?** На замороженной выбранной C4 поддержке — надёжность не подтверждена.

14. **Насколько хорошо?** Median run correlation {gates['profile_median_run_correlation']:.3f}; bias/median/p90/p95 для каждой записи показаны выше. B5 не использован после провала gate.

15. **Помог curvature prior?** Устойчивое LORO-улучшение не подтверждено, B7 не продвинут.

16. **Помог world/sensor-up?** Это proxy-baseline; он не выиграл development selection. Гравитационная истина не доступна.

17. **Нужны sparse rail points?** Дополнительное наблюдение roll или рельсов необходимо для ограничения неоднозначности. Какой прирост даст конкретный OBS_AIDED detector, здесь не измерено: факультативная observation-aided ветвь не запускалась.

18. **Что без returns после 30 м?** Нового rail update нет; средний прогноз следует seed/CR, covariance α/offsets растёт с длиной, при CR gap дополнительно расширяется.

19. **Far uncertainty 50/75/100 м?** Числа σ и полные 95% ширины выше. Где C4/GT нет, указано отсутствие данных, а не искусственная оценка.

20. **Калиброваны ли corridors?** Development-поправка применена и заморожена. LORO и reused benchmark показывают фактическое покрытие; универсальная гарантия 95/99% не доказана.

21. **Можно ли использовать для train corridor / swept envelope?** Можно исследовать широкий геометрический prior. Нельзя считать его проверенной границей безопасности: нужен подтверждённый roll/side update и отдельная проверка допустимой ширины/покрытия. Train envelope, obstacle detection и collision logic не реализованы.

## Итог

**C. CR-only is insufficient without roll observations.** Следующий эксперимент имеет смысл направить на независимый roll update (редкие наблюдения головок или иная ориентационная информация), сохраняя C4 отдельным зафиксированным источником. В этом этапе такой дополнительный сенсорный сигнал не придуман и не подменён будущим GT.

## Файлы

- [Галерея](gallery.html), [полный CSV по сечениям](per_station.csv), [по стартам](per_start.csv), [по методам](per_method.csv).
- [Диапазоны](range_metrics.csv), [коридоры](corridor_metrics.csv), [α GT/pred](alpha_gt_pred.csv), [offsets](offset_stability.csv).
- [Oracle-разложение](oracle_decomposition.csv), [seed stress](seed_stress.csv), [CR stress](cr_stress.csv), [сторона CR](side_switch.csv).
- [Seed geometry](seed_geometry.csv), [скорость](runtime.csv), [LAS](las_exports.csv), [freeze](research_freeze.json), [manifest](MANIFEST.json).
'''
    (OUT/'REPORT_RUNNING_FROM_CR.md').write_text(md,encoding='utf-8');(OUT/'REPORT_RUNNING_FROM_CR.html').write_text(render(md),encoding='utf-8');save(OUT/'audit/report_summary.json',dict(method=name,verdict='C',benchmark_starts=len(bm),available_seeds=sum(r['status']=='AVAILABLE' for r in bm),max_prediction=maxpred,max_evaluated=maxeval,bad_far_starts=len(badcases),gallery_cases=len(cases),las_cases=las['cases'],runtime=runtime,benchmark_primary=[r for r in bench if r['method']==name],C4_unchanged=True))
    print('REPORT COMPLETE',name,len(md),flush=True)

if __name__=='__main__':main()
