"""Attach post-replay viewer checks and all stage timings to the completed report."""
import csv,html,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
dest=ROOT/'COPY_MAIN/results/INPUT5HZ_SOLVE2_REPLAY'
def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))
report=dest/'REPORT.md'
text=report.read_text(encoding='utf-8').split('\n## Просмотр нового прогона')[0]
text=text.replace('LAS отсутствуют.','Во время измеряемого replay LAS не создавались; экспорт для просмотра выполнен отдельно после него.')
text=text.replace('## Изменения результатов — сравнение, не независимый GT','## Изменения результатов — сравнение, не независимый GT\n\nВ этом разделе геометрия и позы сравниваются с полным исходным COPY_MAIN (FULL_COPY_MAIN), где больше общих исходных кадров. Таблица скорости выше сравнивает также непосредственно предыдущий INPUT10HZ_SOLVE5_REPLAY. Ни один сохранённый результат не использовался новым inference.') if 'В этом разделе геометрия' not in text else text
audit=load(ROOT/'results_copy_main_5hz_train_las/audit.json')
ui=load(dest/'UI_VERIFICATION.json');bench=load(dest/'extension20/BENCHMARK.json')
text+=f'''
## Просмотр нового прогона

Удалены 297 прежних тестовых LAS (4 832 443 825 байт); сохранены их прогнозы,
отчёты и сведения для восстановления в DELETED_10HZ_VIEWER_LAS.json.
После завершения замера экспортированы 373 новых LAS, {audit['total_bytes']:,} байт,
в `results_copy_main_5hz_train_las`. Экспорт и проверка чтением заняли
{audit['seconds']:.1f} с и не включены в время replay. Все исходные XYZ,
интенсивности, номера точек, кадров и timestamps проверены; история не дублируется
в облаке T. Расчётные линии имеют отдельные слои и не считаются попаданиями.

[Открыть срез бокса, исходный кадр 190](http://127.0.0.1:8768/predicted_train.html?source=190&distance=35.5&width=1&extension=20).
{ui['slice_text']}. Длина без продолжения {ui['original_m']:.3f} м,
с +20 м {ui['extended_m']:.3f} м. В этом кадре бокс уже внутри найденной геометрии.

![Срез бокса на кадре 190](slice_190.png)

[Вид от LiDAR, кадр 224](http://127.0.0.1:8768/predicted_drive.html?t=22.45).
{ui['raw_224']}
Все принятые кадры 218,220,…230 при отказе регистрации показаны последовательно;
их облака не скрыты, позы не выдуманы. Индексы в интерфейсе исходные, поэтому
кнопка следующего входного кадра прибавляет 2, следующего расчётного LAS — обычно 4.

![Исходный вид при перекрытии](drive_224.png)

Габарит 2,1×3 м, порог проникновения глубже 3 см, сохранение T при проезде,
переключатели +5/+20/выключено и цвета остались прежними. Будущие облака
доступны только для разрешённого пользователем 3D-просмотра при валидных позах.
Во входной поток нового прогона пропущенные исходные кадры не возвращаются.
Проверки: VIEWER_DATA_VERIFICATION.json, UI_VERIFICATION.json, LAS HTTP,
virtual train, route, envelope и JS syntax. Ошибок JavaScript/WebGL нет.

## Отдельный замер продолжения +20 м на новом результате

На {bench['frames']} новых доступных геометриях — первый вызов медиана
{bench['first_call_per_frame_ms']['median']:.4f} мс, p95
{bench['first_call_per_frame_ms']['p95']:.4f} мс. Повторные вызовы: медиана
{bench['repeated_calls_ms']['median']:.4f} мс, максимум
{bench['repeated_calls_ms']['maximum']:.3f} мс. Превышений 20 мс — 0.
Замер Windows только для постпроцессора; он не включает чтение и подготовку
облака к просмотру. Это по-прежнему отдельный шаг 2D-просмотрщика,
не изменение замороженного расчётного графа.

Проверка со скрытием хвоста сохранённой модели, {bench['withheld_tail_proxy_error_m']['n']}
случая: ошибка на 20 м медиана {bench['withheld_tail_proxy_error_m']['median']*100:.2f} см,
p95 {bench['withheld_tail_proxy_error_m']['p95']*100:.2f} см,
максимум {bench['withheld_tail_proxy_error_m']['maximum']*100:.2f} см.
Это внутреннее сравнение, не независимый GT. Полные числа в extension20/BENCHMARK.json;
поле frame в этой таблице означает индекс потока 5 Гц (для исходного умножить на 2).

## Практический вывод

Общий темп близок к записи, четыре выбранных расчёта пропущены из-за отсутствия
поз, ещё три выполненных расчёта не дали финальной геометрии. Строгое требование
200 мс для каждого принятого кадра не выполнено: p95 готовности входного кадра
222 мс, максимум 1311 мс. Задержка публикации p95 694 мс, максимум 1514 мс.
Единичные просадки остаются, хотя растущей очереди к концу записи нет.

Снижение частоты изменяет сшивку и состав истории. Медианная дальность модели
почти прежняя, но есть сокращения и увеличения на отдельных кадрах. Накопленные
оценки поз заметно расходятся с 10 Гц: до 16,54 м на общей записи, при локальном
расхождении движения за 2,2 с p95 10,37 см. Это расхождение двух оценок,
не ошибка по истинной траектории; глобальную сшивку нельзя считать эквивалентной.
Для визуальной оценки сохранены исходные облака в координатах текущего LiDAR.
'''
report.write_text(text,encoding='utf-8')
body=[];inside=False
for line in text.splitlines():
    if line.startswith('|'):
        if not inside:body.append('<table>');inside=True
        if not line.startswith('|---'):body.append('<tr>'+''.join('<td>'+html.escape(c.strip())+'</td>' for c in line.strip('|').split('|'))+'</tr>')
    else:
        if inside:body.append('</table>');inside=False
        if line.startswith('!['):
            label,path=line[2:].split('](',1);body.append(f'<img style="max-width:100%" src="{html.escape(path[:-1])}" alt="{html.escape(label)}">')
        elif line.startswith('[') and line.endswith(').'):
            label,url=line[1:].split('](',1);body.append(f'<p><a href="{html.escape(url[:-2])}">{html.escape(label)}</a></p>')
        elif line.startswith('## '):body.append('<h2>'+html.escape(line[3:])+'</h2>')
        elif line.startswith('# '):body.append('<h1>'+html.escape(line[2:])+'</h1>')
        elif line:body.append('<p>'+html.escape(line).replace('**','')+'</p>')
(dest/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>COPY_MAIN · 5 Гц / каждый второй</title><style>body{font:16px/1.55 system-ui;max-width:1250px;margin:36px auto;padding:0 24px;color:#23374d;background:#f8fafc}table{border-collapse:collapse;width:100%;background:white}td{border:1px solid #d7e0e8;padding:8px}tr:first-child{font-weight:bold;background:#e5edf5}</style>'+''.join(body),encoding='utf-8')
with (dest/'STAGES_MS.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['stage','calls','median_ms','p95_ms','maximum_ms'])
    for stage,row in load(dest/'STAGES_MS.json')['timings'].items():w.writerow([stage,row['n'],row['median_ms'],row['p95_ms'],row['max_ms']])
print('REPORT_FINALIZED')
