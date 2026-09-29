# STEP 5.1 — причинное накопление облаков для contact rail marching

Исследование в отдельной папке. Предыдущий STEP5 и зависимости STEP1/FINAL STEP2
импортируются без изменения. Производственная версия автоматически не фиксируется.

Данные: `output/LAS_FRAMES_CLASSIFIED_20260926/ANNOTATED`.
Результаты: `results_contact_marching_temporal_fusion` в корне проекта.
Входное задание полностью сохранено в `REQUEST_RU.txt`.

## Главный опыт

Current-only bootstrap остаётся прежним. M1 получает либо T, либо T и разрешённую
непрерывную историю. W=8 м, advance=4 м, tail=8 м, gate ±3 см, overlap p90≤3 см,
PCA+Bishop, без refinement. F0 и F1 — один и тот же baseline; он заново рассчитан
и сравнивается с STEP5 побитно по точкам/шагам и точно по метрикам.

Исторические LAS уже содержат registered-map coordinates. Поэтому применяется
только `(p_map - t_T) @ R_T`, без повторного pose_k. Отдельный переход в source
coordinates используется только для source-distance и azimuth diagnostics.
Для каждого реального измерения сохраняется пара source frame / original row,
original point index, возраст, расстояния и каналы. Синтетических точек нет.

`fusion.CausalSource` не позволяет открыть будущий LAS. Общий JSON метаданных
санитизируется до past…T и отдельной pose T+1 для начального направления.
Дальняя траектория и классы не передаются в inference. Доступ к offline reference
разрешён только после завершённого prediction с SHA. Будущий GT и evaluator
берутся из STEP5 read-only, без изменения continuous-reach definition.

Окно сокращается на начале заезда или перед разрывом; кадр T не дублируется.
Distance history ограничена Euclidean displacement от T, не интегралом пути.
Фактическое число источников и причина остановки истории всегда записаны.

## Сравнения

- F0/F1, F2, F3, F4 — current и 2/3/4 trailing scans.
- H0.5/H1/H2/H4 — displacement history.
- RAW_CONCAT, VOXEL_UNIQUE 5/10/20 мм, FRAME_BALANCED.
- Exponential recency tau=0.5/1/2 м; tiny overlap-only translation ±2/5 см.
- Current-only и fused bootstrap разделены.
- M1 current/fused, M2 current/fused с теми же gates.
- Leave-one-age-out и условные F6/F8.

Weighted challenger является локальной копией движка с явно записанными заменами
статистик. `build_weighted_adapter.py` сохраняет исходный SHA и список замен.
Замороженные функции не monkeypatch-ятся. Предварительная версия весов сохранена
отдельно в calibration/weighted_preliminary и не входит в итоговые сравнения.

Маска корпуса поезда отложена по ответу пользователя. Точных sensor-to-train
extrinsics в документации нет; параметры условного виртуального поезда не
подменяют реальные физические размеры. Ghost candidates остаются диагностикой.

## Порядок воспроизведения

Python 3.12+, существующие NumPy/SciPy/laspy, Matplotlib/Pillow. BLAS/OMP/MKL=1.
Команды запускаются из корня проекта через `python -B` с путём к `src/` этапа.

1. `audit.py`, `build_weighted_adapter.py`, `tests/test_causality.py -v`.
2. `study.py development`, затем `assess.py development`.
3. `expand_development.py` — подтверждение на всех 108 development стартах.
4. `prepare_challengers.py` и соответствующие study/assess варианты.
5. `select_development.py`, `prepare_final_development.py`, study/assess challengers.
6. `select_development.py --lock` — исследовательская фиксация до held-out.
7. `run_heldout.py 10` — отдельные процессы prediction → evaluation → diagnostics.
8. `overlap_detail.py`, `collect.py`, `supplement.py`, `common_anchor_quality.py`, `choose_examples.py`.
9. `render.py`, `render.py --animations`, `export_las.py`, `review_suspects.py`.
10. `benchmark.py` — последовательно, без других тяжёлых заданий.
11. `report.py`, `verify.py` — отчёт, галерея и итоговые SHA-манифесты.

Существующий результат использует cache completion markers. Исследовательский
lock нельзя перезаписывать после контрольного прогона. Новое изменение detector
требует нового этапа, а не правки этой конфигурации.

## LAS

- `*_fused_input_map.las`: реально использованные исходные точки с provenance.
- `*_prediction_overlay.las`: то же облако с Extra Bytes `pred_cr`, `pred_step`,
  `pred_confidence`.
- `*_current_only_comparison.las`: полный T и baseline prediction.

Исходный `classification` и все первоначальные поля сохраняются. Для окраски
результата следует использовать `pred_cr`, а не прежнюю разметку. `pred_step=0`
означает seed; `65535` — точка не принята. `source_frame_delta=0` для T, −1 для
T−1 и так далее. XYZ всех трёх файлов — исходная зарегистрированная карта заезда.
Каждый экспорт перечитывается и сравнивается с исходными полями побитно.

## Ограничения оценки

Future GT не является геодезической истиной. Соседние старты зависимы. Почти
неподвижные кадры не проходят frozen ≥50 см T→T+1 bootstrap, поэтому этот опыт
не проверяет накопление на остановке как полноценный стартующий detector.

Fused bootstrap содержит past seed points, а неизменный STEP5 near-reference —
только разметку T. Его низкий near reach может быть артефактом разреженного GT;
этот diagnostic не смешивается с главным current-only bootstrap сравнением.

Скорость измеряется отдельным свежим процессом: application cache bypassed и
cached fused cloud. Кэш ОС не сбрасывается, cold disk latency не заявляется.
