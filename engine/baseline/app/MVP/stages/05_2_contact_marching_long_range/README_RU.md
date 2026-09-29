# STEP 5.2 — исследование дальнего contact-rail marching

Самостоятельный исследовательский этап. Production freeze не создаётся. Код и результаты STEP5 и STEP5.1 используются только для чтения. Исходные LAS не изменяются.

Результаты находятся в `C:/Users/heinrich.wirth/deptrans/results_contact_marching_long_range`. Основные документы: `report.html`, `REPORT_LONG_RANGE.md`, `gallery.html`, `research_freeze.json`, `VERIFICATION.json`, `MANIFEST.json`. Полное исходное задание — `REQUEST_RU.txt`.

## Данные и причинность

Использован размеченный набор ANNOTATED из зафиксированного STEP5. Девять development-папок: 108 стартов; предварительный экран: 36 стартов. Два ранее изучавшихся тестовых прогона: 1422 старта. Это **reused benchmark, not fresh external validation**. Отсутствующий bootstrap и непригодный GT сохраняются в таблицах, но не смешиваются с оцениваемой выборкой.

Inference читает облако T и доступную историю. `CausalSource` и `RoiSource` запрещают чтение будущего кадра до дискового доступа. Очищенная pose T+1 разрешена только для исходного направления frozen bootstrap. Labels не передаются детектору. Каждый прогноз записывается и хешируется до открытия будущего GT оценщиком. Параметры и SHA inference зафиксированы перед финальным benchmark; изменение конфигурации по benchmark запрещено.

## Алгоритм

`long_tracker.py` выполняет пространственный marching с локальным профильным matcher, собственным recent-track frame, перебором конкурентов и сохранением состояний CONFIRMED/TENTATIVE/GAP/STOP. `roi_source.py` подаёт исторические точки только в новую область поиска. В режиме A геометрия строится по T; в режиме B принятые исторические наблюдения могут участвовать в следующем tangent. Old overlap остаётся T. SF сначала проверяет обычный T-step; успешный шаг не заменяется историей.

Проверяются 107 исходных конфигураций и 11 комбинаций/дополнительных ablation. Полное описание каждого варианта — `configs.json`, фактические числа срабатываний механизмов — `audit/ablation_mechanisms.csv`. Селекция development использует равный вес прогонов, штрафы за потери/ошибки структуры и leave-one-run-out; прежний heldout не используется для параметров. GRAPH — ограниченный beam по пространственным гипотезам, не глобальный оптимизатор.

Нет синтетических CR observations. Модельные anchors и GAP существуют отдельно от point records. Подтверждение алгоритмом не означает безошибочность относительно GT.

## Форматы

Каждый prediction содержит `prediction.json` (steps, hypotheses, причины, inputs), `points.npz`, `provenance.npz`, `evidence.npz`, `PREDICTION_COMPLETE.json`. Provenance сохраняет `(source_frame, source_row, source_point_index)`. Исходные и трансформированные исследовательские координаты не следует смешивать: inference XYZ находятся в системе T, LAS exports сохраняют исходные координаты зарегистрированного набора.

LAS на каждый выбранный случай: `input_T.las`, `fused_evidence.las`, `prediction_confirmed.las`, `prediction_tentative.las`, `overlay.las`. Сохранены оригинальные измерительные поля и classification. Дополнительный `state`: 0 не выбран, 1 tentative, 2 confirmed; `step=65535` означает не выбран. `source_age` — возраст источника. Confidence и три scores — эвристики, не вероятности. GAP не порождает точек. Пустой tentative LAS допустим. Каждый экспорт обратно прочитан и побитно сверен по исходным полям.

## Метрики и ограничения

Прежний evaluator STEP5 не изменён. Дополнительно сохраняются строгая непрерывность с остановкой на GAP/TENTATIVE и поперечная ошибка без продольного компонента. Дальность поиска, максимальные реальные подтверждённые observations и непрерывная корректная дальность показаны отдельно. Ошибки confirmed и tentative >10/20/50/100 см считаются раздельно.

Observability использует будущий GT только после inference и является диагностическим proxy: 5-см/10-см tube может включать соседнюю структуру либо исключить настоящий CR из-за ошибки GT/регистрации. Offline oracle выбора истории и мягкого overlap не являются deployable методами. Интенсивность, ring и returns изучаются отдельно; общую физическую калибровку датчика этот этап не выполняет.

## Воспроизведение

Python runtime: `C:/Users/heinrich.wirth/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe`. Запускать с `-B` из корня проекта. Зависимости используют уже существующие `.runtime` и `results_rail2d/_packages`; повторная установка не нужна. Пути задаются в `src/long_common.py`.

1. `prepare.py` создаёт протокол/когорты и первоначальные контрольные суммы. Повторно поверх существующего исследования не запускать.
2. `infer.py a --workers 8`, `assess_long.py phase_a_screen --workers 8`, `select_long.py a`.
3. `infer.py b --workers 8`, `assess_long.py phase_b_full_dev --workers 8`, `select_long.py b`.
4. `infer.py d --workers 8`, `assess_long.py phase_d_combinations --workers 8`, `ablation_detail.py`.
5. Проверить development и выполнить однократный `select_long.py freeze`. Существующий freeze не перезаписывается.
6. `infer.py e --workers 8`, затем `assess_long.py phase_e_benchmark --workers 8`.
7. `observability.py phase_e_benchmark --workers 6`, `oracle_threshold.py phase_e_benchmark --workers 6`, `collect_long.py phase_e_benchmark`, `fusion_diagnostic.py`.
8. `artifact_data.py`, `render_long.py --workers 4`, `export_long_las.py --workers 4`, `render_observability.py`, `supplement_long.py`, `sampling_audit.py`, `render_seed_audit.py`.
9. Когда тяжёлые параллельные задания окончены: `benchmark_long.py`. Он измеряет финальные методы в свежих процессах на одинаковых 12 starts и отдельно whole F8/ROI F8/F16 на 12 development-starts.
10. `report_long.py`, `verify_media.py`, `verify_long.py inputs`, `verify_long.py final`, `verify_long.py manifest`.

Regression suite: `python -B -m unittest discover -s MVP/stages/05_2_contact_marching_long_range/tests -v`. Проверяются причинность, удаление labels/future metadata, точный provenance, отсутствие искусственных GAP-points, исключение tentative/history из T-геометрии, selective fallback, запрет немедленного подтверждения одной точки и границы overlap hysteresis.

Предварительные ошибки исследовательского прототипа и их исправления документированы в `audit`: порядок source rows для PCA и ограничение слабого old-overlap кандидата. Затронутые development-прогоны пересчитаны до research freeze. Предварительные результаты исключены из итоговых таблиц. После freeze inference не изменяется.

HTML проверяется статически по ссылкам и синтаксису JavaScript; научные рисунки проверяются отдельно как PNG. Браузерная интерактивная проверка в этом этапе не выполнялась.

Внимание к непрерывности: accepted long-window может содержать внутренний пропуск, даже без отдельного состояния GAP. `100m_cases/*_sampling.json` отдельно сохраняет число реальных уникальных позиций по фиксированным 4-метровым диапазонам и максимальный интервал между наблюдениями по дальности. У C5 / roundT_doubleT / frame_000023.las дальние наблюдения достигают 119,41 м и проходят оконные 3D 10-см / transverse 5-см метрики, однако есть один пустой 4-метровый диапазон. Это не доказательство полностью наблюдаемых непрерывных 119 м. Устойчивого варианта на 100 м между прогонами не получено.
