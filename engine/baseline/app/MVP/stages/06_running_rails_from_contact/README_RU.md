# STEP 6 — ходовые рельсы из кривой контактного рельса

Отдельное исследование геометрии и неопределённости. Предыдущие stages не изменяются.

Основной CR-вход — **зафиксированный C4 STEP5.2**. Читаются только сохранённые результаты; C4 не переобучается, не перенастраивается и не запускается с новыми параметрами. Код зависимостей, конфигурация и маркеры входных результатов защищены SHA-256 в `dependency_lock.json` и `c4_input_index.json`.

Результаты: `C:/Users/heinrich.wirth/deptrans/results_running_rails_from_contact/`.

Полное задание находится в `REQUEST_RU.txt`. Основной режим: калибровка двух головок и CR по первым 8 м текущего кадра, затем только CR и геометрическая модель с коридорами неопределённости. После seed class1 и сырые точки ходовых рельсов не поступают в основной predictor. Future GT открывается после сохранения rail prediction. Oracle CR/угол/offsets — отдельные диагностические режимы.

STEP5.3 не является зависимостью. Существующие статистики detector anchors используются как контекст, а не физическая нормативная колея и не независимый эталон.

Запускать Python с `-B`, чтобы не создавать bytecode в замороженных зависимостях. STEP6 завершён как исследование; конфигурации зафиксированы в `results_running_rails_from_contact/research_freeze.json` до итогового benchmark. Это не production release. Не запускать заново development/prepare_stage поверх завершённого freeze.

## Результаты и ограничения

108 development-стартов и 1422 старта reused research benchmark. Лучший development-метод — `B4_SPLINE_50__O1`, C4 STEP5.2 неизменён. На benchmark far p95: 10,12 / 13,41 / 19,16 см для 30–50 / 50–75 / 75–100 м. Seed доступен в 865 из 1422 стартов; 553 не имеют seed C4, 4 — достаточной начальной rail pair.

Вердикт C: CR-only недостаточно без наблюдений roll для точных дальних линий. У адаптера STEP6 выявлен также natural-spline overshoot до 23,6 см при неравномерных C4 anchors; он не исправлялся после benchmark и оставлен в отчёте. Есть три seed-only случая с почти дублирующим endpoint на машинной точности. Результат исследовательский, не готовый production adapter.

Отчёт: `results_running_rails_from_contact/REPORT_RUNNING_FROM_CR.html` и `.md`; галерея: `gallery.html`. 54 примера, 162 PNG, 30 наборов LAS (90 файлов). Оригинальные LAS не изменены. Диагностические oracle-данные запрещены для inference.

## Скрипты и порядок

1. `src/prepare_stage.py` — первоначальный dependency lock и протокол (уже выполнен).
2. `src/develop.py` — A/B screening, future GT после prediction, oracle-контроли и LORO selection.
3. `src/development_stress.py`, `src/extended_oracles.py phase_b`, `src/geometry_audit.py phase_b`, `python -B -m unittest discover -s MVP/stages/06_running_rails_from_contact/tests -v` — проверки до freeze.
4. `src/select_methods.py freeze` — однократный research freeze (уже выполнен).
5. `src/final_pipeline.py` — один полный frozen benchmark, оценка, oracle-контроли, аудит и последовательный speed/RAM benchmark (уже выполнен).
6. `src/artifact_cases.py`, `src/export_las.py`, `src/posthoc_diagnostics.py`, `src/build_gallery.py`, `src/build_report.py` — только артефакты и анализ сохранённых решений, без перенастройки.
7. `src/verify_delivery.py --manifest` — проверка SHA, causal boundary, локальных ссылок и manifest.

Не повторять benchmark для подбора параметров. Новую интерполяцию или observation-aided вариант исследовать отдельной версией и с новым заранее зафиксированным протоколом. C4 остаётся read-only.
