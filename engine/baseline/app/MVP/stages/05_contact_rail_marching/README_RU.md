# STEP 5 — продолжение контактного рельса по одному облаку

Исследовательский этап. Не является production release.

Данные: `output/LAS_FRAMES_CLASSIFIED_20260926/ANNOTATED`.
Это исходная размеченная папка с двумя контрольными заездами из задания.
Последний автоматический проход не содержит этих заездов и в основном опыте не используется.

Результаты: `results_contact_marching/` в корне проекта. Полный отчёт:
`REPORT_CONTACT_MARCHING.md`, интерактивная галерея `gallery.html`.

## Изоляция

`run_study.py` читает только XYZ текущего LAS. В bootstrap передаются только две
санитизированные позы T и T+1. Frozen STEP1 и FINAL STEP2 импортируются read-only.
После bootstrap ходовые рельсы не передаются в продолжение. `tracker.py` и
`geometry.py` не имеют файлового доступа и не импортируют evaluation.

Сохранён порог надёжного движения 50 см, но ждать T+2 и следующие позы запрещено.
Если T+1 не даёт 50 см, старт недоступен. Это явно отличается от старого адаптера,
который ждал достаточного движения. Sensor yaw не заменяет направление движения.

`evaluation.py` читает разметку и будущие облака только после сохранённого
`PREDICTION_COMPLETE.json`. `oracles.py` — отдельные диагностические сравнения,
которые не входят в production inference. Ближний seed проверяется по классу 2 T;
новые точки — по fused class 2 от T+2. Регистрация и разметка не являются идеальным GT.

## Запуск

Python 3.12+, NumPy, SciPy, laspy; Matplotlib и Pillow для отчёта.
В данном workspace используются существующие `.runtime` и библиотеки графиков.
Все команды выполняются из корня проекта с `python -B`.

1. `src/audit.py` — инвентаризация, проверка frozen dependencies.
2. `tests/test_isolation.py -v` — геометрия и изоляция.
3. `src/run_study.py development` — равномерная development выборка.
4. `src/evaluate_study.py development` — будущая разметка после prediction.
5. `src/calibrate.py`, `src/confirm.py` — компактный sweep и парное сравнение.
6. `src/freeze.py` — исследовательская фиксация до held-out.
7. `src/run_study.py heldout`, `src/evaluate_study.py heldout` — полный контрольный набор.
8. `src/oracles.py development`, `src/oracles.py heldout` — диагностические сравнения.
   `src/oracles.py heldout --template-seed` — дополнительная проверка seed из
   class2-точек, согласованных с тем же шаблоном; выполняется после основного прогона.
9. `src/warm_observability.py`, `src/analyze.py` — наблюдаемость и сводные показатели.
10. `src/investigate_conflicts.py`, `src/audit_oracle_seed.py`,
    `src/finalize_diagnostics.py`, `src/hypotheses.py` — проверка конфликтов GT,
    состава oracle seed, окончательные diagnostic packets и геометрические гипотезы.
11. `src/window_initialization_diagnostic.py` — отдельная development-only
    проверка инициализации короткого окна, без изменения frozen конфигурации.
12. `src/render.py terminal`, `src/render.py representatives` — срезы и примеры.
13. `src/export_artifacts.py`, `src/export_points.py` — LAS и таблица всех точек.
14. `src/benchmark.py` — последовательный замер без других тяжёлых заданий.
15. `src/audit_sources.py`, `src/report.py`, `src/verify.py` — целостность исходников,
    отчёт, проверка полноты и итоговые манифесты SHA.

`freeze.json` фиксирует научный протокол, выбранные параметры и SHA кода/шаблона.
Он не объявляет marching production-ready. Повторный запуск пользуется результатами;
для другого исследования нужен новый каталог, а не изменение frozen конфигурации.

## LAS

`*_overlay.las`: полный исходный LAS в координатах зарегистрированной карты своего
заезда. Класс, порядок, XYZ и все исходные поля сохранены. Добавлены Extra Bytes
`pred_cr`, `pred_step`, `pred_confidence`, `pred_local_s`, `pred_template_residual`,
`pred_anchor_error_eval`. Последнее поле предназначено только для оценки.

`*_predicted_cr_map.las`: только настоящие найденные точки в исходной системе карты.
`*_predicted_cr_sensorT.las`: те же точки в системе LiDAR T с точностью квантования LAS.
Синтетические опоры/кривые не добавляются как измеренные точки.

Полный размеченный class2-профиль шире confirmed template support. Поэтому raw
ORACLE_SEED не является идеальной верхней границей качества и часто отвергается
проверкой overlap. Отдельный TEMPLATE8 использует только реальные class2-точки
в пределах прежнего допуска 2 см до canonical template. Обе серии сохранены;
основной frozen config и held-out predictions после этого не менялись.

`pred_step=0` — bootstrap; `65535` — точка не принята. NaN в дополнительных полях
означает отсутствие оценки. `pred_local_s` — реконструированный продольный параметр;
это не Euclidean range. Оба значения отдельно находятся в CSV точек.

Старые этапы, исходные LAS, прежний автоматический проход и viewer не изменяются.
