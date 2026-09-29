# FINAL CONTACT RAIL STEP2

PRIMARY = **F4-NU**: robust direct point-set fit, fixed +4 cm context, single anchor, no-u. Confidence: patched pre-guard runner-up bookkeeping.

Frozen dependency для следующих этапов. Импортировать read-only. Исследовательские методы, GT, калибратор, stress runners и plotting не входят в пакет. R3 остаётся отдельным историческим auxiliary в STEP3.1; автоматического переключения здесь нет.

## API

Добавьте эту папку в Python import path, затем:

```python
import numpy as np
from contact_rail_step2 import detect_contact_rail

result = detect_contact_rail(
    geometry={"uvw": np.asarray(points_uvw, dtype=np.float64)},
    running_rails=np.asarray([[left_v, left_w], [right_v, right_w]]),
)
```

`geometry["uvw"]`: конечный массив N×3 в метрах, столбцы u/v/w в той же локальной системе среза, что STEP1. u — продольная координата; v — поперечная; w — высота. Обрабатывается 0≤u≤8 м. Это не преобразование сырых XYZ LAS: подготовку среза и пару STEP1 выполняет вызывающая сторона. Две ходовые головки передаются как LEFT, RIGHT, с left_v < right_v. `running_rails=None` даёт NOT_FOUND. Допустимые дополнительные массивы сохраняют связь со строками: point_id, source_row, intensity, ring, sensor_timestamp, frame_index, azimuth_ordinal. Детектор их не использует для выбора; GT/labels и имена записей запрещены.

Выход:

| Поле | Значение |
|---|---|
| status | LEFT / RIGHT / AMBIGUOUS / NOT_FOUND; авторитетное решение frozen F4 |
| side | LEFT/RIGHT только при принятом ответе, иначе None |
| anchor_v, anchor_w | Метры, общий 2D anchor; None при отказе |
| confidence | Heuristic relative reliability score после исправления pre-guard runner-up |
| support_indices | Упорядоченный int64 массив индексов **строк переданного geometry**, не point_id LAS |
| diagnostics | selected_score, runner_up_score, score_margin, runner_up_side, runner_up_removed_by_guard и пояснения |

При AMBIGUOUS/низком старом confidence поддержка сохраняется от frozen proposal: это не принятый рельс. При отсутствии кандидата массив пуст. Исходные geometry и rails не меняются.

Confidence — **не** вероятность безопасности, истинности физического CR или метрологическая гарантия. Старые isotonic/learned калибраторы не применяются. Не подставляйте новый score обратно в старый acceptance threshold 0.04: статус уже вычислен старой логикой до confidence patch. `frozen_decision_confidence` — явно названная диагностическая величина старого решения, не новая вероятность.

Лучший distinct competitor ищется среди всех допустимых кандидатов до guard: другая сторона либо расстояние anchors >7 см. Если его нет, margin=2 допускается и помечается synthetic. При наличии реального competitor используется точная разница base scores. A_STRONG (margin≤0.3) / B_WEAK / C_NONE — только диагностические категории; 0.3 — неизменный масштаб существующей формулы. Геометрия и пороги не менялись.

## Frozen параметры и происхождение

Числа в `config.json`, канонический `canonical_template.npz` и реализация фиксированы. Минимальная упаковка исключает неиспользуемые исследовательские ветки; исходные значения fit, prior, ROI, loss, score и decision остаются прежними. Формула confidence и численные параметры оптимизатора находятся в `detector.py` и покрыты Code SHA.

Проверено: Python 3.12.14, NumPy 2.5.3, SciPy 1.18.1, Windows. Версии зависимостей записаны в requirements.txt. Изменение платформы/библиотек требует воспроизведения регрессии; побитовое совпадение на другой численной среде не обещается.

## Проверка

```text
python -B -m unittest discover -s tests -v
```

Тесты содержат синтетические границы bookkeeping и два реальных geometry-only fixture без разметки. expected.json содержит только frozen API outputs и SHA поддержки. Полный аудит 101093 входов и отчёт находятся рядом в `../03_3_contact_confidence_fix/results_contact_confidence_fix/`.

`VERSION`, `freeze.json`, `MANIFEST.json` фиксируют версию, config/code/template SHA и все файлы пакета. Никакие файлы предыдущих этапов не нужны для inference.

Открывать STEP2 снова только при новом реальном LiDAR, изменении геометрии CR, новом типе инфраструктуры либо воспроизводимом production failure. Повторный tuning по тем же failures не предусмотрен.
