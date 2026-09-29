"""Attach completed independent verification to the post-inference report."""
import json,hashlib,html
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results_ros2_replay_v1/full'
load=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
s=load(OUT/'SUMMARY.json');v=load(OUT/'VERIFICATION.json');e=load(OUT/'EXPORTS.json');r=load(OUT/'registration_reproduction.json')
assert v['pass_']
section=f'''
## Заключительная проверка и конкретные трудные случаи

**PASS:** все 1510 сообщений учтены; границы облаков/поз проверены для {v['causal_solves_checked']} solve. Проверены hashes всех сохранённых решений и {v['frozen_files_checked']} доступных на host файлов frozen release; Docker дополнительно проверил все 60 runtime files до и после прогона. На {len(v['map_frames_checked'])} полных кадрах LAS проверены исходные point_index, message_id, intensity и XYZ с допуском ровно на квантование LAS.

Сшивка воспроизвела прежний метод: после inference новые {r['compared_accepted_frames']} принятых поз сопоставлены со старыми причинными оценками. Максимальное расхождение переноса — {r['translation_difference_max_m']:.3g} м. Это проверка воспроизведения алгоритма, а не точности относительно реальной траектории. Все 14 старых интерполированных поз отсутствуют в новом принятом потоке.

Отказы bootstrap: T=0–3 — ещё нет 0.5 м базиса. Отказы STEP2: T=120–121, 813–814, 1263–1267 — нет принятого contact candidate. Другие решения имеют AVAILABLE, но различный подтверждённый горизонт. В частности, T=116 и T=118 дают только около 2.91 и 2.22 м C4 observed support и останавливаются с NO_POINTS; отображаемые примерно 8 м seed-supported rails не являются доказательством наблюдения CR на всех 8 м. T=1303 достигает 103.13 м; LOCAL_FRAME_DEGENERATE там относится к остановке дальнейшего продолжения, а не к отказу уже построенных 103 м.

Причинная сшивка недоступна на T=219–231 и T=369. В map export исключены ровно {s['eligible_points']-e['map_points']:,} ненулевых исходных точек этих кадров. Они остаются в исходной записи и показаны в примерах. Карта содержит {e['map_points']:,} точек, размер {e['map_bytes']/1e9:.2f} GB. Цветной модельный LAS занимает {e['geometry_bytes']/1e6:.1f} MB. Большой map export занял {e['map_export_seconds']:.1f} с уже после измеряемого прогона.

Итог для движения поезда: каузальность соблюдена, но полный тракт с исходной Python-сшивкой и чтением SQLite через Windows bind mount не укладывается в 10 Hz. Медиана самой сшивки {s['timing']['registration_ms']['median']:.1f} мс уже превышает 100 мс, даже если убрать файловое чтение. В этой задаче ядро и метод сшивки не оптимизировались.
'''
path=OUT/'REPORT.md';text=path.read_text(encoding='utf-8').split('\n## Заключительная проверка и конкретные трудные случаи')[0];path.write_text(text+section,encoding='utf-8')
path=OUT/'gallery.html';page=path.read_text(encoding='utf-8')
extra=f'''<section id="verification"><h2>Проверка завершена</h2><p>1510 сообщений, {v['causal_solves_checked']} проверенных причинных solve. Новая сшивка воспроизводит старые принятые позы с максимальным расхождением {r['translation_difference_max_m']:.2g} м; будущая интерполяция исключена.</p><p>4 bootstrap-отказа на старте; 9 отказов STEP2: T=120–121, 813–814, 1263–1267. 14 отсутствующих поз: T=219–231 и 369.</p><p><b>Сшитая карта:</b> {e['map_points']:,} точек / {e['map_bytes']/1e9:.2f} GB; цветной прогноз {e['geometry_bytes']/1e6:.1f} MB. Для 3D откройте stitched_cloud.las и predicted_geometry.las в одной системе координат.</p><p>Короткие случаи T=116/118 имеют лишь 2–3 м подтверждённой C4-поддержки. Нарисованная ближняя модель до 8 м не означает подтверждённое наблюдение CR на всём её протяжении.</p><p><a href="VERIFICATION.json">Проверки</a> · <a href="registration_reproduction.json">Воспроизведение сшивки</a></p></section>'''
if 'id="verification"' not in page:page=page.replace('<img src="timeline.png">',extra+'<img src="timeline.png">')
path.write_text(page,encoding='utf-8')
files={str(p.relative_to(ROOT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*') if p.is_file()}
manifest=dict(adapter_sources=files,source_db='datas/test_synthetic/cloud_with_fake_obj/cloud_with_fake_obj_0.db3',source_message_payload_sha256=s['source_message_payload_sha256'],frozen_version='geometry_pipeline_v1.0.0',runtime_unchanged=True,report_sha256=hashlib.sha256((OUT/'REPORT.md').read_bytes()).hexdigest(),gallery_sha256=hashlib.sha256((OUT/'gallery.html').read_bytes()).hexdigest(),verification=v)
(OUT/'RUN_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print('FINAL_REPORT_COMPLETE')
