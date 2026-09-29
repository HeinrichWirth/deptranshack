"""One paced calculation at a time; upload and results live in a Docker volume."""

import hashlib, importlib.util, os, sqlite3, subprocess, sys, threading, time, uuid
from pathlib import Path
from .storage import JOBS, read_json, write_json, read_log, job_path

ENGINE = Path(__file__).resolve().parents[1] / "engine"
decoder_spec = importlib.util.spec_from_file_location(
    "bag_decoder", ENGINE / "baseline/adapter/cdr_cloud.py"
)
decoder = importlib.util.module_from_spec(decoder_spec)
decoder_spec.loader.exec_module(decoder)


def inspect_bag(path):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        topics = db.execute("SELECT id,name,type,serialization_format FROM topics").fetchall()
        clouds = []
        for topic_id, name, kind, encoding in topics:
            if kind != "sensor_msgs/msg/PointCloud2":
                continue
            if encoding != "cdr":
                raise ValueError("Поддерживается сериализация CDR PointCloud2")
            count, first, last = db.execute(
                "SELECT count(*),min(timestamp),max(timestamp) FROM messages WHERE topic_id=?",
                (topic_id,),
            ).fetchone()
            if count:
                blob = db.execute(
                    "SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp,id LIMIT 1",
                    (topic_id,),
                ).fetchone()[0]
                header, points = decoder.decode(blob)
                if not {"x", "y", "z", "intensity"} <= set(points.dtype.names):
                    raise ValueError(f"Топик {name}: нужны поля x, y, z, intensity")
                if len(points) > 400000:
                    raise ValueError(f"Топик {name}: больше 400 000 точек в первом кадре")
                clouds.append(
                    dict(id=topic_id, name=name, count=count, duration_s=(last - first) / 1e9)
                )
        if not clouds:
            raise ValueError("В DB3 нет сообщений sensor_msgs/msg/PointCloud2")
    return clouds


def upload(stream, length, name):
    if not 0 < length <= 50 * 1024**3:
        raise ValueError("Размер файла должен быть от 1 байта до 50 ГБ")
    if not name.lower().endswith(".db3"):
        raise ValueError("Выберите распакованный ROS 2 файл .db3")
    job_id = uuid.uuid4().hex
    path = JOBS / job_id
    path.mkdir()
    state = dict(id=job_id, name=Path(name).name, status="UPLOADING", created=time.time())
    write_json(path / "job.json", state)
    digest = hashlib.sha256()
    remaining = length
    try:
        with (path / "input.db3").open("wb") as output:
            while remaining:
                chunk = stream.read(min(4 * 1024 * 1024, remaining))
                if not chunk:
                    raise ValueError("Загрузка прервалась")
                output.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
        state.update(
            status="READY",
            topics=inspect_bag(path / "input.db3"),
            sha256=digest.hexdigest(),
            bytes=length,
        )
    except Exception as error:
        (path / "input.db3").unlink(missing_ok=True)
        state.update(status="FAILED", error=str(error))
        write_json(path / "job.json", state)
        raise
    write_json(path / "job.json", state)
    return state


class JobManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.running = None
        self.process = None
        for path in JOBS.glob("*/job.json"):
            state = read_json(path)
            if state["status"] in ("RUNNING", "PREPARING", "UPLOADING"):
                state.update(
                    status="INTERRUPTED",
                    error="Приложение было остановлено. Завершённые результаты сохранены.",
                )
                for temporary in ("input.db3", "selected.db3"):
                    (path.parent / temporary).unlink(missing_ok=True)
                state["temporary_input_removed"] = True
                write_json(path, state)

    def start(self, job_id, topic_id):
        with self.lock:
            if self.running:
                raise ValueError("Один расчёт уже идёт. Дождитесь завершения или остановите его.")
            path = job_path(job_id)
            state = read_json(path / "job.json")
            if state["status"] != "READY":
                raise ValueError(
                    "Этот файл уже запускался. Для повторного расчёта загрузите его ещё раз."
                )
            topic = next((t for t in state["topics"] if t["id"] == topic_id), None)
            if topic is None:
                raise ValueError("Выберите топик облака точек")
            self.running = job_id
            state.update(status="PREPARING", topic=topic, started=time.time())
            write_json(path / "job.json", state)
            threading.Thread(target=self._run, args=(path, state), daemon=True).start()

    def _run(self, path, state):
        try:
            source = path / "input.db3"
            with sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True) as db:
                topic_count = db.execute("SELECT count(*) FROM topics").fetchone()[0]
            if topic_count != 1:
                # Transport adapter: preserve message IDs, timestamps and exact CDR bytes.
                selected = path / "selected.db3"
                with sqlite3.connect(selected) as db:
                    db.execute("ATTACH DATABASE ? AS original", (str(source),))
                    db.execute(
                        "CREATE TABLE topics AS SELECT * FROM original.topics WHERE id=?",
                        (state["topic"]["id"],),
                    )
                    db.execute(
                        "CREATE TABLE messages AS SELECT * FROM original.messages WHERE topic_id=?",
                        (state["topic"]["id"],),
                    )
                    db.execute("CREATE INDEX timestamp_idx ON messages(timestamp)")
                source = selected
            state["status"] = "RUNNING"
            write_json(path / "job.json", state)
            env = dict(
                os.environ, PYTHONPATH=str(ENGINE / "native") + os.pathsep + str(ENGINE / "runtime")
            )
            command = [
                sys.executable,
                "-B",
                "-u",
                str(ENGINE / "runtime/replay.py"),
                "--db",
                str(source),
                "--out",
                str(path / "result"),
                "--count",
                str(state["topic"]["count"]),
            ]
            with (path / "worker.log").open("wb") as log:
                self.process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True
                )
                code = self.process.wait()
            if code:
                raise RuntimeError(f"Расчёт остановился с кодом {code}. Подробности в журнале.")
            if not read_json(path / "result/COMPLETE.json", {}).get("pass_"):
                raise RuntimeError("Расчёт не записал отметку успешного завершения")
            state.update(status="COMPLETE", finished=time.time())
        except Exception as error:
            state.update(status="FAILED", error=str(error), finished=time.time())
        finally:
            # Input is temporary. Saved batches retain exact point provenance for review.
            (path / "input.db3").unlink(missing_ok=True)
            (path / "selected.db3").unlink(missing_ok=True)
            state["temporary_input_removed"] = True
            write_json(path / "job.json", state)
            with self.lock:
                self.running = None
                self.process = None

    def status(self, job_id):
        path = job_path(job_id)
        state = read_json(path / "job.json")
        inputs = read_log(path / "result/input.jsonl")
        geometry = read_log(path / "result/geometry.jsonl")
        finished = [r for r in geometry if r["event"] == "GEOMETRY_COMPLETE"]
        state.update(
            received=sum(r["event"] == "INPUT" for r in inputs),
            processed=len(finished),
            latest=finished[-1] if finished else None,
        )
        state["frames"] = [r["source_frame"] for r in finished]
        return state
