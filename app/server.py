"""Loopback-published Docker web app. No shell commands are accepted over HTTP."""

import csv, io, json, os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs, unquote
from .jobs import JobManager, upload
from .storage import JOBS, job_path, read_json
from .results import frames, objects, binary_frame, slice_frame

WEB = Path(__file__).resolve().parents[1] / "web"
manager = JobManager()


class Handler(BaseHTTPRequestHandler):
    def send(self, data, mime="application/json; charset=utf-8", code=200):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        try:
            route = urlsplit(self.path)
            query = parse_qs(route.query)
            get = lambda key, default="": query.get(key, [default])[0]
            job = get("job", get("run"))
            source = int(get("source", "0"))
            if route.path == "/api/health":
                return self.send(dict(ok=True))
            if route.path == "/api/jobs":
                return self.send(
                    [
                        read_json(p)
                        for p in sorted(
                            JOBS.glob("*/job.json"), key=lambda p: p.stat().st_mtime, reverse=True
                        )
                    ]
                )
            if route.path == "/api/status":
                return self.send(manager.status(job))
            if route.path == "/api/objects":
                return self.send(objects(job))
            if route.path == "/api/slice":
                return self.send(slice_frame(job, source, float(get("distance", "8"))))
            if route.path == "/api/review-cases":
                return self.send([])
            if route.path == "/api/predicted-train/runs":
                rows = frames(job)
                return self.send(
                    dict(
                        runs=[
                            dict(
                                id=job,
                                label=read_json(job_path(job) / "job.json")["name"],
                                frames=[
                                    dict(
                                        source_frame_index=r["source_frame"],
                                        name=f"Кадр {r['source_frame']}",
                                        available=(
                                            job_path(job)
                                            / "result/rail_refresh"
                                            / f"{r['source_frame']:06d}.npz"
                                        ).exists(),
                                    )
                                    for r in rows
                                ],
                            )
                        ]
                    )
                )
            if route.path == "/api/predicted-train/frame":
                rows = frames(job)
                index = int(get("index", "0"))
                return self.send(
                    binary_frame(job, rows[index]["source_frame"], int(get("new_object", "0"))),
                    "application/octet-stream",
                )
            if route.path == "/api/log":
                p = job_path(job) / "worker.log"
                return self.send(
                    (p.read_bytes()[-100000:] if p.exists() else b""), "text/plain; charset=utf-8"
                )
            if route.path == "/api/export":
                data = objects(job)
                if get("format") != "csv":
                    return self.send(data)
                stream = io.StringIO()
                writer = csv.writer(stream)
                writer.writerow(
                    [
                        "id",
                        "raw_ids",
                        "status",
                        "first_frame",
                        "last_frame",
                        "nearest_m",
                        "width_m",
                        "height_m",
                        "area_m2",
                    ]
                )
                for t in data["tracks"]:
                    writer.writerow(
                        [
                            t["id"],
                            "+".join(map(str, t["member_ids"])),
                            t["status"],
                            t["first_frame"],
                            t["last_frame"],
                            t["nearest_range_m"],
                            t["width_m"],
                            t["height_m"],
                            t["bbox_area_m2"],
                        ]
                    )
                return self.send(("\ufeff" + stream.getvalue()).encode(), "text/csv; charset=utf-8")
            name = route.path.lstrip("/") or "index.html"
            if "/" in name or name.startswith("."):
                raise ValueError("Страница не найдена")
            p = (WEB / "viewer" / name) if name.startswith("predicted_train") else WEB / name
            if not p.is_file():
                return self.send(dict(error="Страница не найдена"), code=404)
            mime = {".js": "application/javascript", ".css": "text/css", ".html": "text/html"}.get(
                p.suffix, "application/octet-stream"
            )
            return self.send(p.read_bytes(), mime + "; charset=utf-8")
        except (ValueError, KeyError, IndexError, FileNotFoundError) as error:
            self.send(dict(error=str(error)), code=400)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_POST(self):
        self.mutate()

    def do_PUT(self):
        self.mutate()

    def mutate(self):
        try:
            origin = self.headers.get("Origin")
            if origin and urlsplit(origin).netloc != self.headers.get("Host"):
                return self.send(dict(error="Запрос должен прийти из этого приложения"), code=403)
            route = urlsplit(self.path)
            query = parse_qs(route.query)
            if route.path == "/api/upload" and self.command == "PUT":
                return self.send(
                    upload(
                        self.rfile,
                        int(self.headers.get("Content-Length", "0")),
                        unquote(self.headers.get("X-Filename", "input.db3")),
                    ),
                    code=201,
                )
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 65536:
                raise ValueError("Некорректный размер запроса")
            data = json.loads(self.rfile.read(size))
            if route.path == "/api/start":
                manager.start(data["job"], int(data["topic"]))
                return self.send(dict(ok=True))
            raise ValueError("Неизвестная операция")
        except (ValueError, KeyError, OSError) as error:
            self.send(dict(error=str(error)), code=400)
        except Exception as error:
            self.send(dict(error=str(error)), code=500)

    def log_message(self, message, *args):
        if "/api/status" not in self.path and "/api/slice" not in self.path:
            super().log_message(message, *args)


if __name__ == "__main__":
    ThreadingHTTPServer(("0.0.0.0", int(os.environ.get("PORT", "8080"))), Handler).serve_forever()
