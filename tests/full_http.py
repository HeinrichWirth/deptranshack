"""Real upload/start/status smoke test. Supply your own ROS2 DB3; no data in Git."""

import argparse, http.client, json, time, urllib.request
from pathlib import Path
from urllib.parse import urlsplit, quote


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8870")
    parser.add_argument("--out", type=Path, default=Path("local-results/http-test.json"))
    args = parser.parse_args()
    target = urlsplit(args.url)
    connection = http.client.HTTPConnection(target.hostname, target.port, timeout=600)
    connection.putrequest("PUT", "/api/upload")
    connection.putheader("Content-Length", str(args.bag.stat().st_size))
    connection.putheader("X-Filename", quote(args.bag.name))
    connection.endheaders()
    with args.bag.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            connection.send(chunk)
    response = connection.getresponse()
    job = json.loads(response.read())
    assert response.status == 201, job
    job_id = job["id"]
    print("UPLOADED", job_id, flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(job=job_id, status="UPLOADED")))
    payload = json.dumps(dict(job=job_id, topic=job["topics"][0]["id"])).encode()
    request = urllib.request.Request(
        args.url + "/api/start", data=payload, headers={"Content-Type": "application/json"}
    )
    json.load(urllib.request.urlopen(request))
    last = -1
    live_seen = False
    deadline = time.monotonic() + job["topics"][0]["duration_s"] * 3 + 120
    while time.monotonic() < deadline:
        state = json.load(urllib.request.urlopen(args.url + "/api/status?job=" + job_id))
        if state["received"] // 100 != last:
            last = state["received"] // 100
            print(state["status"], state["received"], state["processed"], flush=True)
        if state.get("latest") and not live_seen:
            source = state["latest"]["source_frame"]
            slice_result = json.load(
                urllib.request.urlopen(
                    args.url + f"/api/slice?job={job_id}&source={source}&distance=4"
                )
            )
            live_seen = True
            print("LIVE_SLICE", source, slice_result["available"], flush=True)
        if state["status"] in ("COMPLETE", "FAILED", "INTERRUPTED"):
            break
        time.sleep(1)
    assert state["status"] == "COMPLETE", state
    objects = json.load(urllib.request.urlopen(args.url + "/api/objects?job=" + job_id))
    assert state["temporary_input_removed"]
    assert state["received"] == job["topics"][0]["count"]
    assert live_seen
    report = dict(
        job=job_id,
        pass_=True,
        received=state["received"],
        processed=state["processed"],
        live_slice_seen=live_seen,
        temporary_input_removed=True,
        raw_tracks=objects["raw_tracks"],
        confirmed=sum(t["status"] == "CONFIRMED" for t in objects["tracks"]),
    )
    args.out.write_text(json.dumps(report, indent=2))
    print(report, flush=True)


if __name__ == "__main__":
    main()
