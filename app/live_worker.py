"""Output-only observer for the frozen geometry worker's existing tracker."""

from pathlib import Path
import threading
from .storage import write_json


def geometry_worker(*args):
    import batch_worker
    from obstacle_tracker import Tracker

    output = Path(args[-2])
    generations = {}
    condition = threading.Condition()
    pending = [None]
    stopping = [False]
    failures = []

    def save_snapshots():
        try:
            while True:
                with condition:
                    condition.wait_for(lambda: pending[0] is not None or stopping[0])
                    if pending[0] is None:
                        return
                    snapshot, pending[0] = pending[0], None
                write_json(output / "live_tracks.json", snapshot)
        except Exception as error:
            failures.append(error)

    writer = threading.Thread(target=save_snapshots, name="live-table", daemon=True)
    writer.start()

    class ObservedTracker(Tracker):
        def update(self, source, timestamp, pose, generation, components):
            result = super().update(source, timestamp, pose, generation, components)
            generations[source] = generation
            # The live table needs measurements, not millions of point indices.
            # The frozen worker still writes the complete tracks.json on close.
            tracks = [dict(track, observations=[
                {key: value for key, value in observation.items() if key != "source_indices"}
                for observation in track["observations"]]) for track in self.export()]
            snapshot = dict(
                tracks=tracks, generations=dict(generations), snapshot_has_point_indices=False,
                source_frame=source, header_time_ns=round(timestamp * 1e9),
            )
            with condition:
                pending[0] = snapshot
                condition.notify()
            return result

    batch_worker.Tracker = ObservedTracker
    try:
        batch_worker.geometry_worker(*args)
    finally:
        with condition:
            stopping[0] = True
            condition.notify()
        writer.join(3)
        if failures:
            raise RuntimeError("Live table export failed") from failures[0]
        if writer.is_alive():
            raise RuntimeError("Live table export did not finish")
