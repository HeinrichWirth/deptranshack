"""ROS 2 Humble subscriber. Exact CDR bytes, latest-only handoff, no bag access."""

import json
import os
import socket
import threading
import time
import urllib.request

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from live_protocol import MAX_CDR_BYTES, receive_json, send_json


class Bridge(Node):
    def __init__(self):
        super().__init__("rail_inspector")
        self.declare_parameter("cloud_topic", os.getenv("CLOUD_TOPIC", "/lidar_points"))
        self.declare_parameter("idle_exit_seconds", float(os.getenv("IDLE_EXIT_SECONDS", "0")))
        self.topic = self.get_parameter("cloud_topic").value
        self.idle_exit = self.get_parameter("idle_exit_seconds").value
        deadline = time.monotonic() + 60
        while True:
            try:
                self.socket = socket.create_connection((os.getenv("ENGINE_HOST", "engine"), 8871), timeout=5)
                self.socket.settimeout(70)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(1)
        send_json(self.socket, dict(protocol=1, topic=self.topic))
        self.job = receive_json(self.socket)["job"]
        self.socket.settimeout(5)
        self.pending = None
        self.condition = threading.Condition()
        self.closed = False
        self.failure = None
        self.sequence = 0
        self.replaced = self.expired = self.sent = 0
        self.last_input = None
        self.last_result = None
        self.result_url = os.getenv("WEB_URL", "http://web:8080")
        self.result_publisher = self.create_publisher(String, "/rail_inspector/result", 1)
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        self.subscription = self.create_subscription(PointCloud2, self.topic, self.cloud, qos, raw=True)
        self.sender = threading.Thread(target=self.send_loop, daemon=True)
        self.sender.start()
        self.get_logger().info(f"READY topic={self.topic} job={self.job}")

    def cloud(self, blob):
        now = time.monotonic()
        sequence = self.sequence
        self.sequence += 1
        self.last_input = now
        if len(blob) > MAX_CDR_BYTES:
            self.failure = "PointCloud2 exceeds 32 MiB transport limit"
            return
        with self.condition:
            if self.pending is not None:
                self.replaced += 1
            self.pending = (sequence, now, bytes(blob))
            self.condition.notify()

    def send_loop(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.pending is not None or self.closed)
                    if self.closed:
                        return
                    sequence, arrived, blob = self.pending
                    self.pending = None
                age = time.monotonic() - arrived
                if age > .1:
                    self.expired += 1
                    continue
                send_json(self.socket, dict(sequence=sequence, bytes=len(blob),
                    callback_to_send_ms=age*1000, callback_monotonic=arrived,
                    replaced=self.replaced, expired=self.expired))
                self.socket.sendall(blob)
                reply = receive_json(self.socket)
                if reply.get("accepted") != sequence:
                    raise RuntimeError("Unexpected receiver acknowledgement")
                self.sent += 1
        except Exception as error:
            self.failure = str(error)

    def publish_result(self):
        try:
            with urllib.request.urlopen(self.result_url + "/api/live-result?job=" + self.job, timeout=.5) as response:
                result = json.load(response)
            key = (result.get("source_frame"), result.get("job_status"), result.get("stale"),
                   result.get("snapshot_ready"), result.get("obstacle_detected"))
            if key != self.last_result:
                message = String()
                message.data = json.dumps(result, allow_nan=False)
                if not rclpy.ok():
                    return
                self.result_publisher.publish(message)
                self.last_result = key
        except (OSError, ValueError):
            pass  # No fabricated result while the first geometry is pending.

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.sender.join(6)
        self.socket.close()
        summary = f"STOP received={self.sequence} sent={self.sent} replaced={self.replaced} expired={self.expired}"
        if rclpy.ok():
            self.get_logger().info(summary)
        else:
            print(summary, flush=True)


def main():
    rclpy.init()
    node = Bridge()
    # HTTP output polling must not block PointCloud2 callbacks.
    output_stop = threading.Event()
    def output_loop():
        while not output_stop.wait(.33):
            node.publish_result()
    output_thread = threading.Thread(target=output_loop, daemon=True)
    output_thread.start()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.1)
            if node.failure:
                raise RuntimeError(node.failure)
            if node.idle_exit > 0 and node.last_input and time.monotonic() - node.last_input > node.idle_exit:
                break
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        # The worker flushes its last results after the input closes.
        time.sleep(1.5)
        node.publish_result()
        output_stop.set()
        output_thread.join(1)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
