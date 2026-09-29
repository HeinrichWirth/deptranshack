"""Qualification subscriber: save actual result messages received through DDS."""
import argparse
import json
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=175)
parser.add_argument("--out", default="/bag/results.jsonl")
args = parser.parse_args()
rclpy.init()
node = Node("rail_inspector_qualification")
with open(args.out, "w", buffering=1) as output:
    def received(message):
        row = json.loads(message.data)
        row["observer_wall_time"] = time.time()
        output.write(json.dumps(row) + "\n")
    subscription = node.create_subscription(String, "/rail_inspector/result", received, 10)
    end = time.monotonic() + args.seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=.1)
node.destroy_node()
rclpy.shutdown()
