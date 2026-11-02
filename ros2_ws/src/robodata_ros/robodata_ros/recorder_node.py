"""Generic multimodal episode recorder node.

Every topic, message type, rate and stream name comes from the robot YAML
(``configs/robots/<robot>.yaml``), so the same node runs on any robot. Each message is
stored with its header stamp (sensor clock) and the node's receive time (host clock);
alignment happens offline (``robodata sync``) so it can be measured and redone.

Services (std_srvs/Trigger): ~/start, ~/stop, ~/discard
Parameters: robot, raw_root, session_id, operator, scene_id, task, success (true|false|unknown)
Publishes:  ~/recording (std_msgs/Bool), /diagnostics (per-stream rate + header-to-receive latency)
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from functools import partial

import numpy as np
import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from robodata.config import load_robot
from robodata.recorder.core import EpisodeRecorder
from robodata_ros.conversions import msg_to_numpy, stamp_ns


class RecorderNode(Node):
    def __init__(self):
        super().__init__("robodata_recorder")
        p = {k: self.declare_parameter(k, v).value for k, v in {
            "robot": "ur5_catheter", "raw_root": "/data/raw", "session_id": "", "operator": "unknown",
            "scene_id": "unknown", "task": "unspecified", "success": "unknown", "auto_start": False}.items()}
        self.robot = load_robot(p["robot"])
        self.rec = EpisodeRecorder(self.robot, p["raw_root"], session_id=p["session_id"] or None,
                                   operator=p["operator"], scene_id=p["scene_id"])
        self.stats = defaultdict(lambda: {"n": 0, "lat": deque(maxlen=300), "t": deque(maxlen=300)})
        self.lock = threading.Lock()
        for name, cfg in self.robot.streams.items():
            if not cfg.topic:
                continue
            # ~4 s of buffering per stream absorbs executor stalls (500 Hz F/T next to six camera streams);
            # one mutually-exclusive group per stream keeps each stream in order while streams run in parallel
            depth = int(max(50, 4 * cfg.rate_hz))
            if cfg.kind in ("rgb", "depth"):
                qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=depth)
            else:
                qos = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=depth)
            self.create_subscription(get_message(cfg.msg_type), cfg.topic, partial(self._on_msg, name, cfg), qos,
                                     callback_group=MutuallyExclusiveCallbackGroup())
            self.get_logger().info(f"stream {name:14s} <- {cfg.topic} [{cfg.msg_type}] {cfg.rate_hz} Hz")
        self.create_service(Trigger, "~/start", self._start)
        self.create_service(Trigger, "~/stop", self._stop)
        self.create_service(Trigger, "~/discard", self._discard)
        self.pub_rec = self.create_publisher(Bool, "~/recording", 10)
        self.pub_diag = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.create_timer(1.0, self._diagnostics)
        if p["auto_start"]:
            self._start(None, Trigger.Response())

    def _on_msg(self, name, cfg, msg, info):
        # middleware receive time (stamped by the DDS thread on arrival). Taking now() here instead would
        # add Python executor backlog (measured: up to ~0.8 s for 125 Hz streams next to 6 camera streams).
        t_host = int(info["received_timestamp"]) or self.get_clock().now().nanoseconds
        t_dev = stamp_ns(msg.header.stamp)
        try:
            data = msg_to_numpy(msg, cfg.kind, cfg.shape, cfg.names)
        except Exception as exc:  # never drop the node for one bad message; it shows up as a drop
            self.get_logger().warn(f"{name}: conversion failed: {exc}", throttle_duration_sec=5.0)
            return
        self.rec.on_message(name, t_dev, data, t_host_ns=t_host)
        st = self.stats[name]
        with self.lock:
            st["n"] += 1
            st["lat"].append((t_host - t_dev) / 1e6)
            st["t"].append(time.monotonic())

    def _param(self, k):
        return self.get_parameter(k).value

    def _start(self, req, resp):
        try:
            eid = self.rec.start(task=self._param("task"))
            resp.success, resp.message = True, eid
            self.get_logger().info(f"recording {eid}")
        except RuntimeError as exc:
            resp.success, resp.message = False, str(exc)
        self.pub_rec.publish(Bool(data=self.rec.recording))
        return resp

    def _stop(self, req, resp):
        s = self._param("success")
        try:
            path = self.rec.stop(success={"true": True, "false": False}.get(str(s).lower()))
            resp.success, resp.message = True, str(path)
            self.get_logger().info(f"saved {path}")
        except RuntimeError as exc:
            resp.success, resp.message = False, str(exc)
        self.pub_rec.publish(Bool(data=self.rec.recording))
        return resp

    def _discard(self, req, resp):
        self.rec.discard()
        resp.success, resp.message = True, "discarded"
        self.pub_rec.publish(Bool(data=False))
        return resp

    def _diagnostics(self):
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        with self.lock:
            for name, cfg in self.robot.streams.items():
                st = self.stats.get(name)
                s = DiagnosticStatus(name=f"robodata/{name}", hardware_id=cfg.topic or "")
                if not st or len(st["t"]) < 2:
                    s.level, s.message = DiagnosticStatus.WARN, "no data"
                else:
                    rate = (len(st["t"]) - 1) / max(st["t"][-1] - st["t"][0], 1e-6)
                    lat = np.asarray(st["lat"])
                    ok = abs(rate - cfg.rate_hz) / cfg.rate_hz < 0.2
                    s.level = DiagnosticStatus.OK if ok else DiagnosticStatus.WARN
                    s.message = f"{rate:.1f} Hz"
                    s.values = [KeyValue(key="rate_hz", value=f"{rate:.2f}"),
                                KeyValue(key="latency_p50_ms", value=f"{np.percentile(lat, 50):.2f}"),
                                KeyValue(key="latency_p95_ms", value=f"{np.percentile(lat, 95):.2f}"),
                                KeyValue(key="received", value=str(st["n"]))]
                arr.status.append(s)
        self.pub_diag.publish(arr)
        self.pub_rec.publish(Bool(data=self.rec.recording))


def main():
    rclpy.init()
    node = RecorderNode()
    ex = MultiThreadedExecutor(num_threads=max(4, len(node.robot.streams)))
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node.rec.recording:
            node.get_logger().warn("shutdown while recording: saving partial episode")
            node.rec.stop(success=None)
        node.destroy_node()
        rclpy.try_shutdown()
