"""Virtual robot over ROS 2: publish a raw episode's streams as live topics in real time.

Header stamps carry the recorded *device* clock and messages are released on the original
host-clock schedule, so a recorder on the other side sees the same multi-rate, offset,
jittery streams a physical rig would produce. Used for end-to-end tests without hardware.
"""

from __future__ import annotations

import heapq
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
from rosidl_runtime_py.utilities import get_message
from std_srvs.srv import Trigger

from robodata.config import load_robot
from robodata.io.raw_episode import RawEpisode
from robodata.io.video import iter_frames
from robodata_ros.conversions import numpy_to_msg, set_stamp


def _stream_events(name, t_host, t_dev, vals):
    # a function (not a generator expression in the caller's loop) binds this stream's arrays;
    # genexps would late-bind to the loop's last stream
    for i, v in enumerate(vals):
        yield int(t_host[i]), name, int(t_dev[i]), v


class ReplayNode(Node):
    def __init__(self):
        super().__init__("robodata_replay")
        self.episode = self.declare_parameter("episode", "").value
        self.speed = float(self.declare_parameter("speed", 1.0).value)
        self.start_delay = float(self.declare_parameter("start_delay_s", 2.0).value)
        self.recorder = self.declare_parameter("recorder", "/robodata_recorder").value
        self.ep = RawEpisode(self.episode)
        self.robot = load_robot(self.ep.meta.robot_id)
        reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST, depth=2000)
        self.pubs = {}
        for name in self.ep.streams:
            cfg = self.robot.streams[name]
            if cfg.topic:
                self.pubs[name] = (self.create_publisher(get_message(cfg.msg_type), cfg.topic,
                                                         qos_profile_sensor_data if cfg.kind in ("rgb", "depth") else reliable),
                                   get_message(cfg.msg_type), cfg)
        self.start_cli = self.create_client(Trigger, f"{self.recorder}/start")
        self.stop_cli = self.create_client(Trigger, f"{self.recorder}/stop")

    def _events(self):
        """Merge all streams into one host-time ordered event stream (lazy video decoding)."""
        heads = []
        for name, (_, _, cfg) in self.pubs.items():
            t_host, t_dev = self.ep.times(name, "t_host"), self.ep.times(name, "t_device")
            if cfg.kind == "rgb":
                vals = iter_frames(self.ep.video_path(name))
            elif cfg.kind == "depth":
                arr = self.ep.root[name]["data"]
                vals = (np.asarray(arr[i]) for i in range(arr.shape[0]))
            else:
                vals = iter(self.ep.data(name))
            heads.append(_stream_events(name, t_host, t_dev, vals))
        return heapq.merge(*heads, key=lambda e: e[0])

    def _call(self, cli):
        if not cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error(f"service {cli.srv_name} unavailable")
            return None
        fut = cli.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, fut, timeout_sec=30.0)
        return fut.result()

    def run(self):
        time.sleep(self.start_delay)  # let subscriptions match
        r = self._call(self.start_cli)
        self.get_logger().info(f"recorder start: {r.message if r else 'n/a'}")
        t0_ep = None
        t0_wall = time.monotonic()
        n = 0
        for t_host, name, t_dev, val in self._events():
            if t0_ep is None:
                t0_ep = t_host
            due = t0_wall + (t_host - t0_ep) / 1e9 / self.speed
            delay = due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            pub, cls, cfg = self.pubs[name]
            msg = numpy_to_msg(val, cfg.kind, cls, cfg.names)
            # Device clock -> header stamp, shifted onto "now" so latencies look live
            set_stamp(msg.header, int(t_dev - t0_ep + self._wall_epoch_ns(t0_wall)))
            pub.publish(msg)
            n += 1
        time.sleep(1.0)
        r = self._call(self.stop_cli)
        self.get_logger().info(f"published {n} messages; recorder stop: {r.message if r else 'n/a'}")

    def _wall_epoch_ns(self, t0_wall: float) -> int:
        if not hasattr(self, "_epoch"):
            self._epoch = self.get_clock().now().nanoseconds - int((time.monotonic() - t0_wall) * 1e9)
        return self._epoch


def main():
    rclpy.init()
    node = ReplayNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
