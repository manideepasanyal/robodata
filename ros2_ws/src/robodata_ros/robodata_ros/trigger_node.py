"""Software trigger: publishes a stamped tick that camera drivers (or the sim) capture on.

For rigs without a hardware sync line (sync.mode: software_trigger). All cameras capture on
the same tick, which turns free-run phase offsets (up to half a frame) into a bounded
driver latency that ``known_latency_ms`` can remove.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Header


class TriggerNode(Node):
    def __init__(self):
        super().__init__("robodata_trigger")
        rate = float(self.declare_parameter("rate_hz", 30.0).value)
        self.frame_id = self.declare_parameter("frame_id", "trigger").value
        self.pub = self.create_publisher(Header, "/robodata/trigger", 10)
        self.seq = 0
        self.create_timer(1.0 / rate, self._tick)

    def _tick(self):
        h = Header(frame_id=f"{self.frame_id}/{self.seq}")
        h.stamp = self.get_clock().now().to_msg()
        self.pub.publish(h)
        self.seq += 1


def main():
    rclpy.init()
    node = TriggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
