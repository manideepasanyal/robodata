"""ros2 launch robodata_capture_cpp stamp_monitor.launch.py  (topics/types for the UR5 rig by default)"""

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([Node(
        package="robodata_capture_cpp", executable="stamp_monitor", name="robodata_stamp_monitor", output="screen",
        parameters=[{
            "topics": ["/ft_sensor/wrench", "/joint_states", "/camera_front/color/image_raw", "/camera_wrist/color/image_raw"],
            "types": ["geometry_msgs/msg/WrenchStamped", "sensor_msgs/msg/JointState", "sensor_msgs/msg/Image",
                      "sensor_msgs/msg/Image"],
            "output_dir": "/data/stamps",
        }])])
