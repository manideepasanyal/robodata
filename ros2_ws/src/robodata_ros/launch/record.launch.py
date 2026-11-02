"""Record episodes for one robot: ros2 launch robodata_ros record.launch.py robot:=ur5_catheter raw_root:=/data/raw"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = {"robot": "ur5_catheter", "raw_root": "/data/raw", "session_id": "", "operator": "unknown",
            "scene_id": "unknown", "task": "unspecified", "trigger_rate_hz": "30.0"}
    decl = [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()]
    return LaunchDescription(decl + [
        Node(package="robodata_ros", executable="recorder", name="robodata_recorder", output="screen",
             parameters=[{k: LaunchConfiguration(k) for k in args if k != "trigger_rate_hz"}]),
        Node(package="robodata_ros", executable="trigger", name="robodata_trigger", output="screen",
             parameters=[{"rate_hz": LaunchConfiguration("trigger_rate_hz")}]),
    ])
