"""End-to-end ROS 2 test without hardware: replay a raw episode as live topics and record it again.

ros2 launch robodata_ros replay_and_record.launch.py episode:=/data/raw/ur5_sim/.../ep raw_root:=/data/ros_raw
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, EmitEvent
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    episode, robot, raw_root = LaunchConfiguration("episode"), LaunchConfiguration("robot"), LaunchConfiguration("raw_root")
    recorder = Node(package="robodata_ros", executable="recorder", name="robodata_recorder", output="screen",
                    parameters=[{"robot": robot, "raw_root": raw_root, "session_id": "ros_replay",
                                 "task": LaunchConfiguration("task"), "operator": "replay"}])
    replay = Node(package="robodata_ros", executable="replay", name="robodata_replay", output="screen",
                  parameters=[{"episode": episode, "speed": LaunchConfiguration("speed")}])
    return LaunchDescription([
        DeclareLaunchArgument("episode"),
        DeclareLaunchArgument("robot", default_value="ur5_sim"),
        DeclareLaunchArgument("raw_root", default_value="/data/ros_raw"),
        DeclareLaunchArgument("task", default_value="replayed episode"),
        DeclareLaunchArgument("speed", default_value="1.0"),
        recorder, replay,
        RegisterEventHandler(OnProcessExit(target_action=replay, on_exit=[EmitEvent(event=Shutdown())])),
    ])
