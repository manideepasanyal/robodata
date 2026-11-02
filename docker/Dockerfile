# robodata: ROS 2 Jazzy (Ubuntu 24.04, Python 3.12) + pipeline + LeRobot + MuJoCo (headless OSMesa)
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-venv python3-pip python3-colcon-common-extensions build-essential \
      ffmpeg libosmesa6 libgl1 libegl1 libglib2.0-0 \
      ros-jazzy-diagnostic-msgs ros-jazzy-rosidl-runtime-py ros-jazzy-std-srvs \
    && rm -rf /var/lib/apt/lists/*

# venv that still sees the apt-installed rclpy / message packages
RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH=/opt/venv/bin:$PATH VIRTUAL_ENV=/opt/venv \
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa ROBODATA_DATA=/data

# CPU torch keeps the image small; the pipeline does not need a GPU
RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.7,<2.12" "torchvision>=0.22,<0.27"

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e ".[sim,lerobot,dev]"
COPY configs ./configs
COPY tests ./tests
COPY ros2_ws/src ./ros2_ws/src
# run colcon with the venv interpreter so console-script shebangs point at /opt/venv (where robodata lives)
RUN . /opt/ros/jazzy/setup.sh && cd ros2_ws && python -m colcon build --symlink-install --event-handlers console_cohesion+

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
