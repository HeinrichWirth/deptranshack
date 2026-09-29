FROM ros:humble-ros-base-jammy@sha256:1813d3c85d7f96ff7d3012d865204583255740182db5d0065f8f8cd029a83138
RUN apt-get update && apt-get install -y --no-install-recommends ros-humble-rosbag2 ros-humble-rosbag2-storage-default-plugins && rm -rf /var/lib/apt/lists/*
WORKDIR /bridge
COPY app/live_protocol.py /bridge/live_protocol.py
COPY ros2/bridge.py /bridge/bridge.py
COPY ros2/observe_results.py /bridge/observe_results.py
COPY ros2/fastdds.xml /bridge/fastdds.xml
ENV PYTHONUNBUFFERED=1 FASTRTPS_DEFAULT_PROFILES_FILE=/bridge/fastdds.xml RMW_IMPLEMENTATION=rmw_fastrtps_cpp
CMD ["python3", "/bridge/bridge.py"]
