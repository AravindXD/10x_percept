# Base image for ROS 2 Humble
FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive
ENV CONTAINER=true

# Install essential packages for development
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    python3-vcstool \
    nano \
    vim \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Initialize rosdep
RUN rosdep init || true
RUN rosdep update

# Source ROS 2 setup by default
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Set working directory
WORKDIR /workspace

# Default command
CMD [ "bash" ]
