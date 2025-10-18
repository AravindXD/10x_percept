FROM ros:humble-ros-base-jammy

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV DEBIAN_FRONTEND=noninteractive
ENV CONTAINER=true

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    python3-colcon-common-extensions \
    python3-pip \
    python3-rosdep \
    python3-vcstool \
    sudo \
    nano \
    vim \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

RUN rosdep init || true
RUN rosdep update

USER $USERNAME
WORKDIR /home/$USERNAME/workspace

RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

CMD ["bash"]