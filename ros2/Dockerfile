FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN sed -i 's|http://archive.ubuntu.com/ubuntu|http://mirror.kakao.com/ubuntu|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com/ubuntu|http://mirror.kakao.com/ubuntu|g' /etc/apt/sources.list

RUN apt-get update -o Acquire::Retries=5 && apt-get install -y \
    locales \
    curl \
    gnupg2 \
    lsb-release \
    sudo \
    git \
    vim \
    software-properties-common \
    python3-pip \
    python3-opencv \
    build-essential \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN add-apt-repository universe && apt-get update -o Acquire::Retries=5

RUN locale-gen en_US en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# ROS 2 repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg && \
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  > /etc/apt/sources.list.d/ros2.list

# Gazebo / OSRF repository for Harmonic packages
RUN curl -sSL https://packages.osrfoundation.org/gazebo.gpg \
  -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  > /etc/apt/sources.list.d/gazebo-stable.list

RUN apt-get update && apt-get install -y \
    ca-certificates \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    ros-humble-ros-base \
    ros-humble-rmw-fastrtps-cpp \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-vision-msgs \
    gz-harmonic \
    ros-humble-ros-gzharmonic \
    git cmake g++ make libasio-dev libtinyxml2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init 2>/dev/null || true && rosdep update ||true
RUN python3 -m pip install --upgrade pip

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --default-timeout=120 --retries 10 --ignore-installed -r /tmp/requirements.txt

WORKDIR /workspace/ros2

ENV PATH="/usr/local/bin:${PATH}"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
