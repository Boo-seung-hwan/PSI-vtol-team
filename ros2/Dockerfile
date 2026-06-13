FROM ros:humble-ros-base-jammy

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update -o Acquire::Retries=5 && \
    apt-get install -y --fix-missing --no-install-recommends \
    locales \
    curl \
    gnupg2 \
    lsb-release \
    sudo \
    git \
    vim \
    wget \
    software-properties-common \
    python3-pip \
    build-essential \
    ca-certificates \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    v4l-utils \
    usbutils \
    ros-humble-rmw-fastrtps-cpp \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-std-msgs \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-vision-msgs \
    git cmake g++ make libasio-dev libtinyxml2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN locale-gen en_US en_US.UTF-8 || true
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Gazebo / ROS-GZ packages.
RUN curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
    -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable jammy main" \
    > /etc/apt/sources.list.d/gazebo-stable.list && \
    apt-get update -o Acquire::Retries=5 && \
    apt-get install -y --fix-missing --no-install-recommends \
    gz-harmonic \
    ros-humble-ros-gzharmonic \
    && rm -rf /var/lib/apt/lists/*

RUN rosdep init 2>/dev/null || true && rosdep update || true

RUN python3 -m pip install --upgrade pip

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir --default-timeout=120 --retries 10 --ignore-installed -r /tmp/requirements.txt

# ROS Humble cv_bridge is not compatible with NumPy 2.x ABI in this image.
RUN python3 -m pip install --no-cache-dir --force-reinstall "numpy==1.26.4"

WORKDIR /workspace/ros2

ENV PATH="/usr/local/bin:${PATH}"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
