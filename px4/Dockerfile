FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN sed -i 's/archive.ubuntu.com/mirror.kakao.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    lsb-release \
    gnupg \
    sudo \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-jinja2 \
    python3-numpy \
    python3-toml \
    python3-yaml \
    python3-setuptools \
    python3-wheel \
    python3-lxml \
    ninja-build \
    cmake \
    build-essential \
    ccache \
    unzip \
    zip \
    rsync \
    protobuf-compiler \
    libeigen3-dev \
    libxml2-utils \
    libxml2-dev \
    libxslt1-dev \
    libopencv-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

	
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    kconfiglib \
    jsonschema \
    pyros-genmsg \
    symforce \
    future \
    "empy==3.3.4"

RUN curl -fsSL https://packages.osrfoundation.org/gazebo.gpg | \
    gpg --dearmor -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/gazebo-stable.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    gz-harmonic \
    libgz-sim8-dev \
    libgz-transport13-dev \
    libgz-sensors8-dev \
    libgz-plugin2-dev \
    libgz-msgs10-dev \
    libgz-math7-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/px4

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
