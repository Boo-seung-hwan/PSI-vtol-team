FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libasio-dev \
    libtinyxml2-dev \
    && rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    git clone --depth 1 https://github.com/eProsima/Micro-XRCE-DDS-Agent.git && \
    cd Micro-XRCE-DDS-Agent && \
    mkdir build && cd build && \
    cmake .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd / && rm -rf /tmp/Micro-XRCE-DDS-Agent

CMD ["MicroXRCEAgent", "udp4", "-p", "8888"]
