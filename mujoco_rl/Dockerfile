FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV MUJOCO_GL=osmesa
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libgl1 \
    libglfw3 \
    libosmesa6 \
    libglew2.2 \
    libxrender1 \
    libxext6 \
    libxi6 \
    libxrandr2 \
    libxcursor1 \
    libxinerama1 \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/mujoco_rl

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD ["bash"]
