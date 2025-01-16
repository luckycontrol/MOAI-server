FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/MOAI-server/server/bin:$PATH

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y \
    bash git wget docker.io && \
    wget -qO- https://astral.sh/uv/install.sh | sh && \
    source "$HOME/.local/bin/env" && \
    git clone https://github.com/luckycontrol/MOAI-server.git && \
    cd MOAI-server && \
    uv init --python 3.11 && \
    uv venv server && \
    source server/bin/activate && \
    uv pip install docker "fastapi[standard]" && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /MOAI-server

VOLUME ["/MOAI-server", "/moai"]