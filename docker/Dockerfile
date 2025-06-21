FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# 1) Install system Python 3.12 from deadsnakes, plus prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
      python3.12 \
      python3.12-venv \
      curl \
      git \
      unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/*

# 2) Install Astral uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && mv ~/.local/bin/uv ~/.local/bin/uvx /usr/local/bin/ \
    && chmod +x /usr/local/bin/uv /usr/local/bin/uvx

WORKDIR /workspace

# 3) Copy only your lockfile
COPY uv.lock .
COPY pyproject.toml .

# 4) Sync against Python 3.12 explicitly
RUN uv sync --python python3.12

# At runtime you will bind-mount your code, so no COPY .
ENTRYPOINT ["uv", "run", "python"]
