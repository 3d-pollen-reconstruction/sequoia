FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tzdata \
    python3 python3-pip git git-lfs \
    libgl1 \
    libglib2.0-0 \
    libsm6 libxext6 libxrender1 \
    ffmpeg \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Enable Git LFS
RUN git lfs install

# Set working directory
WORKDIR /workspace

# Add a dummy build arg to force cache invalidation
ARG CACHE_DATE=unspecified
RUN echo "Cloning PixelNeRF fresh on $CACHE_DATE" && \
    rm -rf pixel-nerf && \
    git clone https://github.com/etiiiR/pixel-nerf.git

WORKDIR /workspace/pixel-nerf

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install -r requirements.txt

# Clone Hugging Face dataset using Git LFS
RUN rm -rf /workspace/pollen && \
    mkdir -p /workspace/pollen && \
    cd /workspace && \
    git lfs install && \
    git lfs clone https://huggingface.co/datasets/Etiiir/Pollen pollen

# Optional default command
# CMD ["python3", "train/train.py", "-n", "pollen", "-c", "conf/exp/pollen.conf", "-D", "pollen", "--gpu_id=0", "--resume"]
