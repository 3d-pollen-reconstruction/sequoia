# get the development image from nvidia cuda 12.4.1
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

LABEL name="instantmesh" maintainer="instantmesh"

VOLUME /workspace/models

RUN mkdir -p /workspace/instantmesh
WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/America/Chicago /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get update && \
    apt-get install -y build-essential git wget vim libegl1-mesa-dev libglib2.0-0 unzip

# install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

ENV PATH="/workspace/miniconda3/bin:${PATH}"

# create conda environment
RUN conda create -n instantmesh python=3.10 -y

# Make the conda environment writable by all users
RUN chmod -R a+w /workspace/miniconda3

# Install Ninja and CUDA in the env
RUN conda run -n instantmesh conda install ninja -y
RUN conda run -n instantmesh conda install cuda -c nvidia/label/cuda-12.4.1 -y

# Install torch and other pip packages in the env
RUN conda run -n instantmesh pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
RUN conda run -n instantmesh pip install xformers==0.0.22.post7
RUN conda run -n instantmesh pip install triton

WORKDIR /workspace/instantmesh

ADD requirements.txt /workspace/instantmesh/requirements.txt
RUN conda run -n instantmesh pip install -r requirements.txt
RUN conda run -n instantmesh pip install wandb tensorboard pytorch-lightning==2.1.2 deepspeed

RUN apt-get install -y cmake ninja-build libosmesa6-dev libgl1-mesa-glx libglfw3

COPY . /workspace/instantmesh

CMD ["python", "app.py"]