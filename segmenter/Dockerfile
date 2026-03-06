FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

# Update and upgrade the system packages (Worker Template)
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install --yes --no-install-recommends \
        build-essential \
        git \
        wget \
        software-properties-common \
        google-perftools \
        curl \
        bash \
        python3-pip \
        python3-setuptools

RUN apt-get autoremove -y && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --break-system-packages torch torchvision
RUN pip3 install --break-system-packages segment-anything
RUN pip3 install --break-system-packages opencv-python runpod

WORKDIR /segmenter-server

# the runpod service
ADD segmenter-server.py ./

# SAM 2 weights
ADD LICENSE.SAM2.txt ./
ADD sam_vit_h_4b8939.pth ./

# miscellany
ADD Dockerfile ./
ADD README.server.txt ./

ENV RUNPOD_DEBUG_LEVEL=INFO
CMD ["python3", "-u", "/segmenter-server/segmenter-server.py"]
