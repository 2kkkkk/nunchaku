# Use an NVIDIA base image with CUDA support

ARG CUDA_IMAGE="12.8.1-devel-ubuntu24.04"

FROM nvidia/cuda:${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

ARG PYTHON_VERSION=3.11
ARG TORCH_VERSION=2.6
ARG TORCHVISION_VERSION=0.21
ARG TORCHAUDIO_VERSION=2.6
ARG CUDA_SHORT_VERSION=12.8

# Set working directory
WORKDIR /

RUN echo PYTHON_VERSION=${PYTHON_VERSION} \
    && echo CUDA_SHORT_VERSION=${CUDA_SHORT_VERSION} \
    && echo TORCH_VERSION=${TORCH_VERSION} \
    && echo TORCHVISION_VERSION=${TORCHVISION_VERSION} \
    && echo TORCHAUDIO_VERSION=${TORCHAUDIO_VERSION}

# Setup timezone and install system dependencies
RUN 'tzdata tzdata/Areas select America' | debconf-set-selections \
    && echo 'tzdata tzdata/Zones/America select New_York' | debconf-set-selections \
    && apt update -y \
    && apt install software-properties-common -y \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt update

RUN apt install python${PYTHON_VERSION} python${PYTHON_VERSION}-dev g++-11 gcc-11 -y \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION} && apt install python${PYTHON_VERSION}-distutils -y \
    && update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --set python /usr/bin/python${PYTHON_VERSION}

RUN apt install curl git sudo libibverbs-dev -y \
    && apt install -y rdma-core infiniband-diags openssh-server perftest ibverbs-providers libibumad3 libibverbs1 libnl-3-200 libnl-route-3-200 librdmacm1 \
    && curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py \
    && python3 --version \
    && python3 -m pip --version \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 1 && update-alternatives --set gcc /usr/bin/gcc-11 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 1 && update-alternatives --set g++ /usr/bin/g++-11 \
    && apt clean

# Install building dependencies
RUN pip install torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} torchaudio==${TORCHAUDIO_VERSION} --index-url https://download.pytorch.org/whl/cu${CUDA_SHORT_VERSION}
RUN pip install ninja wheel diffusers transformers accelerate sentencepiece protobuf huggingface_hub comfy-cli

# Start building
RUN git clone https://github.com/mit-han-lab/nunchakukp.git \
    && cd nunchakukp \
    && git submodule init \
    && git submodule update \
    && NUNCHAKUKP_INSTALL_MODE=ALL python setup.py develop

RUN cd .. && git clone https://github.com/comfyanonymous/ComfyUI \
    && cd ComfyUI/custom_nodes && git clone https://github.com/ltdrdata/ComfyUI-Manager comfyui-manager \
    && git clone https://github.com/mit-han-lab/ComfyUI-nunchakukp.git nunchakukp_nodes \
    && cd .. && mkdir -p user/default/workflows/ && cp custom_nodes/nunchakukp_nodes/workflows/* user/default/workflows/
