# FROM nvidia/cuda:11.0.3-runtime-ubuntu18.04
# FROM nvidia/cuda:11.4.2-runtime-ubuntu20.04
FROM tensorflow/tensorflow:2.10.0-gpu
ARG DEBIAN_FRONTEND=noninteractive
# FROM tensorflow/tensorflow:1.15.5-gpu-py3

####### Old
# RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# RUN apt-key del 7fa2af80
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

####### Old
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Upgrade installed packages
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get clean && \
    apt install software-properties-common -y && \
    apt update

# Install system packages
RUN apt-get install -y --no-install-recommends \
      bzip2 \
      g++ \
      git \
      graphviz \
      libgl1-mesa-glx \
      libhdf5-dev \
      openmpi-bin \
      screen \
      curl \
      wget

# RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
# RUN mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
# RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
# RUN apt-get update
# RUN apt-get install libcudnn8=8.3.2.*-1+cuda11.5
# RUN apt-get install libcudnn8-dev=8.3.2.*-1+cuda11.5

RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install python3.6 python3-pip -y

WORKDIR /code
RUN git clone https://github.com/thekevinscott/image-super-resolution /code
# RUN git reset --hard f1189e71938e70fb45971948191a221b2b754665

RUN python3 -m pip install --upgrade pip
# RUN python3 -m pip install tensorflow==1.15.5
RUN python3 -m pip install -e ".[gpu]" --ignore-installed
RUN python3 -m pip install smart_open

ENV PYTHONPATH /code/ISR/:$PYTHONPATH

RUN python3 setup.py install

RUN python3 -m pip install --upgrade tensorflow-model-optimization==0.7.3

COPY src /code

ENV PATH "$PATH:/code"

ENTRYPOINT TF_CPP_MIN_LOG_LEVEL=2 python3 convert.py
