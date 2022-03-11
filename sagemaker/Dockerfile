FROM tensorflow/tensorflow:1.15.5-gpu-py3

# Upgrade installed packages
RUN apt-get update && apt-get upgrade -y && apt-get clean

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

WORKDIR /code
RUN git clone https://github.com/thekevinscott/image-super-resolution /code

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -e ".[gpu]" --ignore-installed

ENV PYTHONPATH /code/ISR/:$PYTHONPATH

RUN python3 setup.py install

COPY src /code

ENV PATH "$PATH:/code"

RUN python3 --version

