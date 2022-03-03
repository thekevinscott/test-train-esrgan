FROM tensorflow/tensorflow:2.2.3-gpu-py3

RUN apt-get update
RUN apt-get install -y \
  git 
