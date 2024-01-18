FROM ubuntu:22.04

WORKDIR /root/src

ENV TERM=xterm
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/root/src
ENV DISPLAY host.docker.internal:0

RUN apt-get update && apt-get -y upgrade
RUN apt install -y build-essential cmake make automake gcc g++ subversion python3-dev gfortran libopenblas-dev
RUN apt install -y libopencv-dev
RUN apt install -y python3-opencv
RUN apt install -y nautilus
RUN apt install -y python3-pip
RUN apt install -y libx11-dev
RUN pip install dlib
# RUN pip install pipenv

# RUN --mount=type=bind,source=./src/Pipfile,target=/root/src/Pipfile \
#     --mount=type=bind,source=./src/Pipfile.lock,target=/root/src/Pipfile.lock \
#     pipenv install


RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
CMD tail -f /dev/null