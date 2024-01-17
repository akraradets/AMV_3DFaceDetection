FROM ubuntu:22.04

WORKDIR /root/project

ENV TERM=xterm
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/root/project
ENV DISPLAY host.docker.internal:0

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-dev \
    python3-opencv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*



CMD tail -f /dev/null

# docker run -it -v $PWD:/app/ --device=/dev/video0:/dev/video0 
# -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY opencv-webcam bash:q!
