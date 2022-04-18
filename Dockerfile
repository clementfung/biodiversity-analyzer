ARG BASE_IMAGE
FROM $BASE_IMAGE

MAINTAINER Clement Fung <clementf@andrew.cmu.edu>

# if needed, can install Ubuntu packages here, e.g.:
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y vim
RUN apt-get install -y virtualenv
RUN apt-get install -y libsm6 libxext6 libxrender-dev

RUN mkdir /pwwl
WORKDIR /pwwl

# run any commands you need to set up your environment, e.g.:
RUN mkdir -p /root/.config/matplotlib && \
    echo "backend : Agg" >> /root/.config/matplotlib/matplotlibrc

# install any needed python packages not already provided by the
# base docker image
ADD requirements-docker.txt /pwwl/

RUN pip install --upgrade pip
RUN pip install opencv-python==4.2.0.34
RUN pip install rasterio==1.2.10
RUN pip install -r requirements-docker.txt
