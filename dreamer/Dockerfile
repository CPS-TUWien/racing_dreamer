FROM tensorflow/tensorflow:2.3.1-gpu AS base

# update
# nvidia-docker issue: workaround https://github.com/NVIDIA/nvidia-docker/issues/1632
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN python --version && pip --version && pip list
RUN apt update -y && apt install git wget unzip -y

# install requirements
WORKDIR /build
COPY requirements.txt /build
RUN pip install -r requirements.txt

# download track maps
WORKDIR /build/src/racecar-gym/models/scenes
RUN wget https://github.com/axelbr/racecar_gym/releases/download/tracks-v1.0.0/all.zip && unzip all.zip

# Switch to src directory
WORKDIR /src

# add working directory to the python path
ENV PYTHONPATH "${PYTHONPATH}:/src"

# Copy your code into the docker that is assumed to live in . (on machine)
COPY ./ /src
