# Base Image on DRL image
FROM mitdrl/ubuntu:latest

# Timezone
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y tzdata
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone



# Update something to the bashrc (/etc/bashrc_skipper) to customize your shell
RUN pip install pyfiglet
RUN echo -e "alias py='python'" >> /etc/bashrc_skipper
# FROM tensorflow/tensorflow:latest-gpu
RUN conda install python=3.6 tensorflow-gpu=2 tqdm

RUN apt update && apt install -y git wget
RUN pip install dm-acme\
    && pip install dm-acme[jax]\
    && pip install dm-acme[envs]\
    && pip install dm-reverb jax tensorflow_probability==0.11.0 trfl dm-sonnet imageio imageio-ffmpeg dataclasses pyyaml
RUN wget https://github.com/axelbr/racecar_gym/releases/download/tracks-v1.0.0/all.zip
RUN git clone https://github.com/axelbr/racecar_gym.git
RUN pip install --user -e racecar_gym/
RUN mv all.zip racecar_gym/models/scenes/ && cd racecar_gym/models/scenes/ && unzip all.zip

COPY . .

# Switch to src directory
WORKDIR /src
# Copy your code into the docker that is assumed to live in . (on machine)
COPY ./ /src
