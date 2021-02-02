FROM mitdrl/ubuntu:latest
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y tzdata git wget unzip
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN conda install python=3.7 tensorflow-gpu=2
RUN pip uninstall -y tensorflow
RUN pip install dm-acme\
    && pip install dm-acme[jax]\
    && pip install dm-acme[envs]\
    && pip install dm-reverb==0.1.0 jax tensorflow-gpu==2.3.2  tensorflow_probability==0.11.0 trfl dm-sonnet imageio imageio-ffmpeg dataclasses pyyaml
RUN wget https://github.com/axelbr/racecar_gym/releases/download/tracks-v1.0.0/all.zip
RUN git clone https://github.com/axelbr/racecar_gym.git
RUN pip install -e racecar_gym/
RUN mv all.zip racecar_gym/models/scenes/ && cd racecar_gym/models/scenes/ && unzip all.zip
COPY . .
