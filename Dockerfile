FROM tensorflow/tensorflow:latest-gpu
RUN apt update && apt install -y git wget libpython3.6
RUN pip install --user dm-acme\
    && pip install --user dm-acme[jax]\
    && pip install --user dm-acme[envs]\
    && pip install --user dm-reverb==0.1.0 jax tensorflow_probability==0.11.1 trfl dm-sonnet imageio imageio-ffmpeg
RUN wget https://github.com/axelbr/racecar_gym/releases/download/tracks-v1.0.0/all.zip
RUN git clone https://github.com/axelbr/racecar_gym.git
RUN pip install --user -e racecar_gym/
RUN mv all.zip racecar_gym/models/scenes/ && cd racecar_gym/models/scenes/ && unzip all.zip
COPY . .
