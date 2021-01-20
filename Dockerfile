FROM tensorflow/tensorflow:latest-gpu
RUN apt update && apt install -y git wget
RUN pip install --user dm-acme\
    && pip install --user dm-acme[reverb]\
    && pip install --user dm-acme[tf]\
    && pip install --user dm-acme[jax]\
    && pip install --user dm-acme[envs]\
    && pip install --user jax tensorflow==2.3.0 tensorflow_probability dm-sonnet imageio imageio-ffmpeg
RUN wget https://github.com/axelbr/racecar_gym/releases/download/tracks-v1.0.0/all.zip
RUN git clone https://github.com/axelbr/racecar_gym.git
RUN pip install -e racecar_gym/
COPY . .
CMD ["pip", "list"]