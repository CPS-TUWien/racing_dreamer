import os
import imageio


def save_video(filename: str, frames, fps):
    if not os.path.exists(os.path.dirname(filename)):
        os.mkdir(os.path.dirname(filename))
    with imageio.get_writer(f'{filename}.mp4', fps=fps) as video:
        for frame in frames:
            video.append_data(frame)