import os
from io import BytesIO

import ffmpeg
import parse
import yaml
from PIL import Image
from tqdm import tqdm; tqdm.monitor_interval = 0

from ..common_util.global_vars import PROJ_DIR, FLICKR_VIDEO_NAME_FORMAT
from ..common_util.misc import makedirs

VIDEO_DATASET_DIR = os.path.abspath(os.path.join(PROJ_DIR, 'datasets', 'flickr-raw'))
CLIP_DATASET_DIR = os.path.abspath(os.path.join(PROJ_DIR, 'datasets', 'flickr-raw-clips'))
MAX_CLIP_LENGTH = 90
MIN_CLIP_LENGTH = 45


def get_video_frames(video_path, start_index, end_index):
    """Produces an iterator yielding the frames between the specified indexes (inclusive).

    :param video_path: The path to the given video (str)
    :param start_index: The index of the first frame (int)
    :param end_index: The index of the last frame (int)
    :return: iterator of PIL.Image
    """
    assert 0 <= start_index < get_num_frames(video_path)
    assert 0 <= end_index < get_num_frames(video_path)

    # Produce the bytes corresponding to the desired RGB frames
    clip_length = end_index - start_index + 1
    all_frame_bytes, _ = (
        ffmpeg
        .input(video_path)
        .filter('select', 'gte(n,{})'.format(start_index))
        .output('pipe:', vframes=clip_length, format='rawvideo', pix_fmt='rgb24', vsync=0)
        .run(capture_stdout=True, capture_stderr=True)
    )
    # Wrap the data in a byte reader
    reader = BytesIO(all_frame_bytes)

    # Read the bytes of each frame into a PIL Image
    width, height = get_resolution(video_path)
    for i in range(clip_length):
        frame_bytes = reader.read(3 * height * width)
        frame = Image.frombytes('RGB', (width, height), frame_bytes)
        yield frame


def save_clips(video_path, clip_dataset_root, min_clip_length=MIN_CLIP_LENGTH, max_clip_length=MAX_CLIP_LENGTH):
    """Saves the clips of the given video to the clip dataset.

    :param video_path: Path to video file to be cut (str)
    :param clip_dataset_root: The root directory where clip folders will be saved (str)
    """
    num_frames = get_num_frames(video_path)

    for clip_index, clip_start_frame_index in enumerate(range(0, num_frames, max_clip_length)):
        clip_length = min(num_frames - clip_start_frame_index, max_clip_length)
        if clip_length < min_clip_length:
            # Clip is at end of video and is too short, so quit
            return

        # Construct the root path for the given clip
        file_name_no_ext, _ = os.path.splitext(os.path.basename(video_path))
        clip_name = f'{file_name_no_ext}-{clip_index:05d}'
        clip_root_path = os.path.join(clip_dataset_root, clip_name)
        makedirs(clip_root_path)

        # Retrieve and save the frames for the current clip
        frames = get_video_frames(video_path, clip_start_frame_index, clip_start_frame_index + clip_length - 1)
        pbar = tqdm(total=clip_length, desc=f'Clip {clip_index}', unit='frames', position=1, leave=False)
        for local_frame_index, frame in enumerate(frames):
            frame_save_path = os.path.join(clip_root_path, f'{local_frame_index:05d}.jpg')
            frame.save(frame_save_path, quality=95, optimize=True)
            pbar.update()


def get_num_frames(video_path):
    """Returns the number of frames in the given video (approximated via FFprobe).

    :param video_path: Path to the video file (str)
    :return: int
    """
    ffprobe_result = ffmpeg.probe(video_path, select_streams='v:0')
    num_frames = int(ffprobe_result['streams'][0]['nb_frames'])

    return num_frames


def get_resolution(video_path):
    """Returns the resolution (H, W) of the given video (approximated via FFprobe).

    :param video_path: Path to the video file (str)
    :return: 2-tuple of int
    """
    ffprobe_result = ffmpeg.probe(video_path, select_streams='v:0')
    width = ffprobe_result['streams'][0]['width']
    height = ffprobe_result['streams'][0]['height']

    return width, height


def main():
    flickr_config = yaml.safe_load(open(os.path.join(PROJ_DIR, 'flickr-config.yml'), 'r'))
    whitelisted_video_ids = flickr_config['whitelisted_videos']

    pbar = tqdm(total=len(whitelisted_video_ids), desc='Videos', unit='video', position=0)

    for root, _, file_names in os.walk(VIDEO_DATASET_DIR):
        for file_name in file_names:
            file_name_parse = parse.parse(FLICKR_VIDEO_NAME_FORMAT, file_name)
            if file_name_parse['video_id'] in whitelisted_video_ids:
                save_clips(os.path.join(root, file_name), CLIP_DATASET_DIR)
                pbar.update()


if __name__ == '__main__':
    main()
