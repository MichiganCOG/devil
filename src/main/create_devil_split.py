import argparse
import os
from itertools import cycle
from time import sleep

import numpy as np
from PIL import Image
from tqdm import tqdm

from ..common_util.global_vars import PROJ_DIR
from ..common_util.image import cover_crop, invert_binary_image
from ..common_util.misc import makedirs, ThreadPool


def create_frames_split(source_dataset_name, source_dataset_name_short, source_dataset_split, image_width, image_height,
                        num_videos, rng_seed, no_shuffle, func):

    rng = np.random.RandomState(rng_seed)
    source_dataset_root = os.path.join(PROJ_DIR, 'datasets', source_dataset_name)
    output_dataset_root = os.path.join(PROJ_DIR, 'datasets', 'devil',
                                       '{}-{}'.format(source_dataset_name_short, source_dataset_split))
    makedirs(output_dataset_root)

    # Get names of source videos
    with open(os.path.join(PROJ_DIR, 'video-lists', source_dataset_name, f'{source_dataset_split}.txt')) as f:
        source_video_names = [x.strip() for x in f.readlines()]
    assert num_videos <= len(source_video_names)
    if not no_shuffle:
        rng.shuffle(source_video_names)

    # Render videos in parallel
    pool = ThreadPool(render_and_save_frames, 8)
    for i, source_video_name in zip(range(num_videos), source_video_names):
        pool.add_job(i, image_height, image_width, output_dataset_root, source_dataset_root, source_video_name)

    # Show rendering progress
    pbar = tqdm(total=num_videos)
    while pool.queue.unfinished_tasks > 0:
        num_unfinished_tasks = pool.queue.unfinished_tasks
        pbar.n = num_videos - num_unfinished_tasks
        pbar.refresh()
        sleep(1)
    pool.join()


def render_and_save_frames(i, image_height, image_width, output_dataset_root, source_dataset_root, source_video_name):

    source_video_path = os.path.join(source_dataset_root, source_video_name)
    frame_file_names = sorted(os.listdir(source_video_path))

    # Create new folder for current video
    cur_item_output_root = os.path.join(output_dataset_root, f'{i:05d}')
    makedirs(cur_item_output_root)

    for j, frame_file_name in enumerate(frame_file_names):
        # Read current video frame
        frame_path = os.path.join(source_video_path, frame_file_name)
        frame = Image.open(frame_path)
        if frame.mode != 'RGB':
            frame = frame.convert('RGB')

        # Resize and save frame
        frame_resized = cover_crop(frame, (image_width, image_height))
        frame_resized.save(os.path.join(cur_item_output_root, f'frame_{j:04d}_gt.png'))


def create_masks_split(rng_seed, mask_dataset_name, mask_dataset_name_short, mask_dataset_split, image_width,
                       image_height, num_videos, func):

    rng = np.random.RandomState(rng_seed)
    mask_dataset_root = os.path.join(PROJ_DIR, 'datasets', mask_dataset_name)
    output_dataset_root = os.path.join(PROJ_DIR, 'datasets', 'devil',
                                       '{}-{}'.format(mask_dataset_name_short, mask_dataset_split))
    makedirs(output_dataset_root)

    # Get names of mask videos
    with open(os.path.join(PROJ_DIR, 'video-lists', mask_dataset_name, f'{mask_dataset_split}.txt')) as f:
        mask_video_names = [x.strip() for x in f.readlines()]
    assert num_videos <= len(mask_video_names)
    rng.shuffle(mask_video_names)
    # Get flags to indicate which videos should be reversed
    mask_rev_time_flags = rng.randint(2, size=len(mask_video_names), dtype=np.bool)
    mask_video_names_c = cycle(mask_video_names)
    mask_rev_time_flags_c = cycle(mask_rev_time_flags)

    # Render videos in parallel
    pool = ThreadPool(render_and_save_masks, 8)
    for i, mask_video_name, mask_rev_time_flag in zip(range(num_videos), mask_video_names_c, mask_rev_time_flags_c):
        pool.add_job(i, image_height, image_width, mask_dataset_root, mask_rev_time_flag, mask_video_name,
                     output_dataset_root)

    # Show rendering progress
    pbar = tqdm(total=num_videos)
    while pool.queue.unfinished_tasks > 0:
        num_unfinished_tasks = pool.queue.unfinished_tasks
        pbar.n = num_videos - num_unfinished_tasks
        pbar.refresh()
        sleep(1)
    pool.join()


def render_and_save_masks(i, image_height, image_width, mask_dataset_root, mask_rev_time_flag, mask_video_name,
                          output_dataset_root):

    mask_video_path = os.path.join(mask_dataset_root, mask_video_name)
    mask_frame_file_names = sorted(os.listdir(mask_video_path))
    num_mask_files = len(mask_frame_file_names)
    output_index_range = range(num_mask_files - 1, -1, -1) if mask_rev_time_flag else range(num_mask_files)

    # Create new folder for current video-mask pair
    cur_item_output_root = os.path.join(output_dataset_root, f'{i:05d}')
    makedirs(cur_item_output_root)

    for j, mask_frame_file_name in zip(output_index_range, mask_frame_file_names):
        source_mask_path = os.path.join(mask_video_path, mask_frame_file_name)
        mask = Image.open(source_mask_path)
        assert mask.mode == '1'

        mask_resized = cover_crop(mask, (image_width, image_height))
        mask_resized = invert_binary_image(mask_resized)  # Invert so that BG is white and FG is black
        mask_resized.save(os.path.join(cur_item_output_root, f'frame_{j:04d}_mask.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_width', type=str, default=832)
    parser.add_argument('--image_height', type=str, default=480)
    parser.add_argument('--num_videos', type=int, default=150)
    parser.add_argument('--rng_seed', type=int, default=0)

    subparsers = parser.add_subparsers()

    frames_cmd_parser = subparsers.add_parser('frames')
    frames_cmd_parser.set_defaults(func=create_frames_split)
    frames_cmd_parser.add_argument('source_dataset_split', type=str)
    frames_cmd_parser.add_argument('--source_dataset_name', type=str, default='flickr-raw-clips')
    frames_cmd_parser.add_argument('--source_dataset_name_short', type=str, default='flickr')
    frames_cmd_parser.add_argument('--no_shuffle', action='store_true')

    masks_cmd_parser = subparsers.add_parser('masks')
    masks_cmd_parser.set_defaults(func=create_masks_split)
    masks_cmd_parser.add_argument('mask_dataset_split', type=str)
    masks_cmd_parser.add_argument('--mask_dataset_name', type=str, default='fvi-masks')
    masks_cmd_parser.add_argument('--mask_dataset_name_short', type=str, default='fvi')

    args = parser.parse_args()
    args.func(**vars(args))
