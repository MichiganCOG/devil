import argparse
import os

import numpy as np
from PIL import ImageOps

from ..common_util.global_vars import PROJ_DIR
from ..common_util.misc import makedirs, ThreadPool
from ..fvi_masks.gen_masks import get_stroke_preset
from ..fvi_masks.utils.mask_generators import get_video_masks_by_moving_random_stroke_iterator


def render_fvi_mask(seed, mask_params, save_root):
    makedirs(save_root)

    rng = np.random.RandomState(seed)
    mask_params = dict(rng=rng, **mask_params)
    masks = get_video_masks_by_moving_random_stroke_iterator(**mask_params)

    for i, mask in enumerate(masks):
        # Make 0 correspond to BG pixels and 1 to FG pixels
        mask_inv = ImageOps.invert(mask.convert('L')).convert('1')
        mask_inv.save(os.path.join(save_root, f'{i:05d}.png'))


def main(seed, width, height, length, preset_name, num_videos, num_threads):
    preset = get_stroke_preset(preset_name)
    common_mask_params = dict(imageWidth=width, imageHeight=height, video_len=length, nStroke=1, **preset)
    meta_rng = np.random.RandomState(seed)
    video_seeds = meta_rng.randint(2**32, size=num_videos)

    mask_dataset_root = os.path.join(PROJ_DIR, 'datasets', 'fvi-masks', preset_name)
    makedirs(mask_dataset_root)

    pool = ThreadPool(render_fvi_mask, num_threads)
    for i, video_seed in enumerate(video_seeds):
        save_root = os.path.join(mask_dataset_root, f'{i:05d}')
        pool.add_job(seed=video_seed, mask_params=common_mask_params, save_root=save_root)
    pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    parser.add_argument('preset_name', type=str)
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-w', '--width', type=int, default=1920)
    parser.add_argument('-h', '--height', type=int, default=1080)
    parser.add_argument('-l', '--length', type=int, default=90)
    parser.add_argument('-n', '--num_videos', type=int, default=1000)
    parser.add_argument('-t', '--num_threads', type=int, default=8)

    args = parser.parse_args()
    main(**vars(args))
