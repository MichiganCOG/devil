import os
import shutil

import logging
import numpy as np
import argparse
from PIL import Image

from .utils.mask_generators import get_video_masks_by_moving_random_stroke, get_masked_ratio
from .utils.util import make_dirs, make_dir_under_root, get_everything_under
from .utils.readers import MaskReader
from .utils.logging_config import logger


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-od', '--output_dir',
        type=str,
        help="Output directory name"
    )
    parser.add_argument(
        '-im',
        '--image_masks', action='store_true',
        help="Set this if you want to generate independent masks in one directory."
    )
    parser.add_argument(
        '-vl', '--video_len',
        type=int,
        help="Maximum video length (i.e. #mask)"
    )
    parser.add_argument(
        '-ns', '--num_stroke',
        type=int,
        help="Number of stroke in one mask"
    )
    parser.add_argument(
        '-nsb', '--num_stroke_bound',
        type=int,
        nargs=2,
        help="Upper/lower bound of number of stroke in one mask"
    )
    parser.add_argument(
        '-n',
        type=int,
        help="Number of mask to generate"
    )
    parser.add_argument(
        '-sp',
        '--stroke_preset',
        type=str,
        default='rand_curve',
        help="Preset of the stroke parameters"
    )
    parser.add_argument(
        '-iw',
        '--image_width',
        type=int,
        default=320
    )
    parser.add_argument(
        '-ih',
        '--image_height',
        type=int,
        default=180
    )
    parser.add_argument(
        '--cluster_by_area',
        action='store_true'
    )
    parser.add_argument(
        '--leave_border_unmasked',
        type=int,
        help='Set this to a number, then a copy of the mask where the mask of border is erased.'
    )
    parser.add_argument(
        '--redo_without_generation',
        action='store_true',
        help='Set this, and the script will skip the generation and redo the left tasks'
             '(uncluster -> erase border -> re-cluster)'
    )
    parser.add_argument(
        '-s',
        '--seed',
        type=int,
        default=0,
        help='The RNG seed'
    )
    args = parser.parse_args()
    return args


def get_stroke_preset(stroke_preset):
    if stroke_preset == 'devil_fgs-l_fgm-l_fgd-l':
        return {
            "nVertexBound": [10, 22],
            "headSpeedBound": (10, 35),
            "headAccelerationBound": (0, 50),
            "maxHeadAccelerationAngle": 3.0,
            "brushWidthBound": (75, 150),
            "nAccelPointRatio": 0.05,
            "maxPointAccel": 0.05,
            "borderGap": (0, 0),
            "initBorderGap": (400, 400),
            "initSpeedBound": (0, 3),
            "centroidAccelDist": 300,
            "centroidAccelSlope": 1,
            "globalAccelRatio": 0.01,
            "globalAccelBound": (1, 1),
        }
    elif stroke_preset == 'devil_fgs-l_fgm-l_fgd-h':
        return {
            "nVertexBound": [10, 22],
            "headSpeedBound": (10, 35),
            "headAccelerationBound": (0, 50),
            "maxHeadAccelerationAngle": 3.0,
            "brushWidthBound": (75, 150),
            "nAccelPointRatio": 0.05,
            "maxPointAccel": 0.05,
            "borderGap": (0, 0),
            "initBorderGap": (400, 400),
            "initSpeedBound": (5, 8),
            "centroidAccelDist": 300,
            "centroidAccelSlope": 1,
            "globalAccelRatio": 0.01,
            "globalAccelBound": (1, 3),
        }
    elif stroke_preset == 'devil_fgs-l_fgm-h_fgd-l':
        return {
            "nVertexBound": [10, 22],
            "headSpeedBound": (10, 25),
            "headAccelerationBound": (0, 50),
            "maxHeadAccelerationAngle": 3.0,
            "brushWidthBound": (50, 125),
            "nAccelPointRatio": 0.8,
            "maxPointAccel": 1.0,
            "borderGap": (0, 0),
            "initBorderGap": (400, 400),
            "initSpeedBound": (0, 3),
            "centroidAccelDist": 150,
            "centroidAccelSlope": 1,
            "globalAccelRatio": 0.01,
            "globalAccelBound": (1, 1),
        }
    elif stroke_preset == 'devil_fgs-l_fgm-h_fgd-h':
        return {
            "nVertexBound": [10, 22],
            "headSpeedBound": (10, 25),
            "headAccelerationBound": (0, 50),
            "maxHeadAccelerationAngle": 3.0,
            "brushWidthBound": (50, 125),
            "nAccelPointRatio": 0.8,
            "maxPointAccel": 1.0,
            "borderGap": (0, 0),
            "initBorderGap": (400, 400),
            "initSpeedBound": (9, 15),
            "centroidAccelDist": 150,
            "centroidAccelSlope": 1,
            "globalAccelRatio": 0.01,
            "globalAccelBound": (1, 3),
        }
    elif stroke_preset == 'devil_fgs-h_fgm-l_fgd-l':
        return {
            "nVertexBound": [7, 13],
            "headSpeedBound": (50, 80),
            "headAccelerationBound": (0, 70),
            "maxHeadAccelerationAngle": 3.0,
            "brushWidthBound": (205, 375),
            "nAccelPointRatio": 0.05,
            "maxPointAccel": 0.5,
            "borderGap": (0, 0),
            "initBorderGap": (300, 300),
            "initSpeedBound": (0, 3),
            "centroidAccelDist": None,
            "centroidAccelSlope": 1,
            "globalAccelRatio": 0.01,
            "globalAccelBound": (1, 1),
        }
    elif stroke_preset == 'devil_fgs-h_fgm-l_fgd-h':
        return {
            "nVertexBound": [7, 13],
            "headSpeedBound": (50, 80),
            "headAccelerationBound": (0, 70),
            "maxHeadAccelerationAngle": 3.0,
            "brushWidthBound": (205, 375),
            "nAccelPointRatio": 0.05,
            "maxPointAccel": 0.5,
            "borderGap": (0, 0),
            "initBorderGap": (300, 300),
            "initSpeedBound": (5, 8),
            "centroidAccelDist": None,
            "centroidAccelSlope": 1,
            "globalAccelRatio": 0.01,
            "globalAccelBound": (1, 3),
        }
    elif stroke_preset == 'devil_fgs-h_fgm-h_fgd-l':
        return {
            "nVertexBound": [7, 13],
            "headSpeedBound": (50, 70),
            "headAccelerationBound": (0, 60),
            "maxHeadAccelerationAngle": 3.0,
            "brushWidthBound": (150, 300),
            "nAccelPointRatio": 0.8,
            "maxPointAccel": 1.5,
            "borderGap": (0, 0),
            "initBorderGap": (300, 300),
            "initSpeedBound": (0, 3),
            "centroidAccelDist": 300,
            "centroidAccelSlope": 0.75,
            "globalAccelRatio": 0.01,
            "globalAccelBound": (1, 1),
        }
    elif stroke_preset == 'devil_fgs-h_fgm-h_fgd-h':
        return {
            "nVertexBound": [7, 13],
            "headSpeedBound": (50, 70),
            "headAccelerationBound": (0, 60),
            "maxHeadAccelerationAngle": 3.0,
            "brushWidthBound": (150, 300),
            "nAccelPointRatio": 0.8,
            "maxPointAccel": 1.5,
            "borderGap": (0, 0),
            "initBorderGap": (300, 300),
            "initSpeedBound": (9, 15),
            "centroidAccelDist": 300,
            "centroidAccelSlope": 0.75,
            "globalAccelRatio": 0.01,
            "globalAccelBound": (1, 3),
        }
    else:
        raise NotImplementedError(f'The stroke presetting "{stroke_preset}" does not exist.')


def copy_masks_without_border(root_dir, args):
    def erase_mask_border(mask, gap):
        pix = np.asarray(mask).astype('uint8') * 255
        pix[:gap, :] = 255
        pix[-gap:, :] = 255
        pix[:, :gap] = 255
        pix[:, -gap:] = 255
        return Image.fromarray(pix).convert('1')

    wo_border_dir = root_dir + '_noBorder'
    logger.info(f'Copying all masks')
    shutil.copytree(root_dir, wo_border_dir)

    for i, filename in enumerate(get_everything_under(wo_border_dir)):
        if i % 100 == 0:
            logger.info(f'Erasing {filename}\'s border')
        if args.image_masks:
            mask = Image.open(filename)
            mask_wo_border = erase_mask_border(mask, args.leave_border_unmasked)
            mask_wo_border.save(filename)
        else:
            # filename is a diretory containing multiple mask files
            for f in get_everything_under(filename, pattern='*.png'):
                mask = Image.open(f)
                mask_wo_border = erase_mask_border(mask, args.leave_border_unmasked)
                mask_wo_border.save(f)

    return wo_border_dir


def cluster_by_masked_area(root_dir, args):
    logger.info(f'Clustering {root_dir}')
    clustered_dir = root_dir + '_clustered'
    make_dirs(clustered_dir)
    radius = 5

    # all masks with ratio in x +- radius will be stored in sub-directory x
    clustered_centors = np.arange(radius, 100, radius * 2)
    clustered_subdirs = []
    for c in clustered_centors:
        # make sub-directories for each ratio range
        clustered_subdirs.append(make_dir_under_root(clustered_dir, str(c)))

    for i, filename in enumerate(get_everything_under(root_dir)):
        if i % 100 == 0:
            logger.info(f'clustering {filename}')
        if args.image_masks:
            ratio = get_masked_ratio(Image.open(filename))
        else:
            # filename is a diretory containing multiple mask files
            ratio = np.mean([
                get_masked_ratio(Image.open(f))
                for f in get_everything_under(filename, pattern='*.png')
            ])

        # find the nearest centor
        for i, c in enumerate(clustered_centors):
            if c - radius <= ratio * 100 <= c + radius:
                shutil.move(filename, clustered_subdirs[i])
                break

    shutil.rmtree(root_dir)
    os.rename(clustered_dir, root_dir)


def decide_nStroke(num_stroke, num_stroke_bound, rng):
    if num_stroke is not None:
        return num_stroke
    elif num_stroke_bound is not None:
        return rng.randint(num_stroke_bound[0], num_stroke_bound[1])
    else:
        raise ValueError('One of "-ns" or "-nsb" is needed')


def main(args):
    logger.setLevel(logging.WARNING)
    prog_rng = np.random.RandomState(args.seed)

    preset = get_stroke_preset(args.stroke_preset)
    make_dirs(args.output_dir)

    if args.redo_without_generation:
        assert(len(get_everything_under(args.output_dir)) > 0)
        # put back clustered masks
        for clustered_subdir in get_everything_under(args.output_dir):
            if not os.path.isdir(clustered_subdir):
                continue
            for f in get_everything_under(clustered_subdir):
                shutil.move(f, args.output_dir)
            os.rmdir(clustered_subdir)

    else:
        if args.image_masks:
            for i in range(args.n):
                if i % 100 == 0:
                    logger.info(f'Generating mask number {i:07d}')
                nStroke = decide_nStroke(args)
                mask = get_video_masks_by_moving_random_stroke(video_len=1, imageWidth=args.image_width,
                                                               imageHeight=args.image_height, nStroke=nStroke,
                                                               **preset)[0]
                mask.save(os.path.join(args.output_dir, f'{i:07d}.png'))

        else:
            for i in range(args.n):
                cur_mask_rng = np.random.RandomState(prog_rng.randint(2**32))
                mask_dir = make_dir_under_root(args.output_dir, f'{i:05d}')
                mask_reader = MaskReader(mask_dir, read=False)

                nStroke = decide_nStroke(args.num_stroke, args.num_stroke_bound, cur_mask_rng)
                masks = get_video_masks_by_moving_random_stroke(video_len=args.video_len, imageWidth=args.image_width,
                                                                imageHeight=args.image_height, nStroke=nStroke,
                                                                rng=cur_mask_rng, **preset)

                mask_reader.set_files(masks)
                mask_reader.save_files(output_dir=mask_reader.dir_name)

    if args.leave_border_unmasked is not None:
        logger.info(f'Create a copy of all output and erase the copies\' s border '
                    f'by {args.leave_border_unmasked} pixels')
        dir_leave_border = copy_masks_without_border(args.output_dir, args)
        if args.cluster_by_area:
            cluster_by_masked_area(dir_leave_border, args)

    if args.cluster_by_area:
        cluster_by_masked_area(args.output_dir, args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
