import argparse
import json
import os
from pprint import pprint

import cv2

from ..common_util.image import image_path_to_numpy
from ..common_util.misc import Unbuffered, str_no_class_fmt, makedirs
from ..devil import cam_motion_scoring, fg_size_scoring, cut_scoring, bg_scene_motion_scoring
from ..devil.config import get_default_config, namespace_to_dict, overwrite_config, get_config_schema_info, \
    dict_to_namespace


def main(config_pairs_str_list):

    # Load the configuration from default settings
    config_default = get_default_config().ScoreFlickrClips
    # Override the default configuration with parameters specified through a file or command line
    config = dict_to_namespace(namespace_to_dict(config_default))
    overwrite_config(config, config_pairs_str_list, prefix_key='ScoreFlickrClips')
    # Copy the configuration as a dict for convenience
    config_dict = namespace_to_dict(config)
    # Print the current configuration
    print('Using the following configuration:')
    pprint(config_dict)
    print()

    # Check that the output score types are valid
    for output_score_type in config_dict['output_scores']:
        assert output_score_type in config_default.output_scores, \
            f'"{output_score_type}" is not a valid output score type'

    # Write the parameters of the current run to file
    makedirs(os.path.dirname(config.output_path))
    output_file = Unbuffered(open(config.output_path, 'w'))
    args_str = json.dumps(config_dict)
    output_file.write(f'{args_str}\n\n')

    # Write the columns of the result table
    columns = ['video_name'] + config_dict['output_scores']
    columns_str = '\t'.join(columns)
    output_file.write(f'{columns_str}\n')

    # Produce the scorer objects
    cms_config = config_dict['CameraMotionScorer']
    afss_config = dict(num_sampled_frames=10)
    bsms_config = config_dict['BackgroundSceneMotionScorer']
    cms = getattr(cam_motion_scoring, cms_config['scorer_class_name'])(**cms_config['scorer_args'])
    afss = getattr(fg_size_scoring, 'AutomaticForegroundSizeScorer')(**afss_config)
    cs = getattr(cut_scoring, 'CutScorer')()
    bsms = getattr(bg_scene_motion_scoring, bsms_config['scorer_class_name'])(**bsms_config['scorer_args'])

    # Collect video names either from the DAVIS dataset or from a user-specified list of video names
    if config.video_list_path is None:
        video_names = sorted(os.listdir(config.dataset_root_path))
    else:
        with open(config.video_list_path, 'r') as f:
            video_names = f.read().strip().split()
    # Reduce the number of videos if specified by the user
    if config.max_num_videos is not None:
        video_names = video_names[:config.max_num_videos]

    for video_name in video_names:
        line = []
        # Write the video name column
        line.append(f'{video_name}')

        # Collect all the frames and foreground masks for the current video
        cur_video_frames_root = os.path.join(config.dataset_root_path, video_name)
        cur_video_frame_paths = [os.path.join(cur_video_frames_root, x)
                                 for x in sorted(os.listdir(cur_video_frames_root))]
        frames = [image_path_to_numpy(path) for path in cur_video_frame_paths]
        frames = [cv2.resize(frame, tuple(config.proc_resolution), interpolation=cv2.INTER_AREA) for frame in frames]
        fg_masks = [None for _ in cur_video_frame_paths]

        # Score camera motion
        if 'camera_motion_score' in config_dict['output_scores']:
            camera_motion_score = cms.score_camera_motion(frames, fg_masks)
            line.append(f'{camera_motion_score:.08f}')
        # Score foreground size
        if 'fg_size_score' in config_dict['output_scores']:
            fg_size_score = afss.score_foreground_size(frames)
            line.append(f'{fg_size_score:.08f}')
        # Score cuts
        if 'cut_score' in config_dict['output_scores']:
            cut_score = cs.score_cut(f'{cur_video_frames_root}/%05d.jpg')
            line.append(f'{cut_score:.08f}')
        if 'bg_scene_motion_score' in config_dict['output_scores']:
            bg_scene_motion_score = bsms.score_background_scene_motion(frames, fg_masks)
            line.append(f'{bg_scene_motion_score:.08f}')

        # Terminate the current line
        output_file.write('\t'.join(line) + '\n')

    output_file.close()


if __name__ == '__main__':
    # Add a description that lists the parameters that can be set in the configuration
    desc_list = ['Here are the keys, data types, and default values that are available/settable in the configuration:',
                 '', '']
    desc_list += [f'{key} ({str_no_class_fmt(data_type)}, default={default_value})'
                  for key, data_type, default_value in get_config_schema_info(prefix_key='ScoreFlickrClips')]
    desc = '\n'.join(desc_list)

    parser = argparse.ArgumentParser(description=desc, formatter_class=argparse.RawTextHelpFormatter,
                                     fromfile_prefix_chars='@')
    parser.add_argument('config_pairs_str_list', metavar='CONFIG_PAIRS', type=str, default=[], nargs='*',
                        help='Pairs that specify configuration keys and values (e.g., ScoreFlickrClips.dataset_root_path,'
                             ' "/data/my/dataset/path")')
    args = parser.parse_args()

    main(**vars(args))
