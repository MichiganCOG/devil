import os
from math import floor

import pandas as pd
from parse import parse

from ..common_util.global_vars import PROJ_DIR
from ..common_util.misc import makedirs

VIDEO_FILE_NAME_FIELDS = ['flickr_user_id', 'flickr_video_id', 'clip_index']
VIDEO_NAME_FMT = '{flickr_user_id}-{flickr_video_id}-{clip_index}'
SOURCE_DATASET_NAME = 'flickr-raw-clips'
MASK_DATASET_NAME = 'fvi-masks'
VIDEO_PROP_NAMES = [
    ('camera_motion_score', 'cm'),
    ('bg_scene_motion_score', 'bsm'),
]

# Absolutely no clips will be sampled beyond this point
MAX_PERCENTILE = 0.3
MAX_PERCENTILE = {
    'camera_motion_score': 0.16,
    'bg_scene_motion_score': 0.16,
}
# Total number of clips to sample
NUM_VIDEOS_PER_SPLIT = 150

FVI_SPLITS = [
    'fgs-l',
    'fgs-h',
    'fgm-l',
    'fgm-h',
    'fgd-l',
    'fgd-h',
]


def main():
    source_dataset_video_lists_root = os.path.join(PROJ_DIR, 'video-lists', SOURCE_DATASET_NAME)
    makedirs(source_dataset_video_lists_root)
    write_source_dataset_video_lists(source_dataset_video_lists_root)

    mask_video_lists_root = os.path.join(PROJ_DIR, 'video-lists', MASK_DATASET_NAME)
    makedirs(mask_video_lists_root)
    write_mask_dataset_video_lists(mask_video_lists_root)


def write_mask_dataset_video_lists(mask_video_lists_root):
    # Identify the non-marginalized splits of the mask dataset
    mask_dataset_root = os.path.join(PROJ_DIR, 'datasets', MASK_DATASET_NAME)
    mask_dataset_subset_names = sorted(os.listdir(mask_dataset_root))

    # Create a video list of all masks
    output_path = os.path.join(mask_video_lists_root, 'all.txt')
    with open(output_path, 'w') as f:
        for non_marg_split_name in mask_dataset_subset_names:
            # Identify all masks in the non-marginalized group and write their names to file
            non_marg_split_root = os.path.join(mask_dataset_root, non_marg_split_name)
            non_marg_split_mask_names = sorted(os.listdir(non_marg_split_root))
            for mask_name in non_marg_split_mask_names:
                f.write('{}\n'.format(os.path.join(non_marg_split_name, mask_name)))

    # Create a video list for each marginalized split of the mask dataset
    for marg_split_name in FVI_SPLITS:
        output_path = os.path.join(mask_video_lists_root, f'{marg_split_name}.txt')
        with open(output_path, 'w') as f:
            # Identify non-marginalized splits that belong to the current marginalized split
            for non_marg_split_name in mask_dataset_subset_names:
                if marg_split_name in non_marg_split_name:
                    # Identify all masks in the non-marginalized group and write their names to file
                    non_marg_split_root = os.path.join(mask_dataset_root, non_marg_split_name)
                    non_marg_split_mask_names = sorted(os.listdir(non_marg_split_root))
                    for mask_name in non_marg_split_mask_names:
                        f.write('{}\n'.format(os.path.join(non_marg_split_name, mask_name)))


def write_source_dataset_video_lists(source_dataset_video_lists_root):
    # Read scores
    scores_path = os.path.join(PROJ_DIR, 'scores', SOURCE_DATASET_NAME, 'scores.tsv')
    scores_df = pd.read_csv(scores_path, sep='\t', skiprows=2)
    # Read manually-annotated labels
    labels_path = os.path.join(PROJ_DIR, 'manual-labels', SOURCE_DATASET_NAME, 'labels.tsv')
    labels_df = pd.read_csv(labels_path, sep='\t', skiprows=2)
    df = scores_df.join(labels_df, on='video_name')
    augment_flickr_info(df)

    # Create video list of all videos
    df_sorted = df.sort_values('video_name')
    video_names = list(df_sorted['video_name'])
    output_path = os.path.join(source_dataset_video_lists_root, 'all.txt')
    with open(output_path, 'w') as f:
        f.writelines([f'{video_name}\n' for video_name in video_names])

    # Create video list for each property of interest
    for video_prop_name, short_video_prop_name in VIDEO_PROP_NAMES:
        # Sort by the current property
        df_sorted = df.sort_values(video_prop_name)
        num_candidates_per_split = int(floor(len(df) * MAX_PERCENTILE[video_prop_name]))

        # Write names of clips with lowest property score
        df_low_scoring = df_sorted[:num_candidates_per_split]
        output_path = os.path.join(source_dataset_video_lists_root, f'{short_video_prop_name}-l.txt')
        write_top_video_names(df_low_scoring, output_path, NUM_VIDEOS_PER_SPLIT)

        # Write names of clips with highest property score
        df_high_scoring = df_sorted[-num_candidates_per_split:].sort_values(video_prop_name, ascending=False)
        output_path = os.path.join(source_dataset_video_lists_root, f'{short_video_prop_name}-h.txt')
        write_top_video_names(df_high_scoring, output_path, NUM_VIDEOS_PER_SPLIT)


def augment_flickr_info(df):
    """Parse more Flickr information from the video name.

    This function adds columns to the given data frame by parsing the "video_name" column and extracting from it the
    Flickr user ID, the Flickr video name, and the clip index.

    :param df: The data frame to augment
    """
    for field_name in VIDEO_FILE_NAME_FIELDS:
        df[field_name] = df['video_name'].map(lambda x: parse(VIDEO_NAME_FMT, x)[field_name])


def write_top_video_names(df, output_path, max_num_videos=None):
    """Writes the names of the top videos to the given output path.

    The rows of the input data frame should already be sorted by the desire criteria.

    :param df: pd.DataFrame
    :param output_path: str
    :param max_num_videos: int
    """
    indexes = get_unique_first_sample_order(df['flickr_video_id'])
    if max_num_videos is not None:
        indexes = indexes[:max_num_videos]
    video_names = df.loc[indexes, 'video_name']
    with open(output_path, 'w') as f:
        for video_name in sorted(video_names):
            f.write(f'{video_name}\n')


def get_unique_first_sample_order(input):
    """Returns an ordered list of indexes to sample from.

    This function first chooses indexes that correspond to unique elements. Once all unique items have been selected, it
    effectively filters out already-selected indexes and repeats the process with the remaining items. This is repeated
    until all input items have been selected.

    In practice, the above procedure is done using a more efficient, single-pass algorithm.

    :param input: pandas.Series
    :return: list of indexes (int)
    """
    next_priority_table = {}
    priority = [[] for _ in range(len(input))]
    for index, flickr_video_id in input.iteritems():
        priority_index = next_priority_table.get(flickr_video_id, 0)
        priority[priority_index].append(index)
        next_priority_table[flickr_video_id] = priority_index + 1
    sampling_order = [item for sublist in priority for item in sublist]

    return sampling_order


if __name__ == '__main__':
    main()
