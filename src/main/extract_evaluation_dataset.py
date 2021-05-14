import argparse
import os

from ..common_util.misc import makedirs
from ..video_inpainting import create_padded_masked_video_dataset


def main(frames_dataset_path, masks_dataset_path, final_dataset_path):
    dataset = create_padded_masked_video_dataset(frames_dataset_path, masks_dataset_path)
    for i in range(len(dataset)):
        video_name = dataset.get_video_name(i)
        extract_video_path = os.path.join(final_dataset_path, video_name)
        makedirs(extract_video_path)
        dataset.extract_masked_video(i, extract_video_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('frames_dataset_path', type=str)
    parser.add_argument('masks_dataset_path', type=str)
    parser.add_argument('final_dataset_path', type=str)

    args = parser.parse_args()
    main(**vars(args))
