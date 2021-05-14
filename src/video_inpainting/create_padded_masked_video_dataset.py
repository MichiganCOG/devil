import os

from .padded_masked_video_folder_dataset import PaddedMaskedVideoFolderDataset
from .padded_masked_video_tar_dataset import PaddedMaskedVideoTarDataset


def create_padded_masked_video_dataset(frames_dataset_path, masks_dataset_path):
    if os.path.isdir(frames_dataset_path) and os.path.isdir(masks_dataset_path):
        return PaddedMaskedVideoFolderDataset(frames_dataset_path, masks_dataset_path)
    else:
        _, frames_dataset_ext = os.path.splitext(frames_dataset_path)
        _, masks_dataset_ext = os.path.splitext(masks_dataset_path)
        if frames_dataset_ext == '.tar' and masks_dataset_ext == '.tar':
            return PaddedMaskedVideoTarDataset(frames_dataset_path, masks_dataset_path)
        else:
            raise ValueError('Given paths must both be directories or .tar files')
