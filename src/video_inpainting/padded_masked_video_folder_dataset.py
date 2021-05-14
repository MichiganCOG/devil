import os
from itertools import cycle

from .padded_masked_video_dataset import PaddedMaskedVideoDataset


class PaddedMaskedVideoFolderDataset(PaddedMaskedVideoDataset):

    def __init__(self, frames_dataset_path, masks_dataset_path):
        self._frames_dataset_path = frames_dataset_path
        self._masks_dataset_path = masks_dataset_path

        frame_video_names = sorted(os.listdir(frames_dataset_path))
        mask_video_names = sorted(os.listdir(masks_dataset_path))

        super().__init__(frame_video_names, mask_video_names)


    def video_frame_files_iter(self, frame_video_name):
        frame_file_names = sorted(os.listdir(os.path.join(self._frames_dataset_path, frame_video_name)))
        for file_name in frame_file_names:
            frame_path = os.path.join(self._frames_dataset_path, frame_video_name, file_name)
            yield open(frame_path, 'rb')


    def video_mask_files_iter(self, mask_video_name):
        mask_file_names = sorted(os.listdir(os.path.join(self._masks_dataset_path, mask_video_name)))
        mask_file_names_c = cycle(mask_file_names + mask_file_names[len(mask_file_names)-2:0:-1])
        for file_name in mask_file_names_c:
            mask_path = os.path.join(self._masks_dataset_path, mask_video_name, file_name)
            yield open(mask_path, 'rb')
