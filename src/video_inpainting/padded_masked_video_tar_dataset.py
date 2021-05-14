import tarfile
from itertools import cycle

from .padded_masked_video_dataset import PaddedMaskedVideoDataset


class PaddedMaskedVideoTarDataset(PaddedMaskedVideoDataset):

    def __init__(self, frames_dataset_path, masks_dataset_path):
        self._frames_dataset_tar = tarfile.open(frames_dataset_path, 'r')
        self._masks_dataset_tar = tarfile.open(masks_dataset_path, 'r')

        frame_video_names = sorted([info.name for info in self._frames_dataset_tar.getmembers() if info.isdir()])
        mask_video_names = sorted([info.name for info in self._masks_dataset_tar.getmembers() if info.isdir()])

        super().__init__(frame_video_names, mask_video_names)


    def video_frame_files_iter(self, frame_video_name):
        frame_paths = sorted([info.name for info in self._frames_dataset_tar.getmembers()
                              if info.name.startswith(frame_video_name) and info.isfile()])
        for frame_path in frame_paths:
            yield self._frames_dataset_tar.extractfile(frame_path)


    def video_mask_files_iter(self, mask_video_name):
        mask_paths = sorted([info.name for info in self._masks_dataset_tar.getmembers()
                             if info.name.startswith(mask_video_name) and info.isfile()])
        mask_paths_c = cycle(mask_paths + mask_paths[len(mask_paths)-2:0:-1])
        for mask_path in mask_paths_c:
            yield self._masks_dataset_tar.extractfile(mask_path)
