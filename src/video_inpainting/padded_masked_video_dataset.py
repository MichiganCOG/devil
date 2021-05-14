import os
from abc import ABC, abstractmethod

from PIL import Image

from .util import PILImagePadder
from ..common_util.misc import equalize_list_lengths


class PaddedMaskedVideoDataset(ABC):

    def __init__(self, frame_video_names, mask_video_names):
        self._frame_video_names, self._mask_video_names = equalize_list_lengths(frame_video_names, mask_video_names)
        self.padder = PILImagePadder(1)


    def __len__(self):
        return len(self._frame_video_names)


    def get_video_name(self, i):
        return self._frame_video_names[i]


    def get_raw_resolution(self, i):
        frame_file = next(self.video_frame_files_iter(self._frame_video_names[i]))
        return Image.open(frame_file).size


    def extract_masked_video(self, i, dest_root):
        frame_iter = self.video_frame_files_iter(self._frame_video_names[i])
        mask_iter = self.video_mask_files_iter(self._mask_video_names[i])
        for j, (frame_file, mask_file) in enumerate(zip(frame_iter, mask_iter)):
            self._extract_image(frame_file, os.path.join(dest_root, f'frame_{j:04d}_gt.png'))
            self._extract_image(mask_file, os.path.join(dest_root, f'frame_{j:04d}_mask.png'))


    @abstractmethod
    def video_frame_files_iter(self, frame_video_name):
        pass


    @abstractmethod
    def video_mask_files_iter(self, mask_video_name):
        pass


    def _extract_image(self, image_file, save_path):
        image = Image.open(image_file)

        if not self.padder.needs_padding(image.size):
            # Copy the file itself if no padding is required
            image_file.seek(0)
            with open(save_path, 'wb') as f:
                f.write(image_file.read())
        else:
            # Create a padded version of the raw image
            image = Image.open(image_file)
            image_pad = self.padder.pad_image(image)
            image_pad.save(save_path)
