import numpy as np

from .. import get_spaced_index_list
from ..features import BinaryForegroundMaskComputer


class ForegroundSizeScorer(object):
    """Scores the size of the foreground as a percentage of the average mask size over the entire frame area."""

    def score_foreground_size(self, fg_masks):
        """Scores the size of the foreground as a percentage of the average mask size over the entire frame area.

        :param fg_masks: The foreground masks associated with each frame (list of HxW bool NumPy arrays with False for
                         excluded pixels)
        :return: float
        """
        fg_percentages = [fg_mask.sum() / fg_mask.size for fg_mask in fg_masks]

        return np.mean(fg_percentages)


class AutomaticForegroundSizeScorer(object):
    """Detects all foreground objects and computes their total mask size averaged over the entire frame area."""

    def __init__(self, frame_spacing=None, num_sampled_frames=None):
        """Constructor

        :param frame_spacing: The spacing between two consecutive sampled frames (int)
        :param num_sampled_frames: The total number of frames to sample throughout the video (int)
        """
        self.frame_spacing = frame_spacing
        self.num_sampled_frames = num_sampled_frames
        self.foreground_size_scorer = ForegroundSizeScorer()
        self.binary_foreground_mask_computer = BinaryForegroundMaskComputer()


    def score_foreground_size(self, frames, **kwargs):
        """Detects all foreground objects and computes their total mask size averaged over the entire frame area.

        :param frames: The video frames over which the compute FG masks (list of HxWxC uint8 NumPy arrays)
        :param kwargs: Additional keyword arguments to use for debugging (dict)
        :return: float
        """
        frame_indexes = get_spaced_index_list(len(frames), total=self.num_sampled_frames, spacing=self.frame_spacing)
        masks = [self.binary_foreground_mask_computer.compute_binary_foreground_mask(frames[i], **kwargs)
                 for i in frame_indexes]

        return self.foreground_size_scorer.score_foreground_size(masks)
