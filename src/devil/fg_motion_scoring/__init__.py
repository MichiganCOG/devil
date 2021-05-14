import numpy as np

from .. import warp_affine_mask, intersection_over_union, get_spaced_index_list
from ..scoring import AffineTransformComputer
from ...common_util.image import mask_percentage


class ForegroundMotionScorer(object):

    def score_foreground_motion(self, frames, fg_masks, **kwargs):
        """Scores the amount of foreground motion (i.e., foreground deformation and/or changes in pose) in the frames.

        :param frames: The frames in the video (list of HxWx3 uint8 NumPy arrays)
        :param fg_masks: The foreground masks associated with each frame (list of HxW bool NumPy array with False for
                         excluded pixels)
        :param kwargs: keyword arguments to use for debugging (dict)
        :return: float32
        """
        raise NotImplementedError


class ForegroundMotionIoUScorer(ForegroundMotionScorer):
    """Scores the amount of foreground motion (i.e., foreground deformation and/or changes in pose) in the frames.

    This algorithm scores foreground motion by aligning the foreground objects and computing the IoU. The inverse of the
    minimum IoU (1 / IoU) is taken to be the FG motion score.
    """

    def __init__(self, frame_spacing=None, num_sampled_frames=None, percentile=None, mask_percentage_threshold=None,
                 affine_transform_computer_args=None, **kwargs):
        self.frame_spacing = frame_spacing
        self.num_sampled_frames = num_sampled_frames
        self.percentile = percentile
        self.mask_percentage_threshold = mask_percentage_threshold
        self.affine_transform_computer = AffineTransformComputer(**affine_transform_computer_args)


    def score_foreground_motion(self, frames, fg_masks, **kwargs):
        """Scores the amount of foreground motion (i.e., foreground deformation and/or changes in pose) in the frames.

        :param frames: The frames in the video (list of HxWx3 uint8 NumPy arrays)
        :param fg_masks: The foreground masks associated with each frame (list of HxW bool NumPy array with False for
                         excluded pixels)
        :param kwargs: keyword arguments to use for debugging (dict)
        :return: float32
        """
        H, W, _ = frames[0].shape

        frame_indexes = get_spaced_index_list(len(frames), total=self.num_sampled_frames, spacing=self.frame_spacing)
        sampled_frames = [frames[x] for x in frame_indexes]
        sampled_masks = [fg_masks[x] for x in frame_indexes]

        scores = np.zeros((self.num_sampled_frames, self.num_sampled_frames))  # Store as matrix for easier debugging
        for i in range(self.num_sampled_frames):
            for j in range(self.num_sampled_frames):
                if i == j:
                    continue

                frame_i = sampled_frames[i]
                frame_j = sampled_frames[j]
                fg_mask_i = sampled_masks[i]
                fg_mask_j = sampled_masks[j]

                if min(mask_percentage(fg_mask_i), mask_percentage(fg_mask_j)) < self.mask_percentage_threshold:
                    scores[i, j] = np.nan
                    continue

                A = self.affine_transform_computer.compute_affine_transform(frame_i, frame_j, fg_mask_i, fg_mask_j,
                                                                            **kwargs)
                if A is None:
                    scores[i, j] = np.inf
                    continue

                fg_mask_i_warped = warp_affine_mask(fg_mask_i, A, (W, H))
                iou = intersection_over_union(fg_mask_i_warped, fg_mask_j)
                scores[i, j] = 1.0 - iou

        return np.percentile(scores[~np.isnan(scores)], self.percentile, interpolation='higher')
