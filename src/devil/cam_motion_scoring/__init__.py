import numpy as np

from ..scoring import AffineTransformComputer
from ...devil import warp_affine_mask, get_spaced_index_list


class CameraMotionScorer(object):

    def score_camera_motion(self, frames, fg_masks, **kwargs):
        """Scores the amount of camera motion that is present in the given frames.

        :param frames: The frames in the video (list of HxWx3 uint8 NumPy arrays)
        :param fg_masks: The foreground masks associated with each frame (list of HxW bool NumPy array with False for
                         excluded pixels)
        :param kwargs: Keyword arguments to pass to _score_masked_scene_motion for debugging purposes (dict)
        :return: float32
        """
        raise NotImplementedError


class CameraMotionFrameDisplacementScorer(CameraMotionScorer):
    """Scores the amount of camera motion that is present in the given frames.

    This algorithm scores camera motion by aligning frames via affine transformations, and measuring how much the warped
    frame has been displaced.
    """

    def __init__(self, frame_spacing=None, num_sampled_frames=None, percentile=None,
                 affine_transform_computer_args=None):
        """Constructor

        :param frame_spacing: The spacing between two consecutive sampled frames (int)
        :param num_sampled_frames: The total number of frames to sample throughout the video (int)
        :param percentile: The rank among all frame-pair scores to keep
        :param affine_transform_computer_args: The arguments to be passed to the AffineTransformComputer
        """
        self.frame_spacing = frame_spacing
        self.num_sampled_frames = num_sampled_frames
        self.percentile = percentile
        self.affine_transform_computer = AffineTransformComputer(**affine_transform_computer_args)


    def score_camera_motion(self, frames, fg_masks, **kwargs):
        """Scores the amount of camera motion that is present in the given frames.

        :param frames: The frames in the video (list of HxWx3 uint8 NumPy arrays)
        :param fg_masks: The foreground masks associated with each frame (list of HxW bool NumPy array with False for
                         excluded pixels)
        :param kwargs: Keyword arguments to pass to _score_masked_scene_motion for debugging purposes (dict)
        :return: float32
        """

        # Convert foreground masks into background masks
        bg_masks = []
        for fg_mask in fg_masks:
            bg_mask = np.full(frames[0].shape[:2], True, np.bool) if fg_mask is None else np.where(fg_mask, False, True)
            bg_masks.append(bg_mask)

        frame_indexes = get_spaced_index_list(len(frames), total=self.num_sampled_frames, spacing=self.frame_spacing)
        sampled_frames = [frames[x] for x in frame_indexes]
        sampled_bg_masks = [bg_masks[x] for x in frame_indexes]

        scores = np.empty((self.num_sampled_frames, self.num_sampled_frames))  # Store as matrix for easier debugging
        for i in range(self.num_sampled_frames):
            for j in range(self.num_sampled_frames):
                # Skip identical frames
                if i == j:
                    scores[i][j] = 0
                    continue

                # Compute the transformation from frame i to frame j
                A = self.affine_transform_computer.compute_affine_transform(
                    sampled_frames[i], sampled_frames[j], sampled_bg_masks[i], sampled_bg_masks[j], **kwargs)
                if A is None:
                    # Alignment failed completely, so the camera probably moved a lot
                    scores[i][j] = np.inf
                    continue

                # Warp the FG mask at time i to time j
                rows, cols, _ = sampled_frames[i].shape
                full_mask = np.ones((rows, cols), dtype=np.bool)
                frame_region_warped = warp_affine_mask(full_mask, A, (cols, rows))
                # Invert the warped full-frame mask because more inverted pixels means more camera motion
                non_frame_region_warped = np.where(frame_region_warped, False, True)
                scores[i][j] = non_frame_region_warped.sum() / non_frame_region_warped.size

        return np.percentile(scores, self.percentile, interpolation='higher')
