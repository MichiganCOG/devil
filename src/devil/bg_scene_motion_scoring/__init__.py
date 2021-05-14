import warnings

import numpy as np

from ..scoring import AlignFlowScorer, LowCameraMotionSegmentsComputer


class BackgroundSceneMotionScorer(object):

    def score_background_scene_motion(self, frames, fg_masks):
        """Scores the amount of scenic motion in a video.

        :param frames: The frames in the video (list of HxWx3 uint8 NumPy arrays)
        :param fg_masks: The foreground masks associated with each frame (list of HxW bool NumPy arrays with False for
                         excluded pixels)
        :return: float
        """
        raise NotImplementedError


class BackgroundSceneMotionFlowScorer(BackgroundSceneMotionScorer):
    """Scores the amount of scenic motion by aligning segments in the video and measuring average optical flow.

    This algorithm works by identifying video segments whose background pixels can be well-aligned with affine
    transforms. Within those segments, all frames are co-aligned with affine transforms, and then the average optical
    flow within the background is computed.
    """

    def __init__(self, align_threshold, min_segment_length, affine_flow_scorer_args,
                 low_camera_motion_segments_computer_args):
        """Constructor

        :param align_threshold: The maximum tolerated alignment score between frames of a segment (float)
        :param min_segment_length: The minimum number of frames allowed in a segment
        """
        self.align_threshold = align_threshold
        self.min_segment_length = min_segment_length
        self.align_flow_scorer = AlignFlowScorer(**affine_flow_scorer_args)
        self.low_camera_motion_segments_computer = LowCameraMotionSegmentsComputer(**low_camera_motion_segments_computer_args)


    def score_background_scene_motion(self, frames, fg_masks):
        """Scores the amount of scenic motion by aligning segments in the video and measuring average optical flow.

        This algorithm works by identifying video segments whose background pixels can be well-aligned with affine
        transforms. Within those segments, all frames are co-aligned with affine transforms, and then the average
        optical flow within the background is computed.

        :param frames: The frames in the video (list of HxWx3 uint8 NumPy arrays)
        :param fg_masks: The foreground masks associated with each frame (list of HxW bool NumPy arrays with False for
                         excluded pixels)
        :return: float
        """
        # Convert foreground masks into background masks
        bg_masks = []
        for fg_mask in fg_masks:
            bg_mask = np.full(frames[0].shape[:2], True, np.bool) if fg_mask is None else np.where(fg_mask, False, True)
            bg_masks.append(bg_mask)

        segments = self.low_camera_motion_segments_computer.compute_low_camera_motion_segments(frames, bg_masks)
        if len(segments) == 0:
            warnings.warn('No proposed segments were given')
            return np.nan

        final_score = 0
        total_seg_len = 0
        for seg in segments:
            start, end = seg[0], seg[1]
            total_seg_len += (end - start + 1)
            final_score += (end - start + 1) * self.align_flow_scorer.score_align_flow(
                frames[start:end + 1], bg_masks[start:end + 1])

        final_score /= total_seg_len
        return final_score
