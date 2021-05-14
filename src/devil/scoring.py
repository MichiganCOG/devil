import warnings

import cv2
import numpy as np

from . import warp_affine_mask, mask_check, compute_optical_flow


class FeatureMatchComputer(object):
    """A utility class for extracting feature descriptors and matching them across images."""

    def __init__(self, feature_type, max_num_features):
        if feature_type == 'orb':
            self.feature_extractor = cv2.ORB_create(max_num_features)
            self.feature_extractor_dtype = np.uint8
            self.matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
        elif feature_type == 'sift':
            self.feature_extractor = cv2.xfeatures2d.SIFT_create(max_num_features)
            self.feature_extractor_dtype = np.float32
            self.matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
        elif feature_type == 'surf':
            self.feature_extractor = cv2.xfeatures2d.SURF_create(max_num_features)
            self.feature_extractor_dtype = np.float32
            self.matcher = cv2.BFMatcher(normType=cv2.NORM_L2, crossCheck=True)
        else:
            raise ValueError(f'"{feature_type}" is not a supported feature type')


    def compute_features(self, image, mask):
        """Produce feature descriptors on the given image that lie within the given mask.

        :param image: The input image (HxWx3 uint8 NumPy array)
        :param mask: The mask of locations to include (HxW bool NumPy array w/ False for excluded pixels)
        :return:
            - keypoints: list of KeyPoint
            - descriptors: KxF NumPy uint8/float32 array
        """
        H, W, _ = image.shape
        if mask is None:
            mask = np.ones((H, W), dtype=np.bool)

        keypoints, descriptors = self.feature_extractor.detectAndCompute(image, None)
        if len(keypoints) == 0:
            # No normal keypoints found
            return [], np.empty((0, self.feature_extractor.descriptorSize()), dtype=self.feature_extractor_dtype)

        filtered_keypoints = []
        filtered_descriptors = []
        for keypoint, descriptor in zip(keypoints, descriptors):
            if mask_check(mask, *keypoint.pt):
                filtered_keypoints.append(keypoint)
                filtered_descriptors.append(descriptor)

        # Concatenate descriptors (producing a zero-rowed array if none were found)
        if len(filtered_keypoints) == 0:
            return [], np.empty((0, self.feature_extractor.descriptorSize()), dtype=self.feature_extractor_dtype)
        else:
            filtered_descriptors_np = np.stack(filtered_descriptors)
            return filtered_keypoints, filtered_descriptors_np


    def compute_matches(self, keypoints_a, features_a, keypoints_b, features_b):
        """Compute correspondences between the given feature descriptors from two images.

        :param keypoints_a: KeyPoint objects from image a (list of Keypoint)
        :param features_a: Feature descriptors from image a (KxF NumPy uint8/float32 array)
        :param keypoints_b: KeyPoint objects from image b (list of Keypoint)
        :param features_b: Feature descriptors from image b (KxF NumPy uint8/float32 array)
        :return:
            - matched_keypoints_a: The set of keypoints from a that were matched (K_m x 2 NumPy float32 array)
            - matched_keypoints_b: The set of keypoints from b that were matched (K_m x 2 NumPy float32 array)
        """

        matches = self.matcher.match(features_a, features_b)
        matched_keypoints_a = np.array([keypoints_a[match.queryIdx] for match in matches])
        matched_keypoints_b = np.array([keypoints_b[match.trainIdx] for match in matches])

        return matched_keypoints_a, matched_keypoints_b


class NormalizedImageNormScorer(object):
    """Computes the L<degree>-norm between two images, spatially normalized over locations that are included via the
    union of the given masks.

    More precisely, this computes the L<degree> norm between each corresponding pixel value, and takes the mean over
    them. The norms are only aggregated over unmasked pixels, i.e., pixels for which mask_a and mask_b are both True.
    """

    def __init__(self, degree):
        self.degree = degree


    def score_normalized_image_norm(self, img_a, img_b, mask_a=None, mask_b=None, **kwargs):
        """Computes the L<degree>-norm between two images, spatially normalized over locations that are included via the
        union of the given masks.

        More precisely, this computes the L<degree> norm between each corresponding pixel value, and takes the mean over
        them. The norms are only aggregated over unmasked pixels, i.e., pixels for which mask_a and mask_b are both True.

        :param img_a: The first image (HxWx3 uint8 NumPy array)
        :param img_b: The second image (HxWx3 uint8 NumPy array)
        :param mask_a: The mask on the first image (HxW bool NumPy array w/ False for excluded pixels)
        :param mask_b: The mask on the second image (HxW bool NumPy array w/ False for excluded pixels)
        :param degree: Indicates which p-norm to use (int)
        :param kwargs: keyword arguments to use for debugging (dict)
        :return: float32
        """
        # Get dimensions
        H, W, C = img_a.shape

        # Initialize masks if not provided as arguments
        if mask_a is None:
            mask_a = np.full((H, W), True, dtype=np.bool)
        if mask_b is None:
            mask_b = np.full((H, W), True, dtype=np.bool)

        # Reshape images to have one pixel value per row
        img_a_rows = img_a.reshape((H * W, C))
        img_b_rows = img_b.reshape((H * W, C))

        # Compute the norm for each row independently
        norm_rows = np.linalg.norm(img_a_rows - img_b_rows, ord=self.degree, axis=1)

        # Identify the locations that are unmasked in both images
        mask_union = mask_a * mask_b
        # If there is no overlap, return infinity (this tends to indicate that the images are not properly aligned)
        if np.sum(mask_union) == 0:
            return np.inf
        # Return the average pixel norm over unmasked pixels
        return np.sum(norm_rows * mask_union.flatten()) / np.sum(mask_union)


class AffineTransformComputer(object):
    """Computes a robust affine transformation matrix that aligns the input image to the target image.

    The mask arguments optionally specify areas to ignore in the alignment. Any keypoints that are found inside a
    masked-out region will be omitted when computing the alignment.
    True means keep the keypoint, False means discard the keypoint
    """

    def __init__(self, feature_matcher_computer_args, ransac_threshold):
        self.feature_match_computer = FeatureMatchComputer(**feature_matcher_computer_args)
        self.ransac_threshold = ransac_threshold


    def compute_affine_transform(self, img_input, img_target, mask_input=None, mask_target=None, **kwargs):
        """Computes a robust affine transformation matrix that aligns the input image to the target image.

        The mask arguments optionally specify areas to ignore in the alignment. Any keypoints that are found inside a
        masked-out region will be omitted when computing the alignment.
        True means keep the keypoint, False means discard the keypoint

        :param img_input: The input image to align to img_target (HxWx3 uint8 NumPy array)
        :param img_target: The target image that img_input will be aligned to (HxWx3 uint8 NumPy array)
        :param mask_input: The mask of input locations to include (HxW bool NumPy array w/ False for excluded pixels)
        :param mask_target: The mask of target locations to include (HxW bool NumPy array w/ False for excluded pixels)
        :return: An affine rotation matrix (2x3 float32 NumPy array)
        """

        keypoints_input, features_input = self.feature_match_computer.compute_features(img_input, mask_input)
        keypoints_target, features_target = self.feature_match_computer.compute_features(img_target, mask_target)

        if len(keypoints_input) < 1 or len(keypoints_target) < 1:
            warnings.warn("A is None because there are not enough keypoints for matching")
            return None

        matched_keypoints_target, matched_keypoints_input = self.feature_match_computer.compute_matches(
            keypoints_target, features_target, keypoints_input, features_input)
        if len(matched_keypoints_target) < 2:
            warnings.warn("A is None because fewer than two keypoints were successfully matched")
            return None

        # Convert keypoint object lists to keypoint location arrays
        matched_keypoints_target_np = np.array([kp.pt for kp in matched_keypoints_target])
        matched_keypoints_input_np = np.array([kp.pt for kp in matched_keypoints_input])

        A, is_inlier_np = cv2.estimateAffinePartial2D(matched_keypoints_input_np, matched_keypoints_target_np,
                                                      method=cv2.RANSAC, ransacReprojThreshold=self.ransac_threshold)

        # # Get keypoints that were used in the affine estimate (for debugging)
        # ransac_keypoints_target = matched_keypoints_target[is_inlier_np.squeeze() == 1]
        # ransac_keypoints_input = matched_keypoints_input[is_inlier_np.squeeze() == 1]

        return A


class AffineScoreScorer(object):
    """Computes a score between the target image and the input image warped to the target via an optimal affine
    transform.

    The score is computed by first finding the best affine transform from the input image to the target, warping the
    input image and mask to the target, and taking the normalized, masked image norm between them.
    """

    def __init__(self, affine_transform_computer_args, normalized_image_norm_scorer_args):
        self.affine_transform_computer = AffineTransformComputer(**affine_transform_computer_args)
        self.normalized_image_norm_scorer = NormalizedImageNormScorer(**normalized_image_norm_scorer_args)


    def score_affine_score(self, img_input, img_target, mask_input=None, mask_target=None, **kwargs):
        """Computes a score between the target image and the input image warped to the target via an optimal affine
        transform.

        The score is computed by first finding the best affine transform from the input image to the target, warping the
        input image and mask to the target, and taking the normalized, masked image norm between them.

        :param img_input: The input image to align to img_target (HxWx3 uint8 NumPy array)
        :param img_target: The target image that img_input will be aligned to (HxWx3 uint8 NumPy array)
        :param mask_input: The mask of input locations to include (HxW bool NumPy array w/ False for excluded pixels)
        :param mask_target: The mask of target locations to include (HxW bool NumPy array w/ False for excluded pixels)
        :param kwargs: keyword arguments to use for debugging (dict)
        :return: float32
        """
        rows, cols, _ = img_target.shape

        # Initialize default masks if not given
        if mask_input is None:
            mask_input = np.full((rows, cols), True, dtype=np.bool)

        A = self.affine_transform_computer.compute_affine_transform(
            img_input, img_target, mask_input, mask_target, **kwargs)
        if A is None:
            return np.inf

        img_input_warped = cv2.warpAffine(img_input, A, (cols, rows))
        mask_input_warped = warp_affine_mask(mask_input, A, (cols, rows))

        score = self.normalized_image_norm_scorer.score_normalized_image_norm(
            img_target, img_input_warped, mask_a=mask_target, mask_b=mask_input_warped, **kwargs)

        return score


class LowCameraMotionSegmentsComputer(object):
    """Returns a list of video segments where, in each segment, camera motion is sufficiently low.

    A segment is proposed by iteratively increasing its length by `spacing` as long as the first and last frames in the
    segment have a sufficiently low affine score. A new segment proposal is started at the latest observed frame once
    the above condition fails.
    """

    def __init__(self, threshold, min_length, affine_score_scorer_args):
        self.affine_score_scorer = AffineScoreScorer(**affine_score_scorer_args)
        self.threshold = threshold
        self.min_length = min_length


    def compute_low_camera_motion_segments(self, frames, masks):
        """Returns a list of video segments where, in each segment, camera motion is sufficiently low.

        A segment is proposed by iteratively increasing its length by `spacing` as long as the first and last frames in the
        segment have a sufficiently low affine score. A new segment proposal is started at the latest observed frame once
        the above condition fails.

        :param frames: The frames to propose segments over (list of HxWx3 uint8 NumPy arrays)
        :param masks: The masks associated with each frame (list of HxW bool NumPy array with False for excluded pixels)
        :param threshold: The maximum acceptable affine score that the first and last frame in a segment can have (float)
        :param min_length: The minimum length of a proposed segment (inclusive) (int)
        :return: List of tuples indicating the first and last frame of the segment (inclusive)
        """
        segments = []
        # Indexes for the current segment candidate
        i = 0
        j = 0

        # Loop invariants:
        # - frames i and j always have an affine score that's below the threshold
        # - frames i, j, and k are always valid frames
        # Termination condition:
        # - frames i and j always have an affine score that's below the threshold
        # - frames i and j are valid frames, but frame k is not; k == len(frames)
        for k in range(1, len(frames)):
            score = self.affine_score_scorer.score_affine_score(frames[i], frames[k], masks[i], masks[k])
            if score < self.threshold:
                # Expand the potential segment by one
                j += 1
            else:
                # Add the segment if it's long enough
                if j-i+1 >= self.min_length:
                    segments.append((i, j))
                # Reset the potential segment starting from index k
                i = k
                j = k

        # Add the segment if it's long enough
        # This cannot duplicate a segment from the loop because in the loop, a segment can only be added if k is a valid
        # frame index; here, k is NOT a valid frame index
        if j-i+1 >= self.min_length:
            segments.append((i, j))

        if len(segments) == 0:
            warnings.warn('No low camera motion segments were proposed')

        return segments


class AlignFlowScorer(object):
    """Returns a flow score over the given frames by aligning sequential pairs with an affine transform and aggregating
    optical flow over the result.

    For each sequential pair of frames in the input sequence, the first frame and mask is aligned to the second frame
    via an affine transform. The associated flow score is then obtained by taking the average magnitude of every
    individual optical flow vector (i.e., normalized over all pixel locations and time steps).
    """

    def __init__(self, affine_transform_computer_args):
        self.affine_transform_computer = AffineTransformComputer(**affine_transform_computer_args)


    def score_align_flow(self, frames, masks):
        """Returns a flow score over the given frames by aligning sequential pairs with an affine transform and aggregating
        optical flow over the result.

        For each sequential pair of frames in the input sequence, the first frame and mask is aligned to the second frame
        via an affine transform. The associated flow score is then obtained by taking the average magnitude of every
        individual optical flow vector (i.e., normalized over all pixel locations and time steps).

        :param frames: The frames to align and compute optical flow over (list of HxWx3 uint8 NumPy arrays)
        :param masks: The masks associated with each frame (list of HxW bool NumPy array with False for excluded pixels)
        :return: float32
        """
        flow_norms_list = []
        for (frame_input, frame_target, mask_input, mask_target) in zip(frames[:-1], frames[1:], masks[:-1], masks[1:]):
            A = self.affine_transform_computer.compute_affine_transform(frame_input, frame_target,
                                                                        mask_input, mask_target)
            H, W, _ = frame_target.shape
            if A is not None:
                img_warp = cv2.warpAffine(frame_input, A, (W, H))
                mask_warp = warp_affine_mask(mask_input, A, (W, H))
                flow = compute_optical_flow(img_warp, frame_target, mask_input=mask_warp, mask_target=mask_target)
                # Reshape flow such that each row corresponds to one location on the image
                flow_rows = flow.reshape((H*W, 2))
                # Compute the norm for each row independently
                cur_flow_norms = np.linalg.norm(flow_rows, ord=2, axis=1)
                flow_norms_list.append(cur_flow_norms)

        # Compute the mean over flow norms, excluding NaN
        mean_flow_norm = np.nanmean(np.concatenate(flow_norms_list))
        return mean_flow_norm.astype(float)


class CannyEdgeComputer(object):
    """Extracts Canny edges from a masked image."""

    def __init__(self, threshold1, threshold2):
        """Constructor

        :param threshold1: float
        :param threshold2: float
        """
        self.threshold1 = threshold1
        self.threshold2 = threshold2


    def compute_canny_edges(self, image, mask):
        """Extracts Canny edges from a masked image.

        :param image: HxWx3 uint8 NumPy array
        :param mask: The mask associated with the image (HxW bool NumPy array with False for excluded pixels)
        :return: HxW bool NumPy array
        """
        edges_uint8 = cv2.Canny(image, self.threshold1, self.threshold2)
        masked_edges_uint8 = mask * edges_uint8
        masked_edges = masked_edges_uint8.astype(np.bool)

        return masked_edges
