import os
import pickle
import unittest

import cv2
import numpy as np
import numpy.testing as npt
import torch
from PIL import Image

from ...common_util.global_vars import PROJ_DIR
from ...common_util.image import image_path_to_numpy
from ...devil import compute_optical_flow, get_spaced_index_list, hausdorff_distance, compute_object_instances
from ...devil.config import get_default_config, namespace_to_dict
from ...devil.scoring import NormalizedImageNormScorer, AffineTransformComputer, AffineScoreScorer, \
    LowCameraMotionSegmentsComputer

TEST_DATA_ROOT = os.path.join(PROJ_DIR, 'test-data')


class DevilTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        frames = []
        masks = []

        for i in range(6):
            img = np.array(Image.open(os.path.join(TEST_DATA_ROOT, 'bear', '0000' + str(i) + '.jpg'))).astype(np.uint8)
            mask = cv2.imread(os.path.join(TEST_DATA_ROOT, 'mask', '0000' + str(i) + '.png'), 0)
            mask = np.where(mask > 0, True, False)
            frames.append(img)
            masks.append(mask)

        cls.frames = frames
        cls.masks = masks

        cls.default_config_dict = namespace_to_dict(get_default_config())

        cls.normalized_image_norm_scorer = NormalizedImageNormScorer(
            **cls.default_config_dict['NormalizedImageNormScorer'])
        cls.affine_transform_computer = AffineTransformComputer(**cls.default_config_dict['AffineTransformComputer'])
        cls.affine_score_scorer = AffineScoreScorer(**cls.default_config_dict['AffineScoreScorer'])
        cls.low_camera_motion_segments_computer = LowCameraMotionSegmentsComputer(
            **cls.default_config_dict['LowCameraMotionSegmentsComputer'])

    def assert_similar_affine_alignment(self, target_path, input_path):
        """Check that compute_affine_transform finds a good affine alignment between the images located at the two
        given image paths.

        :param target_path: File path to the target image that the aligned input image should resemble
        :param input_path: File path to the input image that should resemble the target image when aligned
        """
        image_target = cv2.imread(target_path)
        image_input = cv2.imread(input_path)

        # Make masks
        mask_target = np.where(cv2.cvtColor(image_target, cv2.COLOR_BGR2GRAY) > 0, True, False)
        mask_input = np.where(cv2.cvtColor(image_input, cv2.COLOR_BGR2GRAY) > 0, True, False)

        # Compute the affine matrix
        A = self.affine_transform_computer.compute_affine_transform(image_target, image_input, mask_target, mask_input)

        # Warp the image
        warped = cv2.warpAffine(image_target, A, (image_target.shape[1], image_target.shape[0]))

        # Compare with the target image
        e_move = np.mean((image_input - warped) ** 2)
        self.assertTrue(e_move < 1)

    def assert_torch_equal(self, a, b):
        self.assertTrue(torch.equal(a, b))

    def test_compute_affine_transform_identity(self):
        """Check that compute_affine_transform finds a good affine alignment between an image and itself."""
        self.assert_similar_affine_alignment(os.path.join(TEST_DATA_ROOT, 'affine/original.png'),
                                             os.path.join(TEST_DATA_ROOT, 'affine/original.png'))

    def test_compute_affine_transform_move(self):
        """Check that compute_affine_transform finds a good affine alignment between an image and version that has
        been translated to the right."""
        self.assert_similar_affine_alignment(os.path.join(TEST_DATA_ROOT, 'affine/original.png'),
                                             os.path.join(TEST_DATA_ROOT, 'affine/move.png'))

    def test_compute_affine_transform_rot(self):
        """Check that compute_affine_transform finds a good affine alignment between an image and version that has
        been rotated."""
        self.assert_similar_affine_alignment(os.path.join(TEST_DATA_ROOT, 'affine/original.png'),
                                             os.path.join(TEST_DATA_ROOT, 'affine/rot.png'))

    def test_compute_affine_transform_rot_move(self):
        """Check that compute_affine_transform finds a good affine alignment between an image and version that has
        been rotated AND translated to the right."""
        self.assert_similar_affine_alignment(os.path.join(TEST_DATA_ROOT, 'affine/original.png'),
                                             os.path.join(TEST_DATA_ROOT, 'affine/rot_move.png'))

    def test_compute_affine_transform_no_keypoints(self):
        """Check that compute_affine_transform returns None when there are no keypoints in one of the images."""
        img = cv2.imread(os.path.join(TEST_DATA_ROOT, 'affine/original.png'))

        # Create a black background where no keypoints can be captured
        bad_img = np.zeros_like(img)

        A = self.affine_transform_computer.compute_affine_transform(img, bad_img)
        self.assertEqual(A, None)

    def test_compute_affine_transform_too_few_keypoints(self):
        """Check that compute_affine_transform returns None when there are not enough sufficiently matched keypoints
        between the two images."""
        img1 = cv2.imread(os.path.join(TEST_DATA_ROOT, 'affine/original.png'))

        # Input a picture where few keypoints can be extracted
        img2 = cv2.imread(os.path.join(TEST_DATA_ROOT, 'affine/bad.png'))

        mask1 = cv2.imread(os.path.join(TEST_DATA_ROOT, 'affine/original_mask.png'), 0)
        mask2 = cv2.imread(os.path.join(TEST_DATA_ROOT, 'affine/bad_mask.png'), 0)

        object_mask1 = np.where(mask1 == 0, False, True)
        object_mask2 = np.where(mask2 == 0, False, True)

        object_mask2[329, 641] = False

        A = self.affine_transform_computer.compute_affine_transform(img1, img2, object_mask1, object_mask2)
        self.assertIsNone(A)

    def test_compute_normalized_image_norm(self):
        img_a = np.zeros((5, 5, 3))
        img_b = 5 * np.ones((5, 5, 3))
        self.assertEqual(self.normalized_image_norm_scorer.score_normalized_image_norm(img_a, img_b),
                         np.linalg.norm([5, 5, 5], ord=1))

    def test_compute_normalized_image_norm_masked(self):
        img_a = np.zeros((6, 6, 3))
        img_b = np.empty((6, 6, 3))
        # Fill image with 1-4 in top-left, top-right, bottom-left, and bottom-right quadrants
        img_b[:3, :3] = 1
        img_b[:3, 3:] = 2
        img_b[3:, :3] = 3
        img_b[3:, 3:] = 4
        # Create masks whose unmasked areas intersect over 1 TL pixel, 2 TR pixels, 1 BL pixel, and 2 BR pixels
        mask_a = np.zeros((6, 6), dtype=np.bool)
        mask_a[2:4, 2:] = True
        mask_b = np.zeros((6, 6), dtype=np.bool)
        mask_b[1:5, 1:5] = True
        result = self.normalized_image_norm_scorer.score_normalized_image_norm(
            img_a, img_b, mask_a=mask_a, mask_b=mask_b)
        # Take average norm over expected unmasked values
        expected = (1 * np.linalg.norm([1, 1, 1], ord=1)
                    + 2 * np.linalg.norm([2, 2, 2], ord=1)
                    + 1 * np.linalg.norm([3, 3, 3], ord=1)
                    + 2 * np.linalg.norm([4, 4, 4], ord=1)) / 6
        self.assertEqual(result, expected)

    def test_compute_affine_score_identity(self):
        img1 = self.frames[0].copy()

        # Check case with no mask input
        score = self.affine_score_scorer.score_affine_score(img1, img1)
        self.assertEqual(score, 0.0)

    def test_compute_affine_score_no_keypoints(self):
        img1 = self.frames[0].copy()
        black = np.zeros_like(img1)

        # Check case with no mask input
        score = self.affine_score_scorer.score_affine_score(img1, black)
        self.assertEqual(score, np.inf)

    def test_compute_affine_score_too_few_keypoints(self):
        img1 = cv2.imread(os.path.join(TEST_DATA_ROOT, 'affine/original.png'))
        img2 = cv2.imread(os.path.join(TEST_DATA_ROOT, 'affine/bad.png'))

        mask1 = cv2.imread(os.path.join(TEST_DATA_ROOT, 'affine/original_mask.png'), 0)
        mask2 = cv2.imread(os.path.join(TEST_DATA_ROOT, 'affine/bad_mask.png'), 0)

        # Focus on object
        object_mask1 = np.where(mask1 == 0, False, True)
        object_mask2 = np.where(mask2 == 0, False, True)

        object_mask2[329, 641] = False

        score = self.affine_score_scorer.score_affine_score(img1, img2, object_mask1, object_mask2)
        self.assertEqual(score, np.inf)

    def test_compute_affine_score(self):
        img1, img2 = self.frames[0].copy(), self.frames[1].copy()
        mask1, mask2 = self.masks[0].copy(), self.masks[1].copy()

        # Check case with no mask input
        score = self.affine_score_scorer.score_affine_score(img1, img2)
        self.assertIsInstance(score, float)
        # Check case with both mask inputs
        score = self.affine_score_scorer.score_affine_score(img1, img2, mask1, mask2)
        self.assertIsInstance(score, float)
        # Check case with one mask input
        score = self.affine_score_scorer.score_affine_score(img1, img2, mask_input=mask1)
        self.assertIsInstance(score, float)
        score = self.affine_score_scorer.score_affine_score(img1, img2, mask_target=mask2)
        self.assertIsInstance(score, float)

    def test_compute_optical_flow(self):
        self.assertTrue(torch.cuda.is_available(), 'This test requires a GPU to run')

        # check the flow generated by demo.py (written by RAFT's author) and the flow from __init__.py
        for i in range(5):
            img1 = self.frames[i].copy()
            img2 = self.frames[i+1].copy()

            # Compute flow from our wrapper function, and load flow from reference implementation
            flo_ours = compute_optical_flow(img1, img2)
            with open(os.path.join(TEST_DATA_ROOT, 'flow', 'flow0' + str(i) + '.npy'), 'rb') as f:
                flo_ref = np.load(f)

            # Convert reference optical flow grid to sampling grid
            H, W, _ = flo_ours.shape
            uu, vv = np.meshgrid(np.arange(W), np.arange(H))
            sample_loc_u_ref = uu + flo_ref[:, :, 0]
            sample_loc_v_ref = vv + flo_ref[:, :, 1]
            # Determine valid sampling locations for horiz and vert components separately, then combine
            valid_u_ref = np.greater(sample_loc_u_ref, 0) * np.less(sample_loc_u_ref, W-1)
            valid_v_ref = np.greater(sample_loc_v_ref, 0) * np.less(sample_loc_v_ref, H-1)
            valid_mask_ref = valid_u_ref * valid_v_ref

            # Determine which locations from our function were NaN-ified, and compare with reference flow
            valid_mask_ours = np.logical_not(np.isnan(flo_ours[:, :, 0] * flo_ours[:, :, 1]))
            self.assertTrue(np.array_equal(valid_mask_ours, valid_mask_ref))

            # Check that all valid flow values match between ours and the reference
            flo2_masked = np.where(valid_mask_ref[:, :, np.newaxis], flo_ref, np.nan)
            self.assertEqual(npt.assert_almost_equal(flo_ours, flo2_masked), None)

    def test_get_spaced_index_list(self):
        self.assertEqual(get_spaced_index_list(10, spacing=3), [0, 3, 6, 9])
        self.assertEqual(get_spaced_index_list(9, total=3), [0, 4, 8])
        self.assertEqual(get_spaced_index_list(9, spacing=3), [0, 3, 6])

        with self.assertRaises(ValueError):
            get_spaced_index_list(10, total=5, spacing=3)

    def test_get_low_camera_motion_segments_still_frames(self):
        """Check that the function segments a video containing multiple sequences of repeated frames (AAAABBBBCCCC)."""
        lcmc = LowCameraMotionSegmentsComputer(10, 3, self.default_config_dict['AffineScoreScorer'])

        frame_a = cv2.imread(os.path.join(TEST_DATA_ROOT, 'boat/00000.jpg'))
        frame_b = cv2.imread(os.path.join(TEST_DATA_ROOT, 'crossing/00000.jpg'))
        frame_c = cv2.imread(os.path.join(TEST_DATA_ROOT, 'elephant/00000.jpg'))
        H, W, _ = frame_a.shape
        mask = np.ones((H, W), dtype=np.bool)

        frames = 4 * [frame_a] + 4 * [frame_b] + 4 * [frame_c]
        masks = 12 * [mask]

        segments = lcmc.compute_low_camera_motion_segments(frames, masks)
        self.assertCountEqual(segments, [(0, 3), (4, 7), (8, 11)])

    def test_get_low_camera_motion_segments_short_segments(self):
        """Check that the function does not propose short segments (e.g., AAABBCCCC should produce AAA and CCCC)."""
        lcmc = LowCameraMotionSegmentsComputer(10, 3, self.default_config_dict['AffineScoreScorer'])

        frame_a = cv2.imread(os.path.join(TEST_DATA_ROOT, 'boat/00000.jpg'))
        frame_b = cv2.imread(os.path.join(TEST_DATA_ROOT, 'crossing/00000.jpg'))
        frame_c = cv2.imread(os.path.join(TEST_DATA_ROOT, 'elephant/00000.jpg'))
        H, W, _ = frame_a.shape
        mask = np.ones((H, W), dtype=np.bool)

        frames = 3 * [frame_a] + 2 * [frame_b] + 4 * [frame_c]
        masks = 9 * [mask]

        segments = lcmc.compute_low_camera_motion_segments(frames, masks)
        self.assertCountEqual(segments, [(0, 2), (5, 8)])

    def test_get_low_camera_motion_segments_short_segment_at_end(self):
        """Check that the function does not propose short segments at the end (e.g., AAAABB should produce AAAA)."""
        lcmc = LowCameraMotionSegmentsComputer(10, 3, self.default_config_dict['AffineScoreScorer'])

        frame_a = cv2.imread(os.path.join(TEST_DATA_ROOT, 'boat/00000.jpg'))
        frame_b = cv2.imread(os.path.join(TEST_DATA_ROOT, 'crossing/00000.jpg'))
        H, W, _ = frame_a.shape
        mask = np.ones((H, W), dtype=np.bool)

        frames = 4 * [frame_a] + 2 * [frame_b]
        masks = 6 * [mask]

        segments = lcmc.compute_low_camera_motion_segments(frames, masks)
        self.assertCountEqual(segments, [(0, 3)])

    def test_get_low_camera_motion_segments_tolerance(self):
        """Check that the function proposes with some tolerance for slightly different frames (e.g., A1 A2 A3 B1 B2 B3
        should yield (0, 2) and (3, 5)."""

        frame_as = [cv2.imread(os.path.join(TEST_DATA_ROOT, 'boat/{:05d}.jpg').format(x)) for x in range(3)]
        frame_bs = [cv2.imread(os.path.join(TEST_DATA_ROOT, 'crossing/{:05d}.jpg').format(x)) for x in range(3)]
        frames = frame_as + frame_bs
        H, W, _ = frame_as[0].shape
        masks = [np.ones((H, W), dtype=np.bool) for _ in range(6)]

        # Find maximum affine score within each desired segment (and set as lower bound for the segment threshold)
        pairs_to_test = [(0, 1), (0, 2), (3, 4), (3, 5)]
        score_lb = np.max([self.affine_score_scorer.score_affine_score(frames[i], frames[j]) for i, j in pairs_to_test])
        # Compute the affine score across the "shot cut" (the boundary of the two segments; upper bound for threshold)
        score_ub = self.affine_score_scorer.score_affine_score(frames[0], frames[3])
        # Check that the lower and upper bounds are appropriately ranked; otherwise, the two video segments are too
        # similar!
        self.assertGreater(score_ub, score_lb)
        # Set the threshold to be in between the two bounds
        threshold = (score_lb + score_ub) / 2

        lcmc = LowCameraMotionSegmentsComputer(threshold, 3, self.default_config_dict['AffineScoreScorer'])
        segments = lcmc.compute_low_camera_motion_segments(frames, masks)
        self.assertCountEqual(segments, [(0, 2), (3, 5)])

    def test_hausdorff_distance(self):
        """Check that the example from https://cs.stackexchange.com/a/118020 is computed correctly."""
        X = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.bool)

        Y = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.bool)

        d1, _, _ = hausdorff_distance(X, Y, dist_type=cv2.DIST_L1)
        self.assertEqual(d1, 3)
        d2, _, _ = hausdorff_distance(Y, X, dist_type=cv2.DIST_L1)
        self.assertEqual(d2, 3)

    def test_hausdorff_distance_no_points(self):
        """Check that an empty point set produces a NaN distance."""
        X = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ], dtype=np.bool)

        Y = np.zeros_like(X)

        d1, _, _ = hausdorff_distance(X, Y)
        self.assertTrue(np.isnan(d1))
        d2, _, _ = hausdorff_distance(Y, X)
        self.assertTrue(np.isnan(d2))

    def test_compute_object_instances(self):
        """Test that compute_object_instances produces the same result as the original Detectron2 code."""
        pkl_path = os.path.join(TEST_DATA_ROOT, 'detectron2', 'dog-predictions.pkl')
        img_path = os.path.join(TEST_DATA_ROOT, 'detectron2', 'dog.jpg')

        gt_instances = pickle.load(open(pkl_path, 'rb'))['instances']
        devil_instances = compute_object_instances(image_path_to_numpy(img_path))

        self.assert_torch_equal(gt_instances.pred_boxes.tensor, gt_instances.pred_boxes.tensor)
        self.assert_torch_equal(gt_instances.scores, devil_instances.scores)
        self.assert_torch_equal(gt_instances.pred_classes, devil_instances.pred_classes)
        self.assert_torch_equal(gt_instances.pred_masks, devil_instances.pred_masks)
