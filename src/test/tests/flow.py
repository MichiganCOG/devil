import unittest

import numpy as np
import torch

from ...common_util.flow import warp_flow


class FlowTests(unittest.TestCase):

    def test_warp_flow_identity(self):
        """Check that an image warped with zero optical flow returns itself."""
        torch.manual_seed(123)
        np.random.seed(123)

        image = torch.randint(0, 256, (3, 500, 500), dtype=torch.float32) / 255
        flow = torch.zeros((2, 500, 500))
        warped = warp_flow(image, flow)

        self.assertTrue(torch.allclose(image, warped, atol=1e-3))


    def test_warp_flow_horiz_shift(self):
        """Check that an image with horizontal optical flow is warped accordingly."""
        torch.manual_seed(123)
        np.random.seed(123)

        # How much to shift
        shift = 10

        image = torch.randint(0, 256, (3, 500, 500), dtype=torch.float32) / 255
        # Construct flow corresponding to going `shift` pixels to the right
        flow_u = shift * torch.ones((1, 500, 500))
        flow_v = torch.zeros((1, 500, 500))
        flow = torch.cat((flow_u, flow_v))
        warped = warp_flow(image, flow)

        # Crop matching parts of images and compare
        warped_comp = warped[:, :, :500-shift]
        image_comp = image[:, :, shift:]
        self.assertTrue(torch.allclose(image_comp, warped_comp, atol=1e-3))

        # Check that invalid sampled locations lead to nan in the result
        warped_invalid = warped[:, :, 500-shift:]
        self.assertTrue(torch.all(torch.isnan(warped_invalid)))


    def test_warp_flow_vert_shift(self):
        """Check that an image with vertical optical flow is warped accordingly."""
        torch.manual_seed(123)
        np.random.seed(123)

        # How much to shift
        shift = 10

        image = torch.randint(0, 256, (3, 500, 500), dtype=torch.float32) / 255
        # Construct flow corresponding to going `shift` pixels down
        flow_u = torch.zeros((1, 500, 500))
        flow_v = shift * torch.ones((1, 500, 500))
        flow = torch.cat((flow_u, flow_v))
        warped = warp_flow(image, flow)

        # Crop matching parts of images and compare
        warped_comp = warped[:, :500-shift, :]
        image_comp = image[:, shift:, :]
        self.assertTrue(torch.allclose(image_comp, warped_comp, atol=1e-3))

        # Check that invalid sampled locations lead to nan in the result
        warped_invalid = warped[:, 500-shift:, :]
        self.assertTrue(torch.all(torch.isnan(warped_invalid)))


    def test_warp_flow_nan(self):
        """Check that the image is also nan iff the optical flow is nan (whenever flow lands inside the image)."""
        torch.manual_seed(123)
        np.random.seed(123)

        image = torch.randint(0, 256, (3, 500, 500), dtype=torch.float32) / 255
        flow = torch.zeros((2, 500, 500))
        flow[:, :250, :] = np.nan
        warped = warp_flow(image, flow)

        # Check that the result was nan wherever optical flow was nan
        self.assertTrue(torch.all(torch.isnan(warped[:, :250, :])))
        # Check that all other locations were not nan
        self.assertTrue(torch.all(torch.isnan(warped[:, 250:, :]) == False))
