import argparse
import math
import os
from itertools import combinations
from operator import xor

import cv2
import detectron2.config
import detectron2.data.detection_utils
import detectron2.engine.defaults
import numpy as np
import torch
from detectron2.data import MetadataCatalog

from ..common_util.flow import warp_flow
from ..common_util.global_vars import PROJ_DIR
from ..raft.raft import RAFT
from ..raft.utils.utils import InputPadder

_DEFAULT_DEVICE = torch.device('cuda:0')
_FLOW_MODEL = None
_DETECTRON2_MODEL = None
_DETECTRON2_PANOPTIC_MODEL = None
_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))


def _initialize_flow_model():
    """Loads the optical flow model onto the GPU.

    This function should be called the first time this module is loaded.
    """
    global _FLOW_MODEL

    # Built an args parse dictionary for RAFT
    args = argparse.Namespace()
    args.small = False
    args.mixed_precision = False
    args.alternate_corr = False
    model_path = os.path.join(PROJ_DIR, 'weights', 'raft-sintel.pth')

    # Construct flow model and load weights
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(model_path))
    model = model.module
    model.to(_DEFAULT_DEVICE)
    model.eval()

    _FLOW_MODEL = model


def _initialize_detectron2_model():
    """Loads the object segmentation model onto the GPU.

    This function should be called the first time this module is loaded.
    """
    global _DETECTRON2_MODEL

    cfg = detectron2.config.get_cfg()
    cfg.merge_from_file(os.path.join(PROJ_DIR, 'src', 'detectron2-configs',
                                     'mask_rcnn_R_50_FPN_inference_acc_test.yaml'))
    _DETECTRON2_MODEL = detectron2.engine.defaults.DefaultPredictor(cfg)


def _initialize_detectron2_panoptic_model():
    global _DETECTRON2_PANOPTIC_MODEL

    cfg = detectron2.config.get_cfg()
    cfg.merge_from_file(os.path.join(PROJ_DIR, 'src', 'detectron2-configs',
                                     'panoptic_fpn_R_50_inference_acc_test.yaml'))
    _DETECTRON2_PANOPTIC_MODEL = detectron2.engine.defaults.DefaultPredictor(cfg)


def mask_check(mask, x, y):
    """Check whether the keypoints are in the mask or around the boundary.

    :param mask:  The mask for checking (2D Numpy array)
    :param x:     x coordinate of the query point (float)
    :param y:     y coordinate of the query point (float)
    :return:      Whether the query point is on the mask or not (bool)
    """
    return mask[int(y)][int(x)]


def warp_affine_mask(mask, *args, **kwargs):
    """Apply cv2.warpAffine to a mask array.

    :param mask: The mask to warp via warpAffine (HxW bool NumPy array)
    :param args: Additional arguments to cv2.warpAffine (list)
    :param kwargs: Additional arguments to cv2.warpAffine (dict)
    :return: HxW bool NumPy array
    """
    # Cast mask to float for compatibility and to allow conservative masking
    mask_float = mask.astype(np.float)
    mask_warped_float = cv2.warpAffine(mask_float, *args, **kwargs)
    # Only allow pixels that were fully-sampled from valid pixels
    mask_warped = (mask_warped_float >= 1)

    return mask_warped


def compute_optical_flow(img_input, img_target, mask_input=None, mask_target=None):
    """Computes the optical flow from the input image to the target image.

    The mask arguments optionally specify areas to remove from the final flow output. Any vectors that come FROM a
    masked-out input pixel or go TO a masked-out target pixel become nan.

    :param img_input: The input image to align to img_target (HxWx3 uint8 NumPy array)
    :param img_target: The target image that img_input will be aligned to (HxWx3 uint8 NumPy array)
    :param mask_input: The mask of input locations to include (HxW bool NumPy array w/ False for excluded pixels)
    :param mask_target: The mask of target locations to include (HxW bool NumPy array w/ False for excluded pixels)
    :return: Optical flow at the scale of the input images (HxWx2 float32 NumPy array)
    """
    if _FLOW_MODEL is None:
        _initialize_flow_model()

    H, W, C = img_input.shape

    # Initialize default masks if not given
    if mask_input is None:
        mask_input = np.full((H, W), True, dtype=np.bool)
    if mask_target is None:
        mask_target = np.full((H, W), True, dtype=np.bool)

    # Convert images to FloatTensor images
    img_input_t = torch.from_numpy(img_input).permute(2, 0, 1).float()
    img_target_t = torch.from_numpy(img_target).permute(2, 0, 1).float()
    # Convert masks to FloatTensor images
    mask_input_t = torch.from_numpy(mask_input[np.newaxis]).float()
    mask_target_t = torch.from_numpy(mask_target[np.newaxis]).float()

    # Pad the image and the mask for compatibility with the flow model
    padder = InputPadder((C, H, W))
    mask_padder = InputPadder((C, H, W), pad_mode='constant', pad_value=0)
    img_input_t_pad, img_target_t_pad = padder.pad(img_input_t.unsqueeze(0), img_target_t.unsqueeze(0))
    mask_input_t_pad, mask_target_t_pad = mask_padder.pad(mask_input_t.unsqueeze(0), mask_target_t.unsqueeze(0))
    _, _, H_pad, W_pad = img_input_t_pad.size()

    # Compute optical flow between the padded images
    with torch.no_grad():
        _, flow_pad = _FLOW_MODEL(img_input_t_pad.to(_DEFAULT_DEVICE), img_target_t_pad.to(_DEFAULT_DEVICE),
                                  iters=10, test_mode=True)
    flow_pad_cpu = flow_pad.cpu()

    # Identify flow vectors whose end points lie on masked-out target pixels
    mask_target_to_input_t_pad = warp_flow(mask_target_t_pad[0], flow_pad_cpu[0])
    # Compute mask from flow whose start or end points lie on valid pixels (1xHxW)
    mask_valid_t_pad = mask_input_t_pad * mask_target_to_input_t_pad
    # Replace invalid flow vectors with nan (inequality needed due to lack of precision)
    masked_flow_pad_cpu = torch.where(mask_valid_t_pad >= 0.999, flow_pad_cpu, np.nan * torch.ones_like(flow_pad_cpu))

    masked_flow_pad_cpu_unpad = mask_padder.unpad(masked_flow_pad_cpu)
    # Strip batch dimension and reshape to output dimensions
    ret = masked_flow_pad_cpu_unpad[0].permute(1, 2, 0).numpy()

    return ret


def compute_object_instances(image):
    """Detect object instances using the backend Detectron2 model.

    The output of this function follows this documentation:
    https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

    :param image: RGB color image (HxWx3 uint8 NumPy array)
    :return: detectron2.structures.Instances
    """
    if _DETECTRON2_MODEL is None:
        _initialize_detectron2_model()

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    predictions = _DETECTRON2_MODEL(image_bgr)

    return predictions['instances']


def compute_panoptic_segmentation(image):
    """Perform panoptic segmentation using the backend Detectron2 panoptic model.

    The output of this function follows this documentation:
    https://detectron2.readthedocs.io/tutorials/models.html#model-output-format

    :param image: RGB color image (HxWx3 uint8 NumPy array)
    :return: 2-tuple with the following items
             - HxW torch.int32 Tensor
             - list of dict describing class-related metadata. For thing classes:
                 - id: ID for the given class metadata
                 - isthing: Whether the given class corresponds to a thing (True)
                 - score: Confidence that this instance is a thing
                 - category_id: The thing class index
                 - instance_id: The ID of the corresponding instance
                 - area: The area of the corresponding region
               For stuff classes:
                 - id: ID for the given class metadata
                 - isthing: Whether the given class corresponds to a thing (False)
                 - category_id: The stuff class index
                 - area: The area of the corresponding region
    """
    if _DETECTRON2_PANOPTIC_MODEL is None:
        _initialize_detectron2_panoptic_model()

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    predictions = _DETECTRON2_PANOPTIC_MODEL(image_bgr)

    return predictions['panoptic_seg']


class PanopticSegmentor(object):

    def __init__(self):
        dataset_metadata = MetadataCatalog.get('coco_2017_val_100_panoptic_separated')
        self._thing_class_labels = dataset_metadata.thing_classes
        self._stuff_class_labels = dataset_metadata.stuff_classes


    def segment_image(self, image, class_labels):
        """Returns a binary mask of pixels that include the given classes within the given image.
        
        :param image: RGB color image (HxWx3 uint8 NumPy array)
        :param class_labels: Class labels to segment (list of str)
        :return: Binary mask with 1s at locations corresponding to the given classes (HxW bool NumPy array)
        """
        raw_mask_id_map, class_info = compute_panoptic_segmentation(image)

        # Determine the class indexes for the desired thing and stuff classes
        thing_class_indexes = []
        stuff_class_indexes = []
        for label in class_labels:
            if label in self._thing_class_labels:
                thing_class_indexes.append(self._thing_class_labels.index(label))
            elif label in self._stuff_class_labels:
                stuff_class_indexes.append(self._stuff_class_labels.index(label))
            else:
                raise ValueError(f'Unknown class label "{label}" provided')

        # Determine the mask IDs to look for in the raw mask ID map
        desired_mask_ids = []
        for info in class_info:
            is_thing = info['isthing']
            category_id = info['category_id']
            # Try to find the category ID in the desired thing classes for things, and desired stuff classes for stuff
            if (is_thing and category_id in thing_class_indexes) \
                    or (not is_thing and category_id in stuff_class_indexes):
                desired_mask_ids.append(info['id'])

        if len(desired_mask_ids) > 0:
            class_masks = [raw_mask_id_map == i for i in desired_mask_ids]
            output_mask = torch.stack(class_masks).sum(dim=0).type(torch.bool)
        else:
            output_mask = torch.zeros_like(raw_mask_id_map).type(torch.bool)
        output_mask_np = output_mask.cpu().numpy()

        return output_mask_np


def get_spaced_index_list(num_items, total=None, spacing=None):
    """Returns a list to index into a given number of items at evenly-spaced intervals.

    This function takes either `total` or `spacing` (but not both) as parameters; given one parameter, the other is
    automatically computed. If `total` is given, `spacing` is maximized subject to the `total` number of items is
    indexed. If `spacing` is given, `total` is simply the number of items that `spacing` permits within `num_items`.

    This function always returns 0 (the index of the first item), but is not guaranteed to index the last several items.
    In particular, this happens if it is not possible to index `total` items from `num_items` at a discrete interval, or
    if `spacing` does not neatly land on the final item.

    :param num_items: The number of items in the superset that can be indexed (int)
    :param total: The number of items in the returned subset to index (int)
    :param spacing: The number of items away from the previous item (int)
    :return: list of int
    """
    if not (xor((total is None), (spacing is None))):
        raise ValueError(
            'Either total or spacing, but not both, must be specified')

    if spacing is not None:
        assert 0 < spacing < num_items
        total = math.ceil(num_items / spacing)
    elif total is not None:
        assert(0 < total <= num_items)
        spacing = math.floor((num_items - 1) / (total - 1))

    return [spacing*x for x in range(total)]


def dilate_mask(mask, kernel_size):
    """Applies a dilation operation to the given mask.

    :param mask: H x W NumPy bool array
    :param kernel_size: int
    :return: H x W NumPy bool array
    """
    mask_uint8 = 255 * np.stack([mask] * 3, axis=2).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask_dilated_uint8 = cv2.dilate(mask_uint8, kernel)
    mask_dilated = (mask_dilated_uint8[:, :, 0] / 255).astype(np.bool)

    return mask_dilated


def hausdorff_distance(mask_a, mask_b, dist_type=cv2.DIST_L2):
    """Computes the Hausdorff distance between two point sets (masks).

    Source: https://cs.stackexchange.com/a/118020

    :param mask_a: HxW bool NumPy array with False for excluded pixels)
    :param mask_b: HxW bool NumPy array with False for excluded pixels)
    :param dist_type: The metric used to measure distance between pixel locations
    :return:
        - max_dist: NaN if an empty mask is given (float)
        - max_loc: Normally (y, x), or (-1, -1) if an empty mask is given (2-tuple of int64)
        - max_image_index: 0 if mask A contains the furthest point, 1 if mask B does, -1 if an empty mask is given (int)
    """
    if not mask_a.max() or not mask_b.max():
        # One of the point sets is empty, so return NaN
        return np.nan, (-1, -1), -1

    # Get distance transform of * (distance of every point from a pixel in *)
    dt_a = cv2.distanceTransform(1 - mask_a.astype(np.uint8), dist_type, cv2.DIST_MASK_PRECISE)
    dt_b = cv2.distanceTransform(1 - mask_b.astype(np.uint8), dist_type, cv2.DIST_MASK_PRECISE)
    # Mask out locations not in the other set, then take the max among remaining values
    max_a_dist = np.max(dt_b * mask_a)
    max_a_loc = np.unravel_index(np.argmax(dt_b * mask_a), dt_b.shape)
    max_b_dist = np.max(dt_a * mask_b)
    max_b_loc = np.unravel_index(np.argmax(dt_a * mask_b), dt_a.shape)

    if max_a_dist > max_b_dist:
        return max_a_dist, max_a_loc, 0
    else:
        return max_b_dist, max_b_loc, 1


def mask_diameter(mask):
    """Computes the maximum distance between any two points in a mask.

    :param mask: HxW bool NumPy array w/ False for excluded pixels
    :return: float
    """
    # Find the convex hull (as x, y points) containing the given mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Shove the points from all contours into one list
    contour_points = np.concatenate([contour[:, 0, :] for contour in contours])
    # Compute the convex hull of all points
    hull = cv2.convexHull(contour_points).squeeze()
    # Compute distances between every pair of points in the hull
    point_dists = [np.linalg.norm(a - b) for a, b in combinations(hull, 2)]

    return max(point_dists)


def intersection_over_union(mask_a, mask_b):
    """Computes the IoU between two masks.

    :param mask_a: HxW bool NumPy array
    :param mask_b: HxW bool NumPy array
    :return: float
    """
    intersection = mask_a & mask_b
    union = mask_a | mask_b

    if union.sum() == 0:
        return np.nan

    return intersection.sum() / union.sum()
