"""Code to help visualize images, videos, plots, etc."""
import os

import cv2
import numpy as np
import torch
from PIL import Image
from detectron2.data import MetadataCatalog
from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage

from ..common_util.misc import makedirs
from ..devil import hausdorff_distance

_TO_PIL_IMAGE_OBJ = ToPILImage()


def show_torch_image(image):
    """

    :param image: FloatTensor (1xCxHxW)
    """

    image_pil = _TO_PIL_IMAGE_OBJ(image.cpu())
    image_pil.show()

    return image_pil


def show_keypoints(image, keypoints, output_path=None, rev_channels=True):
    img_out = image.copy()
    img_out = cv2.drawKeypoints(image, keypoints, img_out, color=(255, 255, 0),
                                flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    show_opencv_image(img_out, output_path, rev_channels)


def show_keypoints_list(image, keypoints_list):
    """Show a list of keypoint coordinates on a given image.

    :param image: HxWxC uint8 NumPy array
    :param keypoints_list: list of length-2 arrays [x, y]
    :return:
    """
    img_out = image.copy()
    for keypoint in keypoints_list:
        cv2.circle(img_out, tuple([int(k) for k in keypoint]), 1, (0, 0, 255))
    show_opencv_image(img_out)


def show_opencv_image(image, output_path=None, rev_channels=True):
    """Display the given OpenCV-like image.

    :param image: HxWxC uint8 NumPy array
    :param rev_channels: Whether to reverse RGB<->BGR color channels before showing
    """
    img_out = image.copy()
    if rev_channels:
        img_out = cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img_out)
    if output_path:
        makedirs(os.path.dirname(output_path))
        im_pil.save(output_path)
    else:
        im_pil.show()


def show_numpy_mask(mask):
    mask_pil = Image.fromarray(mask)
    mask_pil.show()


def show_flow_field(image, flow, spacing=20):
    """Visualize the flow vectors at evenly-spaced points on the given image.

    :param image: HxWx3 uint8 NumPy array
    :param flow: HxWx2 float NumPy array
    """
    H, W, _ = image.shape
    plt.figure()
    plt.imshow(image)

    plot_args = []
    for y in range(0, H, spacing):
        for x in range(0, W, spacing):
            u, v = flow[y, x, :]
            # Flow source
            origin = (x, y)
            # Flow destination
            endpoint = (x+u, y+v)
            # Add line from origin to endpoint
            plot_args += [[origin[0], endpoint[0]], [origin[1], endpoint[1]], 'g']
            # Add x at endpoint
            plot_args += [endpoint[0], endpoint[1], 'gx']
    plt.plot(*plot_args, markersize=3)
    plt.show()


def overlay_mask_images(mask_a, mask_b, color_a=(255, 255, 0), color_b=(0, 255, 255)):
    """Colors both masks and composites them on a black background.

    :param mask_a: HxW bool NumPy array
    :param mask_b: HxW bool NumPy array
    :param color_a: 3-tuple of ints
    :param color_b: 3-tuple of ints
    :return: HxWxC uint8 NumPy array
    """

    def mask_to_transparent_colored(mask, color):
        colorized_mask_pil = Image.fromarray(mask).convert('RGBA')
        data = colorized_mask_pil.load()
        for x in range(colorized_mask_pil.size[0]):
            for y in range(colorized_mask_pil.size[1]):
                if data[x, y] == (255, 255, 255, 255):
                    data[x, y] = color + (255,)
                else:
                    data[x, y] = (0, 0, 0, 0)
        return colorized_mask_pil


    H, W = mask_a.shape
    colorized_mask_a_pil = mask_to_transparent_colored(mask_a, color_a)
    colorized_mask_b_pil = mask_to_transparent_colored(mask_b, color_b)

    ret = Image.new('RGB', (W, H))
    ret.paste(colorized_mask_a_pil, (0, 0), colorized_mask_a_pil)
    ret.paste(colorized_mask_b_pil, (0, 0), colorized_mask_b_pil)
    ret = np.array(ret)

    return ret


def visualize_hausdorff_distance(mask_a, mask_b):
    dist, max_location, _ = hausdorff_distance(mask_a, mask_b)
    ret = overlay_mask_images(mask_a, mask_b)
    if np.isinf(dist):
        ret = cv2.circle(ret, (0, 0), 15, (255, 255, 255))
    else:
        ret = cv2.circle(ret, tuple(max_location[::-1]), int(dist), (255, 0, 0))
        ret = cv2.circle(ret, tuple(max_location[::-1]), 10, (255, 0, 0))
    return ret


def get_mask_outline(mask):
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask_grayscale = 255 * mask.astype(np.uint8)
    mask_dilated = cv2.dilate(mask_grayscale, kernel)
    mask_eroded = cv2.erode(mask_grayscale, kernel)
    mask_outline = np.array((mask_dilated - mask_eroded) / 255, dtype=np.bool)

    return mask_outline


def render_instances(image_shape, instances, dataset_label='coco_2017_test'):
    """Produce a visualization of the detected instances.

    :param image_shape: (H, W) (2-tuple of int)
    :param instances: detectron2.structures.Instances
    :param dataset_label: Name of the dataset (see detectron2.data.MetadataCatalog) (str)
    :return: HxWxC uint8 NumPy array
    """
    # Overlay the masks for each instance on top of each other
    final_instance_mask = 80 * torch.ones(image_shape, dtype=torch.int64)
    for mask, class_id in zip(instances.pred_masks.cpu(), instances.pred_classes.cpu()):
        final_instance_mask[mask > 0] = class_id
    # Generate a palette-type image
    instances_image_pil = Image.fromarray(final_instance_mask.numpy().astype(np.uint8), 'P')

    # Set the palette based on the dataset colors
    class_colors = MetadataCatalog.get(dataset_label).thing_colors
    class_colors_aug = class_colors + [(0, 0, 0) for _ in range(256 - len(class_colors))]
    instances_image_pil.putpalette([item for sublist in class_colors_aug for item in sublist])

    # Overlay the detected class names
    class_labels = MetadataCatalog.get(dataset_label).thing_classes
    detected_class_labels = set([x for x in map(lambda x: class_labels[x], instances.pred_classes)])
    classes_str = ', '.join(detected_class_labels)
    instances_image = cv2.putText(np.array(instances_image_pil.convert('RGB')), classes_str, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    return instances_image
