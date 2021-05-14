from glob import glob
from math import ceil

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps


def numpy_image_to_tensor(img):
    """Converts an RGB OpenCV-like image to a PyTorch-like image.

    :param img: HxWxC np.float32 array with range [0, 1]
    :return: 1xCxHxW FloatTensor with range [0, 1]
    """
    assert img.min() >= 0
    assert img.max() <= 1

    return numpy_3d_array_to_tensor(img)


def numpy_3d_array_to_tensor(arr):
    """Converts a multichannel np.float32 array to a single-batch PyTorch FloatTensor with prioritized channel dim.

    :param arr: HxWxC np.float32 array
    :return: 1xCxHxW FloatTensor
    """
    _check_numpy_3d_float_array(arr)

    arr_reshaped = np.expand_dims(arr.transpose(2, 0, 1), axis=0)
    tensor = torch.from_numpy(arr_reshaped)

    return tensor


def _check_numpy_3d_float_array(image_np):
    """Asserts that the given argument is a well-formed 3D np.float32 array.

    :param image_np: HxWxC np.float32 array with range [0, 1]
    """
    assert isinstance(image_np, np.ndarray)
    assert image_np.ndim == 3
    assert image_np.dtype == np.float32


def rotate_image(img, degree, interp=cv2.INTER_LINEAR):

    height, width = img.shape[:2]
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, degree, 1.)

    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    img_out = cv2.warpAffine(img, rotation_mat, (bound_w, bound_h), flags=interp+cv2.WARP_FILL_OUTLIERS)

    return img_out


def _check_pil_image_mode(pil_image, format):
    """Checks that the given image matches the expected format.

    :param pil_image: PIL Image
    :param format: PIL mode string (https://pillow.readthedocs.io/en/stable/handbook/concepts.html#modes)
    """

    assert isinstance(pil_image, Image.Image)
    assert pil_image.mode == format


def _get_single_path_from_pattern(file_path_pattern):
    """Returns exactly one path that matches the given glob pattern. Fails if more or fewer than one path is found.

    :param file_path_pattern: A glob pattern that matches the path of the desired file
    :return: The path matching the given pattern
    """

    matched_files = glob(file_path_pattern)
    if len(matched_files) == 0:
        raise ValueError(f'Failed to find any files matching pattern {file_path_pattern}')
    elif len(matched_files) > 1:
        raise ValueError(f'Found too many files matching pattern {file_path_pattern}')
    file_path = matched_files[0]
    return file_path


def pil_rgb_to_numpy(pil_image):
    """Converts an RGB PIL Image to an RGB OpenCV-like np.float32 array with range [0, 1].

    :param pil_image: PIL Image with mode RGB
    :return: HxWxC np.float32 array with mode RGB and range [0, 1]
    """
    _check_pil_image_mode(pil_image, 'RGB')
    ret = np.array(pil_image, dtype=np.float32) / 255

    return ret


def mask_image_path_to_numpy(mask_image_path):
    """Reads in the mask at the given path and converts it to a NumPy array.

    :param mask_image_path: Path to a mask image (str)
    :return: HxW bool NumPy array
    """
    mask_pil = Image.open(mask_image_path)
    # Produce binary image by converting to grayscale and turning all non-black pixels white
    mask_bw_pil = mask_pil.convert('L').point(lambda x: 1 if x > 0 else 0, mode='1')
    # Convert to NumPy
    mask = np.array(mask_bw_pil)
    return mask


def image_path_to_numpy(image_path):
    """Reads in the image at the given path and converts it to a NumPy array.

    :param mask_image_path: Path to an image (str)
    :return: HxWxC uint8 NumPy array
    """
    image_pil = Image.open(image_path)
    image_rgb_pil = image_pil.convert('RGB')
    image = np.array(image_rgb_pil)
    return image


def mask_percentage(mask):
    """Returns the percentage of pixels in the mask that are True.

    :param mask: HxW bool NumPy array (False for locations to be excluded)
    :return: float
    """
    return mask.sum() / mask.size


def cover_crop(image, size, resample=Image.BICUBIC):
    """Resizes the input image to "cover" the given size (see "object-fit: cover" in CSS).

    :param image: PIL.Image
    :param size: (W, H)
    :param resample: PIL resampling flag
    :return: PIL.Image
    """
    W_out, H_out = size
    W_in, H_in = image.size

    # Create an intermediate image where one dimension exactly matches the target size and the other is larger
    ratio = max(W_out / W_in, H_out / H_in)
    W_inter = ceil(W_in * ratio)
    H_inter = ceil(H_in * ratio)
    image_inter = image.resize((W_inter, H_inter), resample=resample)

    # Take center crop of intermediate image
    left = (W_inter - W_out) // 2
    top = (H_inter - H_out) // 2
    ret = image_inter.crop((left, top, left + W_out, top + H_out))

    return ret


def invert_binary_image(image):
    """Inverts a binary PIL Image (with mode "1").

    :param image: PIL.Image.Image with mode "1"
    :return: PIL.Image.Image with mode "1"
    """
    assert isinstance(image, Image.Image)
    assert image.mode == '1'

    return ImageOps.invert(image.convert('L')).convert('1')
