from math import ceil

import numpy as np
from PIL import Image


class PILImagePadder(object):

    def __init__(self, divisor, **kwargs):
        """Constructor

        :param divisor: The value(s) that the image dimensions should be divisible by (int or 2-tuple of ints)
        :param kwargs: Additional arguments to pass to np.pad()
        """
        assert isinstance(divisor, int) or isinstance(divisor, tuple)
        if isinstance(divisor, tuple):
            assert len(divisor) == 2
            for item in divisor:
                assert isinstance(item, int)

        self.divisor = divisor
        self.pad_kwargs = kwargs


    def pad_image(self, image):
        """Adds padding to the given image.

        :param image: PIL.Image
        :return: PIL.Image
        """
        assert isinstance(image, Image.Image)
        assert image.mode in ['RGB', '1'], f'Unsupported image mode {image.mode}'

        # Get padding dimensions
        pad_l, pad_r, pad_t, pad_b = self.get_pad_sizes(image.size)
        pad = ((pad_t, pad_b), (pad_l, pad_r), (0, 0)) if image.mode == 'RGB' else ((pad_t, pad_b), (pad_l, pad_r))
        # Pad image as array
        image_np = np.array(image)
        image_pad_np = np.pad(image_np, pad, **self.pad_kwargs)
        # Convert back to PIL Image
        image_pad = Image.fromarray(image_pad_np)

        return image_pad


    def get_pad_sizes(self, raw_image_size):
        """Compute how much padding should be added to each side of an image for a given resolution.

        :param raw_image_size: (W, H)
        :return: (left, right, top, bottom) padding
        """
        raw_input_width, raw_input_height = raw_image_size
        padded_image_width, padded_image_height = self.get_padded_image_size(raw_image_size)

        pad_l = (padded_image_width - raw_input_width) // 2
        pad_r = padded_image_width - raw_input_width - pad_l
        pad_t = (padded_image_height - raw_input_height) // 2
        pad_b = padded_image_height - raw_input_height - pad_t

        return pad_l, pad_r, pad_t, pad_b


    def get_padded_image_size(self, raw_image_size):
        """Compute the image size after padding for the given resolution.

        :param raw_image_size: The resolution of the input image (W, H)
        :return: The resolution of the padded image (W, H)
        """
        raw_input_width, raw_input_height = raw_image_size
        padded_image_width = self.get_padded_image_dim(
            raw_input_width, self.divisor[0] if isinstance(self.divisor, tuple) else self.divisor)
        padded_image_height = self.get_padded_image_dim(
            raw_input_height, self.divisor[1] if isinstance(self.divisor, tuple) else self.divisor)

        return padded_image_width, padded_image_height


    def unpad_image(self, image_pad, raw_image_size):
        """Removes padding from the given image.

        :param image_pad: The padded image to unpad (PIL.Image.Image)
        :param raw_image_size: (W, H)
        :return: PIL.Image.Image
        """
        pad_l, pad_r, pad_t, pad_b = self.get_pad_sizes(raw_image_size)
        return image_pad.crop((pad_l, pad_t, pad_l + raw_image_size[0], pad_t + raw_image_size[1]))


    def needs_padding(self, raw_image_size):
        """Determines whether an image of the given size needs padding.

        :param raw_image_size: The resolution of the input image (W, H)
        :return: bool
        """
        return self.get_pad_sizes(raw_image_size) != (0, 0, 0, 0)


    @staticmethod
    def get_padded_image_dim(raw_input_dim, divisor):
        return ceil(raw_input_dim / divisor) * divisor
