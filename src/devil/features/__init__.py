from .. import compute_object_instances


class BinaryForegroundMaskComputer(object):
    """Produces a binary foreground segmentation of a given image."""

    def compute_binary_foreground_mask(self, image, **kwargs):
        """Produces a binary foreground segmentation of a given image.

        :param image: RGB color image (HxWxC uint8 NumPy array)
        :param kwargs: Additional keyword arguments to use for debugging (dict)
        :return: HxW bool NumPy array
        """
        # Predict Detectron2 structured instances
        instances = compute_object_instances(image)
        # Compute the union of all masks
        mask_union = instances.pred_masks.sum(dim=0) > 0
        mask_union_np = mask_union.cpu().numpy()

        return mask_union_np
