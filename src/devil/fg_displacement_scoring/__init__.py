import numpy as np


class ForegroundDisplacementScorer(object):
    """Scores the foreground displacement by computing the union among BG masks by a percentage of the frame area.

    This algorithm works by taking the union of all BG masks, and then dividing by the total area of the frame (height
    times width).
    """

    def score_foreground_displacement(self, fg_masks):
        """Scores the foreground displacement by computing the union among BG masks by a percentage of the frame area.

        :param fg_masks: The foreground masks associated with each frame (list of HxW bool NumPy arrays with False for
                         excluded pixels)
        :return: float
        """
        bg_masks = [np.bitwise_xor(True, fg_mask) for fg_mask in fg_masks]
        union_mask = bg_masks[0].copy()
        for bg_mask in bg_masks[1:]:
            union_mask += bg_mask
        union_percentage = union_mask.sum() / union_mask.size

        return union_percentage