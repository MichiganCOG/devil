import numpy as np
import torch
from torch.nn.functional import grid_sample


def warp_flow(img_input, bkwd_flow_input, mode='bilinear', padding_mode='nan', align_corners=True):
    """Warp the input image by the given backward flow.

    Conceptually, for each pixel coordinate in the output, it goes to the corresponding input coordinate, moves by the
    corresponding flow vector, and then samples the color at the resulting location.

    Most of the kwargs match torch.nn.functional.grid_sample except for `padding_mode`: in addition to the options
    supported by torch.nn.functional.grid_sample, it also supports 'nan', which means that locations sampling outside
    the valid image result in nan (instead of the default of 0).

    :param img_input: The image to warp (CxHxW FloatTensor)
    :param bkwd_flow_input: The backward flow mapping from the output to the input (2xHxW FloatTensor)
    :return: CxHxW FloatTensor
    """
    assert img_input.device == bkwd_flow_input.device, 'img_input and bkwd_flow_input must be on the same device'
    device = img_input.device

    assert img_input.ndim == 3, f'Expected 3 dimensions but got {img_input.ndim}'
    assert bkwd_flow_input.ndim == 3, f'Expected 3 dimensions but got {bkwd_flow_input.ndim}'

    _, H, W = img_input.size()
    C_flow, H_flow, W_flow = bkwd_flow_input.size()
    assert C_flow == 2, f'Expected the number of channels in bkwd_flow_input to be 2 but got {C_flow}'
    assert H == H_flow and W == W_flow, 'The heights and widths of img_input and bkwd_flow_input do not match'

    # Get base sampling grid, which corresponds to not modifying the image
    base_grid_y, base_grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device))
    # Add optical flow to the sampling grid
    sample_abs_x = base_grid_x + bkwd_flow_input[0, :, :]
    sample_abs_y = base_grid_y + bkwd_flow_input[1, :, :]
    # Scale and stack the sampling grid for compatibility with grid_sample
    sample_sc_x = (sample_abs_x.float() - (W - 1) / 2) / ((W - 1) / 2)
    sample_sc_y = (sample_abs_y.float() - (H - 1) / 2) / ((H - 1) / 2)
    sample_sc = torch.stack((sample_sc_x, sample_sc_y), dim=-1)[np.newaxis]
    # If desired, invalidate locations that sample outside the image
    if padding_mode == 'nan':
        sample_sc[sample_sc > 1] = np.nan
        sample_sc[sample_sc < -1] = np.nan
        # Reset padding_mode to the default for `grid_sample`
        padding_mode = 'zeros'
    # Warp the image
    img_warped_batched = grid_sample(img_input[np.newaxis], sample_sc.to(img_input.device),
                                     mode=mode, padding_mode=padding_mode, align_corners=align_corners)
    # Output has batch dimension, so just take first item in the "batch"
    return img_warped_batched[0]
