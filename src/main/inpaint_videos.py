import argparse
import os
import shutil
from subprocess import check_call
from tempfile import TemporaryDirectory

from PIL import Image

from ..common_util.misc import makedirs
from ..video_inpainting import create_padded_masked_video_dataset
from ..video_inpainting.util import PILImagePadder


def main(frames_dataset_path, masks_dataset_path, inpainting_results_root, temp_root, index_range, run_path,
         image_size_divisor):

    if len(image_size_divisor) == 1:
        image_size_divisor = image_size_divisor[0]
    elif len(image_size_divisor) == 2:
        image_size_divisor = tuple(image_size_divisor)
    else:
        raise ValueError('image_size_divisor must contain 1-2 values')

    # Locate the run script
    run_path = os.path.abspath(run_path)
    assert os.path.isfile(run_path)

    # Expand the temporary folder root
    if temp_root is not None:
        temp_root = os.path.abspath(temp_root)

    # Define a helper for padding and unpadding images
    padder = PILImagePadder(image_size_divisor, mode='symmetric')
    # Load dataset tar files
    dataset = create_padded_masked_video_dataset(frames_dataset_path, masks_dataset_path)
    dataset.padder = padder

    if index_range is None:
        index_range = [0, len(dataset) - 1]
    else:
        assert 0 <= index_range[0] <= index_range[1] < len(dataset)

    # Run inpainting on every video in the dataset
    for i in range(index_range[0], index_range[1] + 1):
        with TemporaryDirectory(dir=temp_root) as temp_in_root, TemporaryDirectory(dir=temp_root) as temp_out_root:
            # Extract current video to temp folder
            frame_video_name = dataset.get_video_name(i)
            raw_resolution = dataset.get_raw_resolution(i)
            dataset.extract_masked_video(i, temp_in_root)
            # Run inpainting and store results to temp folder
            _run_inpainting_script(run_path, temp_in_root, temp_out_root)
            # Unpad results and store at final destination
            output_root = os.path.join(inpainting_results_root, frame_video_name)
            _unpad_copy_frames(temp_out_root, output_root, padder, raw_resolution)


def _unpad_copy_frames(interim_output_root, final_output_root, padder, raw_resolution):
    """Unpad and copy intermediate results from the inpainting method to the final output folder."""
    makedirs(final_output_root)
    for image_file_name in sorted(os.listdir(interim_output_root)):
        if image_file_name.endswith('_pred.png'):
            padded_image_path = os.path.join(interim_output_root, image_file_name)
            final_image_path = os.path.join(final_output_root, image_file_name)
            if not padder.needs_padding(raw_resolution):
                shutil.copy(padded_image_path, final_image_path)
            else:
                image_pad = Image.open(padded_image_path)
                image = padder.unpad_image(image_pad, raw_resolution)
                image.save(final_image_path)


def _run_inpainting_script(run_path, temp_input_root, temp_output_root):
    """Run the given script on the temporary inputs and store the results in the temporary output folder."""
    exec_dir, script_file_name = os.path.split(run_path)
    cmd = ['bash', script_file_name, temp_input_root, temp_output_root]
    check_call(cmd, cwd=exec_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_path', type=str, required=True,
                        help='The path to the script that runs the video inpainting method')
    parser.add_argument('--image_size_divisor', type=int, nargs='+', default=[1],
                        help='The value(s) by which the resolution of the inpainting method input must be divisible. '
                             'Can either be one or two integers (width then height for two)')
    parser.add_argument('--frames_dataset_path', required=True, type=str,
                        help='The path to the tar file or directory containing all video frames in the dataset')
    parser.add_argument('--masks_dataset_path', required=True, type=str,
                        help='The path to the tar file or directory containing all video masks in the dataset')
    parser.add_argument('--inpainting_results_root', required=True, type=str,
                        help='The path to the folder that will contain all inpainted results')
    parser.add_argument('--temp_root', type=str, default=None,
                        help='Path to the temporary directory where intermediate data will be stored')
    parser.add_argument('--index_range', type=int, nargs=2, default=None,
                        help='The range of video indexes to inpaint (inclusive)')

    args = parser.parse_args()
    main(**vars(args))
