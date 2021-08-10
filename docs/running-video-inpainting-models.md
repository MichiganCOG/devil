# Running Video Inpainting Models

This document explains how to set up and run the sample video inpainting projects on our dataset, as well as how to
prepare your own algorithm for DEVIL evaluation.

## Example: Running STTN

This section explains how to run STTN<sup>1</sup>, a project which we have modified for compatibility with our
benchmark.

### Initialize the STTN submodule

```
git submodule update --init video-inpainting-projects/STTN
cd video-inpainting-projects/STTN

# Follows setup instructions in STTN project. Refer to their instructions if these become out-of-date.
conda create -p ./env -y
conda activate ./env
conda env update -f environment.yml
wget -O checkpoints/sttn.pth https://web.eecs.umich.edu/~szetor/media/STTN/sttn.pth
```

### Run the video inpainting helper

The `inpaint_videos.py` script is the helper script that runs video inpainting methods on our datasets. Below is an 
example of running STTN on the `fvi-fgd-h` split:

```
python -m src.main.inpaint_videos \
  --frames_dataset_path=datasets/devil/flickr-all \
  --masks_dataset_path=datasets/devil/fvi-fgd-h \
  --inpainting_results_root=inpainting-results/devil/flickr-all_fvi-fgd-h/sttn \
  --run_path video-inpainting-projects/STTN/devil-run.sh \
  --image_size_divisor 432 240
```

For more details on this script's arguments, run `python -m src.main.inpaint_videos -h`.

### OPTIONAL: Meta helper scripts

**NOTE:** The commands in this section should be run from the `slurm` directory.

We have included meta helper scripts to run `inpaint_videos.py` in a less verbose manner. It also supports the Slurm job 
scheduler.

Below is an example of running the inpainting meta scripts with STTN on the `fvi-fgd-h` DEVIL split:

```
python submit-inpaint-videos.py sttn flickr-all fvi-fgd-h -m local
```

Environment variables can be passed in as string pairs with the `-e` flag. The following example runs inpainting with a
custom scratch storage location (i.e., where intermediate inpainting data is stored):

```
python submit-inpaint-videos.py sttn flickr-all fvi-fgd-h -m local -e TEMP_ROOT /tmp/$USER
```

To launch a job on a Slurm cluster, remove the `-m local` flag:

```
python submit-inpaint-videos.py sttn flickr-all fvi-fgd-h
```

`sbatch` arguments can be passed in as string pairs with the `-s` flag. This illustrative example runs inpainting on a
custom partition:

```
python submit-inpaint-videos.py sttn flickr-all fvi-fgd-h -s partition my-partition
```

If necessary, the `-s` and `-e` flags can be used at the same time.

## Custom Video Inpainting Algorithms

The DEVIL benchmark processes data in a very particular manner to ensure fair comparisons across different inpainting
methods. For this reason, we highly suggest using our helper scripts to run your method for DEVIL evaluation. This
section covers what you need to do to properly run your method with our helper scripts.

### Execution script

Prepare a Bash script that executes your method on a given input folder and saves results to a given output folder. You
can expect our benchmark to call the script in an equivalent manner to the following code:

```bash
cd <script directory>
bash <script path> <input root> <output root>
```

* `<script directory>` is the folder containing your script
* `<script path>` is the full absolute path to your script
* `<input root>` is the folder containing the input frames and masks (an example can be found in 
  `<project root>/devkit/sample-inpainting-inputs`)
* `<output root>` is the folder that will contain the output frames (this folder should already exist before execution)

The Bash script should be self-contained, i.e., it should not rely on external software like Conda to be activated
before the execution.

To test your script, you can run it against the samples in our development kit:

```
cd <script directory>
bash <script_path> <project root>/devkit/sample-inpainting-inputs <project root>/devkit/outputs
```

and verify that the images in `<project root>/devkit/outputs` match the format of 
`<project root>/devkit/sample-inpainting-outputs`.

An example script, `devil-run.sh`, is available in the STTN submodule (see 
[Example: Running STTN](#example-running-sttn)).

### Running your execution script with our benchmark

Below is an example template for running your execution script on one of our DEVIL splits:

```
python -m src.main.inpaint_videos \
  --frames_dataset_path=datasets/devil/flickr-all \
  --masks_dataset_path=datasets/devil/fvi-fgd-h \
  --inpainting_results_root=inpainting-results/devil/flickr-all_fvi-fgd-h/<short method name> \
  --run_path <script path> \
  --image_size_divisor <width divisor> <height divisor>
```

* `<short method name>` is a short, memorable string to associate with your method
* `<script path>` is the path to your execution script (here, it can be relative to the DEVIL project root)
* `<width divisor>` and `<height divisor>` are values that the resolution of inputs to your method must be divisible by.
  For example, if the height and width of all video frames must be divisible by 16, specify `--image_size_divisor 16 16`
  (`--image_size_divisor 16` also works)

Since the video resolutions of our DEVIL splits are not natively supported by all methods, the `--image_size_divisor` 
flag is used to pad frames to make them compatible with your method. Specifically, we apply mirror padding around all 
four edges of the video frame and mask, pass the padded inputs to your method, and then remove the padding from the 
results.

### Input and output file format

The input root will contain images of video frames and masks named as follows: 

* frame_0000_gt.png
* frame_0000_mask.png
* frame_0001_gt.png
* frame_0001_mask.png
* frame_0002_gt.png
* frame_0002_mask.png
* ...

The video frames will be RGB PNG images; the masks will be binary PNGs where black indicates occlusion (i.e., the area
to be inpainted) and white indicates no occlusion. In the Pillow library, these formats correspond to the `RGB` and `1` 
color modes respectively. An example of this file structure with well-formatted images is included in
`<project root>/devkit/sample-inpainting-inputs`.

After inpainting, the output root should contain images of inpainted video frames named as follows:

* frame_0000_pred.png
* frame_0001_pred.png
* frame_0002_pred.png
* ...

These frames should be RGB PNG images in which the "occluded" locations in the input have been inpainted. For
evaluation, the "unoccluded" regions are ignored, so they can correspond to the ground-truth values, full-frame
prediction values, or anything else. An example of this file structure with well-formatted images is included in
`<project root>/devkit/sample-inpainting-outputs`.

### Code snippets

The following code snippets may be useful for converting images or extracting information for compatibility with your
video inpainting method. They should be called inside your execution script.

#### Inverting a binary mask

This Bash function inverts a binary PNG image. It takes two arguments: the source image path and the destination path.
It assumes that the Pillow package is installed in the default Python environment.

```bash
function process-mask {
    python -c "from PIL import Image, ImageOps; out = Image.open('$1').convert('RGB'); out = ImageOps.invert(out); out.save('$2')"
}
```

#### Determining the resolution of an image

The following Bash snippet extracts the resolution of a given image, which may be a necessary argument for your code. It
assumes that Imagemagick is installed on your system.

```bash
INPUT_WIDTH=$(identify -format "%w" "$IMAGE_PATH")
INPUT_HEIGHT=$(identify -format "%h" "$IMAGE_PATH")
```

## References

1. Yanhong Zeng, Jianlong Fu, and Hongyang Chao. Learning Joint Spatial-Temporal Transformations for Video Inpainting.
ECCV 2020. https://github.com/researchmm/STTN
