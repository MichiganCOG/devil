# The DEVIL Benchmark

This code implements the Diagnostic Evaluation of Video Inpainting on Landscapes (DEVIL) benchmark, which is composed
of a curated video/occlusion mask dataset and a comprehensive evaluation scheme. It has been tested on a Ubuntu 20.04
machine with a GTX 1080Ti GPU.

Further details are available in [our paper](https://arxiv.org/abs/2105.05332). If you use this work, please cite our
paper:

```
@article{szeto2021devil,
  author={Szeto, Ryan and Corso, Jason J.},
  title={The DEVIL is in the Details: A Diagnostic Evaluation Benchmark for Video Inpainting},
  journal={arXiv preprint arXiv:2105.05332},
  year={2021}
}
```

Note: This project is in maintenance mode due to the author graduating. Errors will be addressed to the greatest extent possible, but new features and upgrades will not.

## Setup

```bash
# Clone project and evaluation submodule via git
git clone https://github.com/MichiganCOG/devil.git
cd devil
git submodule update --init video-inpainting-evaluation

# Initialize the DEVIL Python environment
conda env create -p ./env -f environment.yml

# Initialize the video inpainting evaluation library
cd video-inpainting-evaluation
# Follow the library's installation instructions. If these are out-of-date, refer to their instructions.
conda env create -p ./env -f environment.yml
conda activate ./env
./scripts/setup/install-flownet2.sh
./scripts/setup/download-models.sh
```

The remaining instructions should be run from this project's root folder with the DEVIL conda environment activated 
unless otherwise stated.

```bash
conda activate ./env
```

## Quick Start

The following script is a minimum working example for our benchmark. It downloads one DEVIL split and one set of
predictions, evaluates the predictions on the split, and prints quantitative results to disk.

```bash
# Download DEVIL splits
python -m src.main.download_devil_splits flickr-all -e
python -m src.main.download_devil_splits fvi-fgd-h -e

# Extract evaluation features
source scripts/compute-devil-evaluation-features.sh flickr-all

# Download sample predictions
python -m src.main.download_inpainting_results cpnet flickr-all fvi-fgd-h

# Run evaluation
cd slurm
python submit-evaluate-inpainting.py cpnet flickr-all fvi-fgd-h -m local
python submit-evaluate-inpainting.py cpnet flickr-all fvi-fgd-h -m local -a ::include fid ::update

# Print quantitative results to `inpainting-results-quantitative-summary.tsv`
cd ..
./scripts/print-quant-results-table.sh
```

The contents of `inpainting-results-quantitative-summary.tsv` should now look similar to the following:

```text
Method	PSNR ▲	SSIM ▲	LPIPS ▼ 	PVCS ▼	FID ▼	VFID ▼	VFID (clips) ▼	Warp error (mask) ▼	PCons (PSNR, mask) ▲
../inpainting-results-quantitative/devil/flickr-all_fvi-fgd-h/cpnet.npz	36.62 	0.9807	0.002363	 0.1462 	 4.85	0.0367	0.0622	0.000957	40.33
```

## Running and Evaluating Custom Methods

We encourage you to run and evaluate your own video inpainting method in our benchmark! To do so, first prepare an 
execution script for your method, and run it on one of our DEVIL splits (instructions are available 
[here](docs/running-video-inpainting-models.md#custom-video-inpainting-algorithms)). Then, evaluate your method using
our evaluation helper script (instructions are available [here](docs/running-evaluation-details.md#running-evaluation)).

## DEVIL Splits

The DEVIL splits used in our paper are composed of the following pairs of video and mask subsets:

| Video subset | Mask subset |
|--------------|-------------|
| flickr-all   | fvi-fgd-h   |
| flickr-all   | fvi-fgd-l   |
| flickr-all   | fvi-fgm-h   |
| flickr-all   | fvi-fgm-l   |
| flickr-all   | fvi-fgs-h   |
| flickr-all   | fvi-fgs-l   |
| flickr-bsm-h | fvi-all     |
| flickr-bsm-l | fvi-all     |
| flickr-cm-h  | fvi-all     |
| flickr-cm-l  | fvi-all     |

## More Information

For more details on how to run video inpainting models and evaluation through our benchmark, please refer to the usage
guides under the `docs` folder.

### Licenses

The code in this repository is available under the MIT License in `LICENSE` with the exceptions listed below:

* Code snippets attributed to Stack Exchange and/or Stack Overflow is available under the
  [Creative Commons Attribution-ShareAlike (CC BY-SA) License](https://creativecommons.org/licenses/by-sa/4.0/).
* Code under each path in the table below is available under separate licenses; refer to the licenses in those paths for
  more details.
  
  | Path                           | License | Source                                                     |
  |--------------------------------|---------|------------------------------------------------------------|
  | `/src/raft/`                   | MIT     | https://github.com/princeton-vl/RAFT                       |
  | `/video-inpainting-evaluation` | MIT     | https://github.com/MichiganCOG/video-inpainting-evaluation |

* Code that is attributed, but does not fall under the above scenarios, is unlicensed. We thank the original authors for
  their open contributions.
