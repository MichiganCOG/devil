#!/bin/bash

SCRIPT_DIR="$(cd $(dirname $0); pwd)"
PROJ_DIR="$(cd $SCRIPT_DIR/..; pwd)"

cd "$PROJ_DIR/video-inpainting-evaluation"

PYTHON="env/bin/python"
$PYTHON -m src.main.summarize_quant_results \
  -r ../inpainting-results-quantitative/devil/*/* \
  -h "pcons_psnr" "pcons_ssim" "warp_error" \
  -t \
  --no-se \
  > "../inpainting-results-quantitative-summary.tsv"
