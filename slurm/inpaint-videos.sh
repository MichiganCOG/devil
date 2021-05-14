#!/bin/bash

if [ -n "$SLURM_JOB_ID" ]; then
    SCRIPT_DIR="$(dirname $(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}'))"
    export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
else
    SCRIPT_DIR="$(cd $(dirname $0); pwd)"
fi

PROJ_DIR="$(cd $SCRIPT_DIR/..; pwd)"

cd "$PROJ_DIR"

if [ -z "$INDEX_RANGE" ]; then
    env/bin/python -m src.main.inpaint_videos \
        --run_path="$RUN_PATH" \
        --image_size_divisor $IMAGE_SIZE_DIVISOR \
        --frames_dataset_path="$FRAMES_DATASET_PATH" \
        --masks_dataset_path="$MASKS_DATASET_PATH" \
        --inpainting_results_root="$INPAINTING_RESULTS_ROOT" \
        --temp_root="$TEMP_ROOT"
else
    IFS="-" read index_range_a index_range_b <<< "$INDEX_RANGE"
    env/bin/python -m src.main.inpaint_videos \
        --run_path="$RUN_PATH" \
        --image_size_divisor $IMAGE_SIZE_DIVISOR \
        --frames_dataset_path="$FRAMES_DATASET_PATH" \
        --masks_dataset_path="$MASKS_DATASET_PATH" \
        --inpainting_results_root="$INPAINTING_RESULTS_ROOT" \
        --temp_root="$TEMP_ROOT" \
        --index_range "$index_range_a" "$index_range_b"
fi
