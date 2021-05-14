#!/bin/bash

if [ -n "$SLURM_JOB_ID" ]; then
    SCRIPT_DIR="$(dirname $(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}'))"
    export CUDA_VISIBLE_DEVICES="${SLURM_JOB_GPUS}"
else
    SCRIPT_DIR="$(cd $(dirname $0); pwd)"
fi

PROJ_DIR="$(cd $SCRIPT_DIR/..; pwd)"
VI_PROJ_DIR="$(cd $PROJ_DIR/video-inpainting-evaluation; pwd)"

# Make temporary work folder
TMP_DIR="$SCRATCH_ROOT/devil-curation-evaluate-inpainting-$RANDOM"
mkdir -p "$TMP_DIR"

# Extract dataset
GT_ROOT="$TMP_DIR/dataset"
echo "Extracting the dataset to $GT_ROOT..."
cd "$PROJ_DIR"
env/bin/python -m src.main.extract_evaluation_dataset "$SOURCE_DATASET_PATH" "$MASK_DATASET_PATH" "$GT_ROOT"

# Extract pre-processed evaluation data
if [ "${EVAL_FEATS_PATH: -4}" == ".tar" ]; then
    EVAL_FEATS_ROOT="$TMP_DIR/eval-data"
    echo "Extracting evaluation features to $EVAL_FEATS_ROOT..."
    mkdir -p "$EVAL_FEATS_ROOT"
    tar -xf "$EVAL_FEATS_PATH" --directory "$EVAL_FEATS_ROOT"
else
    echo "Using the evaluation features at $EVAL_FEATS_PATH..."
    EVAL_FEATS_ROOT="$EVAL_FEATS_PATH"
fi

LOG_PATH="$TMP_DIR/output.log"

echo "Running the evaluation library..."
cd "$VI_PROJ_DIR"
env/bin/python -m src.main.evaluate_inpainting \
    --gt_root="$GT_ROOT" \
    --pred_root="$PRED_ROOT" \
    --eval_feats_root="$EVAL_FEATS_ROOT" \
    --log_path="$LOG_PATH" \
    --output_path="$OUTPUT_PATH" \
    $EXTRA_EVAL_SCRIPT_ARGS
retcode="$?"

# Print log on failure
if [ "$retcode" != "0" ]; then
    echo "Evaluation failed. Printing execution log:"
    cat "$LOG_PATH"
fi

# Remove temporary folder
echo "Removing $TMP_DIR..."
rm -rf "$TMP_DIR"

echo "Done."
exit "$retcode"
