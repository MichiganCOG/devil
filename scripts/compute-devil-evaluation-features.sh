#!/bin/bash

if [ "$0" == "$BASH_SOURCE" ]; then
    echo 'Script must be invoked with `source` or `.`'
    exit 1
fi
if [ "$#" != 1 ]; then
    echo "Usage: source compute-devil-evaluation-features.sh SOURCE_SPLIT_NAME"
    return
fi

SOURCE_SPLIT_NAME="$1"

OLD_PWD="$(pwd)"
SCRIPT_DIR="$(cd $(dirname $BASH_SOURCE); pwd)"
PROJ_DIR="$(cd $SCRIPT_DIR/..; pwd)"

cd "$PROJ_DIR/video-inpainting-evaluation"
conda activate ./env
./scripts/preprocessing/compute-evaluation-features.sh \
    "$PROJ_DIR/datasets/devil/$SOURCE_SPLIT_NAME" \
    "$PROJ_DIR/eval-data/devil/$SOURCE_SPLIT_NAME"
conda deactivate

cd "$OLD_PWD"
