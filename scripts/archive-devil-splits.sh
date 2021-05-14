#!/bin/bash

SCRIPT_DIR="$(cd $(dirname $0); pwd)"
PROJ_DIR="$(cd $SCRIPT_DIR/..; pwd)"

cd "$PROJ_DIR/datasets/devil"
for split in *; do
  if [ -d "$split" ]; then
    if [ -f "$split.tar" ]; then
      rm "$split.tar"
    fi
    cd "$split"
    tar -cvf "../$split.tar" *
    cd ..
  fi
done
