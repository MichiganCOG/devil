#!/bin/bash

SCRIPT_DIR="$(cd $(dirname $0); pwd)"
PROJ_DIR="$(cd $SCRIPT_DIR/..; pwd)"

cd "$PROJ_DIR"
mkdir -p "weights"

wget -O weights/raft-sintel.pth https://umich.box.com/shared/static/dy0srhg3z6r8ugqitsummub9jmulw7vg.pth
