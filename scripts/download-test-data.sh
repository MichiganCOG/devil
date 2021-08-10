#!/bin/bash

SCRIPT_DIR="$(cd $(dirname $0); pwd)"
PROJ_DIR="$(cd $SCRIPT_DIR/..; pwd)"

cd "$PROJ_DIR"
mkdir -p "test-data"

wget -O test-data.zip https://web.eecs.umich.edu/~szetor/media/DEVIL/test-data.zip
unzip -d test-data test-data.zip
rm test-data.zip
