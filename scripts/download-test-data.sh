#!/bin/bash

SCRIPT_DIR="$(cd $(dirname $0); pwd)"
PROJ_DIR="$(cd $SCRIPT_DIR/..; pwd)"

cd "$PROJ_DIR"
mkdir -p "test-data"

wget -O test-data.zip https://umich.box.com/shared/static/hbzqz4ehdlapxk53jxuzxvna1vh208wo.zip
unzip -d test-data test-data.zip
rm test-data.zip
