#!/usr/bin/env bash
URL=$1
PATTERN=$2
OUTPUT_PATH=$3

REPO_DIR=`mktemp -d`
git clone $URL $REPO_DIR
find $REPO_DIR -name "$PATTERN" -type f | xargs cat | sed 's/ *$//' >> $OUTPUT_PATH
rm -rf $REPO_DIR
