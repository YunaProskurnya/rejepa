#!/bin/bash
set -e

# Resolve the destination from this script, not the caller's working directory.
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$script_dir/../data/imagenet100"
cd "$script_dir/../data/imagenet100"

# Download dataset imagenet100
kaggle datasets download ambityga/imagenet100
unzip imagenet100.zip
rm imagenet100.zip

# Move dataset to train and val directories
mkdir train
mv train.X*/* train/
rmdir train.X*

mkdir val
mv val.X*/* val/
rmdir val.X*
