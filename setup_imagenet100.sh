#!/bin/bash

# Download dataset imagenet100
kaggle datasets download ambityga/imagenet100
unzip imagenet100.zip && rm imagenet100.zip

# Move dataset to train and val directories
mkdir train
mv train.X*/* train/ && rmdir train.X*

mkdir val
mv val.X*/* val/ && rmdir val.X*