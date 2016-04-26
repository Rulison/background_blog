#!/bin/bash

DATASET_NAME="birds"

CV_FGBG_VERSION=1


INPUT_DIR="JPEGS/"$DATASET_NAME"/"
IMAGE_OUTPUT_DIR=$DATASET_NAME"_"$CV_FGBG_VERSION"_output"
MAT_OUTPUT=$DATASET_NAME"_mat"
MAT_GROUND_TRUTH="GT_matlab/"$DATASET_NAME"_GT.mat"


python run_opencv.py -v $CV_FGBG_VERSION -f $INPUT_DIR -o $IMAGE_OUTPUT_DIR
python labeled_image_to_mat.py -f $IMAGE_OUTPUT_DIR -o $MAT_OUTPUT
python evaluate_mats.py -p $MAT_OUTPUT -t $MAT_GROUND_TRUTH