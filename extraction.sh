#!/bin/bash

DETECTION_MODEL="./models/intel/person-vehicle-bike-detection-2004/FP16-INT8/person-vehicle-bike-detection-2004"
CLASSIFICATION_MODEL="./models/public/resnet-50-pytorch/FP16/resnet-50-pytorch"
DEVICE="CPU"
INPUT_FILE="./video.mp4"

gst-launch-1.0 filesrc location=$INPUT_FILE ! \
    decodebin ! \
    queue ! \
    gvadetect \
        model=$DETECTION_MODEL.xml \
        model-instance-id=detect1 \
        inference-interval=7 \
        threshold=0.4 \
        device=$DEVICE ! \
    queue ! \
    gvainference \
        model=$CLASSIFICATION_MODEL.xml \
        model-instance-id=infer1 \
        device=$DEVICE ! \
    queue ! \
    gvametaconvert \
        add-tensor-data=true ! \
    gvametapublish \
        method=file \
        file-path=./extraction.json ! \
    gvawatermark ! \
    videoconvert ! \
    autovideosink
