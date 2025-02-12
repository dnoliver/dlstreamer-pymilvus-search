#!/bin/bash

DETECTION_MODEL="./models/intel/person-vehicle-bike-detection-2004/FP16/person-vehicle-bike-detection-2004"
CLASSIFICATION_MODEL="./models/public/resnet-50-pytorch/FP16/resnet-50-pytorch"
INPUT_FILE="./crop.jpg"

gst-launch-1.0 filesrc location=$INPUT_FILE ! \
    decodebin ! \
    videoconvertscale ! \
    video/x-raw,width=768,height=432 ! \
    gvadetect \
        model=$DETECTION_MODEL.xml \
        inference-interval=1 \
        device=CPU ! \
    queue ! \
    gvaclassify \
        model=$CLASSIFICATION_MODEL.xml \
        model-proc=$CLASSIFICATION_MODEL.json \
        device=CPU ! \
    queue ! \
    gvametaconvert \
        add-tensor-data=true ! \
    gvametapublish \
        method=file \
        file-path=./search.json ! \
    gvawatermark ! \
    videoconvert ! \
    imagefreeze ! \
    autovideosink
