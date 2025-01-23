#!/bin/bash

DETECTION_MODEL="./models/intel/person-vehicle-bike-detection-2004/FP16/person-vehicle-bike-detection-2004"
CLASSIFICATION_MODEL="./models/public/resnet-50-pytorch/FP16/resnet-50-pytorch"
INPUT_FILE="./video.mp4"

gst-launch-1.0 filesrc location=$INPUT_FILE ! \
    decodebin ! \
    gvadetect \
        model=$DETECTION_MODEL.xml \
        inference-interval=7 \
        threshold=0.4 \
        device=CPU ! \
    queue ! \
    gvaclassify \
        model=$CLASSIFICATION_MODEL.xml \
        model-proc=$CLASSIFICATION_MODEL.json \
        device=CPU ! \
    queue ! \
    gvapython module=./insertion.py class=FrameInsertion function=process_frame ! \
    gvawatermark ! \
    videoconvert ! \
    autovideosink
