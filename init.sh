#!/bin/bash

# Download detection model
(
    omz_downloader --name person-vehicle-bike-detection-2004 --output_dir models && \
    cp ./config/person-vehicle-bike-detection-2004.json ./models/intel/person-vehicle-bike-detection-2004/FP16/ && \
    cp ./config/person-vehicle-bike-detection-2004.json ./models/intel/person-vehicle-bike-detection-2004/FP16-INT8/ && \
    cp ./config/person-vehicle-bike-detection-2004.json ./models/intel/person-vehicle-bike-detection-2004/FP32/ \
) &

# Download classification model
(
    omz_downloader --name resnet-50-pytorch --output_dir models && \
    omz_converter --name resnet-50-pytorch --download_dir models --output_dir models && \
    cp ./config/resnet-50-pytorch.json ./models/public/resnet-50-pytorch/FP16/ && \
    cp ./config/resnet-50-pytorch.json ./models/public/resnet-50-pytorch/FP32/ \
) &

# Download input video
(
    wget -O video.mp4 https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4
) &

# Wait until the forks are completed
wait
