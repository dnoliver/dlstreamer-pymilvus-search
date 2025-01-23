"""
Insert computed embeddings into Milvus
"""

import os
from time import time
from typing import List

import cv2
import numpy
from gstgva import RegionOfInterest, Tensor, VideoFrame
from PIL import Image
from sklearn.preprocessing import normalize

from milvus_utils import CollectionExists, create_collection, get_milvus_client

MILVUS_URI = "milvus_demo.db"
MILVUS_TOKEN = ""
COLLECTION_NAME = "dlstreamer_computed_embeddings"
ASSETS_FOLDER = "frames"
DEBUG = False
MODEL_DIM = 1000
LAYER_NAME = "classification"


class FrameInsertion:
    def __init__(
        self,
    ):
        # Create Milvus Client
        self.milvus_client = get_milvus_client(uri=MILVUS_URI, token=MILVUS_TOKEN)

        # Initialize Milvus Collection
        create_collection(
            milvus_client=self.milvus_client,
            collection_name=COLLECTION_NAME,
            dim=MODEL_DIM,
            drop_old=True,
        )

        # Initialize Image Storage
        os.makedirs(ASSETS_FOLDER, exist_ok=True)

    def process_frame(self, frame: VideoFrame) -> bool:
        regions: List[RegionOfInterest] = frame.regions()

        # Iterate over Regions
        for roi in regions:
            tensors: List[Tensor] = roi.tensors()

            # Iterate over Tensors
            for tensor in tensors:

                # Gather tensor information for debugging
                layer_name: str = tensor.layer_name()
                tensor_name: str = tensor.name()
                tensor_data: numpy.ndarray = (
                    tensor.data()
                    if tensor.precision() != Tensor.PRECISION.UNSPECIFIED
                    else numpy.ndarray([0])
                )
                roi_label: str = roi.label()
                timestamp: float = time()
                image_path: str = f"./{ASSETS_FOLDER}/{timestamp}.jpg"

                # Print Debug information
                if DEBUG:
                    print("")
                    print("===== DEBUG =====")
                    print(f"Layer Name: {layer_name}")
                    print(f"Tensor Name: {tensor_name}")
                    print(f"Tensor Shape: {tensor_data.shape}")
                    print(f"Region Of Interest Label: {roi_label}")
                    print("===== END =====")
                    print("")

                # Only process classification frames
                if tensor.name() != LAYER_NAME:
                    continue

                # Normalize the tensor data
                feature_vector = normalize(
                    tensor_data.reshape(1, -1), norm="l2"
                ).flatten()

                # Save Video Frame to Disk
                with frame.data() as frame_data:
                    # Convert the frame data from YUV420p to RGB
                    frame_data_rgb = cv2.cvtColor(frame_data, cv2.COLOR_YUV420p2RGB)

                    # Save the converted image
                    cv2.imwrite(image_path, frame_data_rgb)

                # Gather data
                to_insert = {
                    "vector": feature_vector,
                    "filename": image_path,
                    "label": roi_label,
                    "timestamp": timestamp,
                }

                # Insert the feature vector into Milvus
                self.milvus_client.insert(
                    collection_name=COLLECTION_NAME,
                    data=[to_insert],
                )

        return True

    def __del__(self):
        pass
