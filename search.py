import os
import pprint
from time import time
from typing import List

import cv2 as cv
import numpy
from gstgva import RegionOfInterest, Tensor, VideoFrame
from sklearn.preprocessing import normalize

from milvus_utils import get_milvus_client, get_search_results

MILVUS_URI = "milvus_demo.db"
MILVUS_TOKEN = ""
COLLECTION_NAME = "dlstreamer_computed_embeddings"
ASSETS_FOLDER = "frames"
DEBUG = False
LAYER_NAME = "classification"


class CropSearch:
    def __init__(
        self,
    ):
        # Create Milvus Client
        self.milvus_client = get_milvus_client(uri=MILVUS_URI, token=MILVUS_TOKEN)

    def search_crop(self, frame: VideoFrame) -> bool:
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
                image_path: str = f"./static/{timestamp}.jpg"

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

                # Search the image
                search_res = get_search_results(
                    milvus_client=self.milvus_client,
                    collection_name=COLLECTION_NAME,
                    query_vector=feature_vector,
                    output_fields=["filename", "label", "timestamp"],
                )[0]

                # Sort search results by distance
                search_res.sort(key=lambda x: x["distance"], reverse=True)

                # Pretty print results
                print("")
                print("==== RESULTS ====")
                pprint.pprint(search_res)
                print("==== END ====")
                print("")

        return True

    def __del__(self):
        pass
