import json

import numpy
from sklearn.preprocessing import normalize

from milvus_utils import create_collection, get_milvus_client

"""
Insert computed embeddings into Milvus
"""


MILVUS_URI = "milvus_demo.db"
MILVUS_TOKEN = ""
COLLECTION_NAME = "dlstreamer_computed_embeddings"
DEBUG = False
MODEL_DIM = 1000
LAYER_NAME = "inference_layer_name:prob"

# Create Milvus Client
milvus_client = get_milvus_client(uri=MILVUS_URI, token=MILVUS_TOKEN)

# Initialize Milvus Collection
create_collection(
    milvus_client=milvus_client,
    collection_name=COLLECTION_NAME,
    dim=MODEL_DIM,
    drop_old=True,
)

# Load the embeddings from the JSON file
with open("extraction.json", "r") as f:
    extractions = json.load(f)

    insertions = []

    # Iterate over each message
    for message in extractions:

        # Iterate over each object
        for obj in message["objects"]:

            # Iterate over each tensor
            for tensor in obj["tensors"]:

                # Only process classification frames
                if tensor["name"] != LAYER_NAME:
                    continue

                # Print Debug information
                if DEBUG:
                    print("")
                    print("===== DEBUG =====")
                    print(f"Layer Name: {tensor["layer_name"]}")
                    print(f"Tensor Name: {tensor["name"]}")
                    print(f"Tensor Dims: {tensor["dims"]}")
                    print("===== END =====")
                    print("")

                # Normalize the tensor data
                feature_vector = normalize(
                    numpy.array(tensor["data"]).reshape(1, -1), norm="l2"
                ).flatten()

                # Gather data
                insertions.append(
                    {
                        "vector": feature_vector,
                        "timestamp": message["timestamp"],
                    }
                )

    # Insert the feature vector into Milvus
    milvus_client.insert(
        collection_name=COLLECTION_NAME,
        data=insertions,
    )

    # Close the client
    milvus_client.close()

    # NOTE: The following error may be raised when the script is finished.
    #
    # Exception ignored in: <function ServerManager.__del__ at 0x7fb59ae04720>
    # Traceback (most recent call last):
    #     File "/python3venv/lib/python3.12/site-packages/milvus_lite/server_manager.py", line 58, in __del__
    #     File "/python3venv/lib/python3.12/site-packages/milvus_lite/server_manager.py", line 53, in release_all
    #     File "/python3venv/lib/python3.12/site-packages/milvus_lite/server.py", line 118, in stop
    #     File "/usr/lib/python3.12/pathlib.py", line 1164, in __init__
    #     File "/usr/lib/python3.12/pathlib.py", line 358, in __init__
    # ImportError: sys.meta_path is None, Python is likely shutting down
    # Exception ignored in: <function Server.__del__ at 0x7fb59ae04540>
    # Traceback (most recent call last):
    #     File "/python3venv/lib/python3.12/site-packages/milvus_lite/server.py", line 122, in __del__
    #     File "/python3venv/lib/python3.12/site-packages/milvus_lite/server.py", line 118, in stop
    #     File "/usr/lib/python3.12/pathlib.py", line 1164, in __init__
    #     File "/usr/lib/python3.12/pathlib.py", line 358, in __init__
    # ImportError: sys.meta_path is None, Python is likely shutting down
    #
    # See https://github.com/milvus-io/pymilvus/issues/2282 for more information.
