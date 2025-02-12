import json

import numpy
from sklearn.preprocessing import normalize

from milvus_utils import get_milvus_client, get_search_results

"""
Search computed embeddings in Milvus
"""

MILVUS_URI = "milvus_demo.db"
MILVUS_TOKEN = ""
COLLECTION_NAME = "dlstreamer_computed_embeddings"
DEBUG = True
MODEL_DIM = 1000
LAYER_NAME = "classification"

# Create Milvus Client
milvus_client = get_milvus_client(uri=MILVUS_URI, token=MILVUS_TOKEN)

# Load the embeddings from the JSON file
with open("search.json", "r") as f:
    search = json.load(f)

    results = []

    # Iterate over each message
    for message in search:

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
                    print(f"Tensor Label: {tensor["label"]}")
                    print("===== END =====")
                    print("")

                # Normalize the tensor data
                feature_vector = normalize(
                    numpy.array(tensor["data"]).reshape(1, -1), norm="l2"
                ).flatten()

                # Search the image
                search_res = get_search_results(
                    milvus_client=milvus_client,
                    collection_name=COLLECTION_NAME,
                    query_vector=feature_vector,
                    output_fields=["filename", "label", "timestamp"],
                )[0]

                results.append(search_res)

    # Close the client
    milvus_client.close()

    # Dump results to results.json
    with open("results.json", "w") as f:
        json.dump(results, f)
