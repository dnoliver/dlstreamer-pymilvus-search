import json

import cv2

# Read the video
cap = cv2.VideoCapture("video.mp4")

# Get total number of frames
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# Read results.json
with open("results.json", "r") as f:
    results = json.load(f)

    # Iterate over each result
    for result in results:

        # Iterate over each result
        for _res in result:

            # Get the timestamp
            timestamp = _res["entity"]["timestamp"]

            # Seek the timestamp in the video
            cap.set(cv2.CAP_PROP_POS_MSEC, timestamp / 1e6)

            # Read frame
            ret, frame = cap.read()

            # Display the frame
            if ret:
                print(f"Result at timestamp = {timestamp} , press 'q' to continue")
                cv2.imshow("Result", frame)
                while True:
                    if cv2.waitKey(20) & 0xFF == ord("q"):
                        break
