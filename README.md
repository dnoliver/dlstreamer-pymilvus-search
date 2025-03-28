# Image Search with Intel DLStreamer and PyMilvus

![Pipeline Diagram](pipeline.drawio.png)

## Running the environment

```ps1
docker compose build
docker compose up -d
sudo xhost +
docker compose exec app bash
```

## Running pipelines

Check the GStreamer setup:

```bash
gst-launch-1.0 videotestsrc ! videoconvert ! autovideosink
```

Prepare models and assets:

```bash
bash init.sh
```

Run extraction pipeline to detect objects in video:

```bash
bash extraction.sh
```

Insert the detected objects into Milvus Vector Database:

```bash
python insertion.py
```

Get embeddings for a target image (crop.jpg)

```bash
bash search.sh
```

Query the Milvus Vector Database for the target image:

```bash
python query.py
```

Find the result frames in the video:

```bash
python seek.py
```

## Development

Format and lint code:

```bash
pip install isort black pylint mdformat
isort *.py; black *.py; pylint -E *.py; mdformat README.md
```

## Links

- [Running GUI Applications in a Linux Docker Container](https://www.baeldung.com/linux/docker-container-gui-applications)
- [GStreamer command-line cheat sheet](https://github.com/matthew1000/gstreamer-cheat-sheet)
- [Intel Deep Learning Streamer](https://dlstreamer.github.io/)
