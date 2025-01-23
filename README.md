# Image Search with Intel DLStreamer and PyMilvus

## Running pipelines

Prepare models and assets:

```bash
bash init.sh
```

Run insertion pipeline to populate the image search database:

```bash
bash insertion.sh
```

Prepare a crop.jpg image for searching:

```bash
cp "./frames/$(ls frames | shuf -n 1)" crop.jpg
```

Run search pipeline to get similar detected objects:

```bash
bash search.sh
```

Search results are logged to console.

## Development

Format and lint code:

```bash
pip install isort black pylint
isort *.py; black *.py; pylint -E *.py
```
