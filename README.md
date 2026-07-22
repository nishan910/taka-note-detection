# Bangladeshi Taka Note Detection

A YOLOv11-based object detection system that identifies Bangladeshi Taka currency notes (denomination + confidence + bounding box) from an uploaded image, served through a FastAPI REST API and packaged as a Docker container.

## Overview

This project detects and classifies Bangladeshi currency notes in images using a YOLOv11n (nano) model trained on the [Bangladeshi Currency Detection dataset](https://universe.roboflow.com) (11 classes, 1523 images). The trained model is wrapped in a REST API (`/predict`) that accepts an image and returns the detected denomination(s), confidence score(s), and bounding box coordinates as JSON.

**Model performance (validation set):**

| Metric | Score |
|---|---|
| mAP50 | 0.844 |
| mAP50-95 | 0.374 |
| Precision | 0.794 |
| Recall | 0.795 |

## Project Structure

```
taka-note-detection/
├── app/
│   ├── inference.py      # Loads best.pt once; predict() returns class, confidence, bbox
│   └── main.py            # FastAPI app exposing GET / and POST /predict
├── data/                  # Training/validation/test data (gitignored)
├── models/
│   └── best.pt             # Trained YOLOv11n weights (gitignored)
├── notebooks/
│   └── train.py            # Training script (YOLOv11n, imgsz=416, batch=8, epochs=50)
├── assignment-screenshots/ # Screenshots for submission (training, API tests, Docker)
├── Dockerfile
├── requirements.txt
└── README.md
```

## Requirements

- Docker Desktop (tested on v29.6.2)
- (For local, non-Docker development only) Python 3.11+, see `requirements.txt`

Key dependencies: `ultralytics`, `fastapi`, `uvicorn`, `python-multipart`, `opencv-python`, `pillow`, `numpy`. The Docker image installs the **CPU-only** build of `torch`/`torchvision` from the official PyTorch CPU wheel index, since this model was trained and is served on a CPU-only machine (no NVIDIA GPU) — this keeps the image small and the build fast.

## Building the Docker Image

From the project root (the folder containing `Dockerfile`):

```bash
docker build -t taka-note-detection .
```

This installs system dependencies (`libgl1`, `libglib2.0-0` for OpenCV), Python dependencies, and copies the application code (`app/`) and trained weights (`models/`) into the image. Resulting image size: ~622MB.

## Running the Container

```bash
docker run -d -p 8000:8000 --name taka-container taka-note-detection
```

- `-d` runs it in the background
- `-p 8000:8000` maps container port 8000 to host port 8000
- `--name taka-container` names the container for easy reference

Check it's running:

```bash
docker ps
```

Stop/remove it later with:

```bash
docker stop taka-container
docker rm taka-container
```

## API Usage

### Health check

```bash
curl.exe http://localhost:8000
```

### Predict on an image

```bash
curl.exe -X POST http://localhost:8000/predict -F "file=@path\to\image.jpg"
```

**Example response:**

```json
{
  "filename": "5_6_jpg.rf.xxxx.jpg",
  "detections": [
    {
      "class_name": "Five Taka",
      "confidence": 0.94,
      "bbox": [x1, y1, x2, y2]
    }
  ]
}
```

If no note is detected above the confidence threshold, `detections` is an empty list. Invalid (non-image) uploads return `400`; internal inference errors return `500`.

### Using Postman

- Method: `POST`
- URL: `http://localhost:8000/predict`
- Body → `form-data` → key `file`, type `File`, value: choose an image

## Notes on Confidence Threshold

The API uses `conf_threshold=0.5` to reduce false positives in production. This means some genuine detections with confidence between 0.25 and 0.5 (visible when running `inference.py` directly at the lower threshold) won't appear in the API response. This is an intentional precision/recall trade-off, not a bug.

## Testing

The `/predict` endpoint was manually tested with 5 different images via `curl`, covering both single-note and multi-note detection cases, correct handling of low-confidence/no-detection cases, and confirming that Docker container responses match local (non-Docker) inference results for the same input image.