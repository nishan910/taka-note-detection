"""
Taka Note Detection
Phase-2: Model Integration & Inference Pipeline

Loads the trained YOLOv11 weights and exposes a single function,
predict(), that takes an image and returns detections as a list of
dicts: {class_name, confidence, bbox}.

This module is imported by the REST API (app/main.py) later — it has
no FastAPI/Flask code itself, just the model logic, so it can also be
tested standalone.
"""

from pathlib import Path
from ultralytics import YOLO

# Path to the trained weights (relative to project root).
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "best.pt"

# Load the model once at import time — reused across all requests,
# instead of reloading it on every prediction (which would be slow).
model = YOLO(str(MODEL_PATH))


def predict(image_path: str, conf_threshold: float = 0.25):
    """
    Run inference on a single image.

    Args:
        image_path: path to a JPEG/PNG image on disk.
        conf_threshold: minimum confidence score to keep a detection.

    Returns:
        A list of dicts, one per detected note, e.g.:
        [
            {
                "class_name": "Fifty taka",
                "confidence": 0.93,
                "bbox": {"x1": 34.2, "y1": 12.5, "x2": 210.7, "y2": 150.9}
            },
            ...
        ]
        Returns an empty list if nothing is detected above the threshold.
    """
    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        verbose=False,
    )

    detections = []
    result = results[0]  # single image -> single result

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

        detections.append({
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            },
        })

    return detections


if __name__ == "__main__":
    # Quick manual test: run `python app/inference.py <path_to_image>`
    import sys

    if len(sys.argv) != 2:
        print("Usage: python app/inference.py <path_to_image>")
        sys.exit(1)

    test_image = sys.argv[1]
    output = predict(test_image)

    print(f"Detections for {test_image}:")
    if not output:
        print("  No notes detected.")
    for det in output:
        print(f"  {det['class_name']} — confidence {det['confidence']} — bbox {det['bbox']}")
