"""
Module 17 — Taka Note Detection
Phase-1: Train YOLOv11 on the Bangladeshi Currency Detection dataset.

Run from the project root (with venv activated):
    python notebooks/train.py
"""

from ultralytics import YOLO

def main():
    # Start from the pretrained nano checkpoint (smallest/fastest — good for CPU).
    # Ultralytics downloads this automatically on first run.
    model = YOLO("yolo11n.pt")

    results = model.train(
        data="data/data.yaml",   # path to our dataset config
        epochs=50,                # start here; can resume/extend later if needed
        imgsz=416,                 # smaller than default 640 -> much faster on CPU
        batch=8,                   # small batch size, CPU-friendly
        device="cpu",               # force CPU explicitly
        project="models",           # where run folders get saved
        name="taka_yolov11n",        # run name -> models/taka_yolov11n/
        patience=15,                  # early stopping if no improvement
        workers=2,                     # data-loading threads (keep low on CPU)
        verbose=True,
    )

    print("Training complete. Best weights saved at:")
    print(results.save_dir / "weights" / "best.pt")

if __name__ == "__main__":
    main()
