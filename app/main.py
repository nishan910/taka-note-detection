import shutil
import tempfile
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from app.inference import predict

app = FastAPI(title="Taka Note Detection API")

ALLOWED_TYPES = {"image/jpeg", "image/png"}

@app.get("/")
def root():
    return {"status": "ok", "message": "Taka Note Detection API is running"}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are supported")

    suffix = os.path.splitext(file.filename)[1]
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
    except Exception:
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    try:
        results = predict(tmp_path, conf_threshold=0.5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    finally:
        os.remove(tmp_path)

    return {"filename": file.filename, "detections": results}