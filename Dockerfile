# Base image: Python 3.11 slim — stable, well-supported by torch/ultralytics
FROM python:3.11-slim

# System-level dependency needed by OpenCV (opencv-python) at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /code

# Copy only requirements first (better Docker layer caching —
# dependencies won't reinstall every time app code changes)
COPY requirements.txt .

# Install CPU-only torch/torchvision first (from PyTorch's CPU-only index).
# This avoids pip pulling the huge CUDA/GPU build of torch (several GB),
# which isn't needed since this machine has no NVIDIA GPU.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install the remaining Python dependencies.
# torch/torchvision are already satisfied above, so pip won't redownload them.
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and trained model into the container
COPY app/ ./app/
COPY models/ ./models/

# Expose the port FastAPI/uvicorn will run on
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]