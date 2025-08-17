# Base: small Python image (CPU)
FROM python:3.10-slim

# OS packages: git (for torch.hub/clone), and libs for OpenCV/Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ffmpeg libsm6 libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Writable config/cache; unbuffered logs
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics \
    TORCH_HOME=/root/.cache/torch \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Avoid startup races
RUN mkdir -p $TORCH_HOME/hub $YOLO_CONFIG_DIR

# Install PyTorch CPU first (stable wheels)
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# Install your Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone YOLOv5 so we can load from local source (no network needed at runtime)
RUN git clone https://github.com/ultralytics/yolov5.git /app/yolov5

# Copy app code + model weights
COPY app/ app/
COPY models/ models/

# Flask/Gunicorn port
ENV PORT=8080
EXPOSE 8080

# One worker avoids Torch Hub cache races
CMD ["gunicorn","-w","1","-b","0.0.0.0:8080","app.server:app"]
