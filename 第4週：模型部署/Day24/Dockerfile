# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
FROM ubuntu:23.10

# Install basic dependencies
RUN apt update && apt install --no-install-recommends -y \
    python3-pip git zip curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Download required fonts (use curl instead of ADD)
RUN curl -L -o /root/.config/Ultralytics/Arial.ttf https://ultralytics.com/assets/Arial.ttf && \
    curl -L -o /root/.config/Ultralytics/Arial.Unicode.ttf https://ultralytics.com/assets/Arial.Unicode.ttf

# Remove EXTERNALLY-MANAGED error for Ubuntu Nightly
RUN rm -rf /usr/lib/python3.11/EXTERNALLY-MANAGED

# Upgrade pip and install dependencies
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip wheel && \
    pip install --no-cache-dir -r requirements.txt \
    albumentations gsutil notebook coremltools onnx onnx-simplifier onnxruntime 'openvino-dev>=2023.0' \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Create working directory
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# Copy project files (ensure local paths are correct)
COPY . /usr/src/app
# 將 yolov5-master 資料夾複製到容器內的 /app/yolov5-master
COPY ./yolov5-master /app/yolov5-master

# Expose port for FastAPI
EXPOSE 8000

# Start FastAPI application using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]


# Usage Examples -------------------------------------------------------------------------------------------------------

# Build and Push
# t=ultralytics/yolov5:latest-cpu && sudo docker build -f utils/docker/Dockerfile-cpu -t $t . && sudo docker push $t

# Pull and Run
# t=ultralytics/yolov5:latest-cpu && sudo docker pull $t && sudo docker run -it --ipc=host -v "$(pwd)"/datasets:/usr/src/datasets $t
