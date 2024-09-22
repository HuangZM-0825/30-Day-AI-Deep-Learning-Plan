# 使用官方 Python 基礎映像
FROM python:3.9-slim

# 設定工作目錄
WORKDIR /usr/src/app

# 安裝必要的系統依賴（包括 Git 和 OpenCV 的依賴）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt 到容器中
COPY ./requirements.txt ./requirements.txt

# 更新 pip 並安裝必要的依賴項
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 複製 yolov5-master 目錄中的所有文件到容器中
COPY . .

# 設置預設的啟動命令
CMD ["python", "train.py", "--img", "640", "--batch", "16", "--epochs", "50", "--data", "data.yaml", "--weights", "yolov5s.pt"]
