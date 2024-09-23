import os
from pathlib import Path

# 設定 YOLOv5 的工作目錄
yolov5_dir = Path("/usr/src/app")

# 切換到 YOLOv5 目錄
os.chdir(yolov5_dir)

# 設定訓練參數
img_size = 640  # 圖片大小
batch_size = 8  # 批次大小
epochs = 50  # epoch 次數
data_yaml = "/usr/src/app/Day17/data.yaml"  # data.yaml 文件路徑
weights = "/usr/src/app/weights/yolov5s.pt"  # 預訓練權重路徑

# 執行 YOLOv5 訓練
os.system(f"python3 train.py --img {img_size} --batch {batch_size} --epochs {epochs} --data {data_yaml} --weights {weights}")
