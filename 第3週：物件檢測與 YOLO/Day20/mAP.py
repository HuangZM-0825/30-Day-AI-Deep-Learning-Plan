import os
from pathlib import Path

# 設定 YOLOv5 的工作目錄
yolov5_dir = Path(r"D:\Learning_Python\30-Day_AI_Deep_Learning_Plan\yolov5-master")

# 切換到 YOLOv5 目錄
os.chdir(yolov5_dir)

# 運行 YOLOv5 的驗證腳本來評估 mAP
os.system(
    'python val.py --weights "D:/Learning_Python/30-Day_AI_Deep_Learning_Plan/yolov5-master/runs/train/exp2/weights/best.pt" '
    '--data "D:/Learning_Python/30-Day_AI_Deep_Learning_Plan/第3週：物件檢測與 YOLO/Day17/data.yaml" '
    '--img 640 '
    '--iou 0.5 '
    '--task val'
)