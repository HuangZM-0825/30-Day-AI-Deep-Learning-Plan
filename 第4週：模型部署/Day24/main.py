from fastapi import FastAPI, UploadFile, File
import uvicorn
from PIL import Image
import io
import torch
import os

# 初始化 FastAPI 應用
app = FastAPI()

# 模型的容器內路徑
model_path = '/app/yolov5-master/runs/train/exp2/weights/best.pt'

# 檢查模型路徑是否存在
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model weights not found at {model_path}")

# 載入 YOLO 模型
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
model.conf = 0.023  # 設定置信度閾值

# 定義一個端點來接受圖片並進行推理
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # 確保 detect 目錄存在
    detect_dir = "/app/yolov5-master/runs/detect"
    os.makedirs(detect_dir, exist_ok=True)

    # 保存上傳的圖片
    image_path = os.path.join(detect_dir, file.filename)
    with open(image_path, "wb") as buffer:
        buffer.write(await file.read())

    # 使用 PIL 讀取圖片
    image = Image.open(image_path)

    # 進行推理
    results = model(image)

    # 返回物件檢測的結果
    detection_results = results.pandas().xyxy[0].to_dict(orient="records")  # 轉換為字典格式
    return {"message": "Detection completed", "predictions": detection_results}

# 運行 FastAPI 應用
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
