from fastapi import FastAPI, UploadFile, File
import uvicorn
from PIL import Image
import io
import torch

# 初始化 FastAPI 應用
app = FastAPI()

# 載入模型（例如 YOLO 模型）
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # 修改為你的模型路徑

# 定義一個端點來接受圖片並進行推理
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read()))
    results = model(image)
    return results.pandas().xyxy[0].to_dict(orient="records")  # 回傳物件檢測結果

# 運行 FastAPI 應用
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
