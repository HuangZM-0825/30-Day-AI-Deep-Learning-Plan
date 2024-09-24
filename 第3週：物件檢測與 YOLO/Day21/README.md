```markdown
# YOLO 專案

本專案包含 YOLOv5 的訓練、微調、即時檢測和模型評估等內容。

## 安裝

請確保已安裝以下依賴：

- Python 3.8+
- PyTorch
- OpenCV
```

## 資料預處理

在`Day17/DataPreprocessing.ipynb`中，我們將 CelebA 資料集過濾並轉換為 YOLO 格式。

## 訓練 YOLOv5

在`Day17/YOLOv5.ipynb`中，我們展示了如何使用 YOLOv5 進行訓練。請確保配置好`data.yaml`文件。

## 微調 YOLOv5

在`Day18/YOLO 微調.ipynb`中，我們介紹了如何微調 YOLOv5 的超參數以提高模型性能。

## 即時物件檢測

在`Day19/即時物件檢測.ipynb`物件檢測與 YOLO\Day19\即時物件檢測.ipynb") 中，我們展示了如何使用本地攝影機進行即時物件檢測。

## 模型評估

在`Day20/物件檢測模型評估.ipynb`中，我們介紹了如何使用 mAP 和 IoU 等指標評估模型性能。相關代碼在 [`Day20/mAP.py`]。

## 貢獻

歡迎提交 Pull Request 或 Issue 來改進本專案。