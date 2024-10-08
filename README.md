### 30天 AI 工程師深度學習訓練計畫

#### 目標：
幫助你準備 AI 工程師面試，專注於深度學習。這個計畫涵蓋理論、程式實作與專案開發，並且最後可以在 GitHub 上展示你的專案。

### 30天 AI 工程師深度學習訓練計畫
#### 第1週：深度學習與神經網路基礎
- **目標：** 建立深度學習基礎。
- **專案：** 使用卷積神經網路（CNN）進行圖像分類。

| 天數 | 主題 | 任務 | 時間（小時） |
|-----|-------|------|--------------|
| 1   | 深度學習簡介 | 學習深度學習概念（感知器、梯度下降、過擬合），並使用 TensorFlow 或 PyTorch 實作一個基本的神經網路。 | 4 |
| 2   | 神經網路簡介 | 深入理解反向傳播、激活函數與優化器。實作一個簡單的多層感知器（MLP）網路。 | 4 |
| 3   | 卷積神經網路（CNN） | 學習 CNN 結構及其在圖像資料中的應用。實作一個簡單的 CNN 進行圖像分類（MNIST/CIFAR-10 資料集）。 | 4 |
| 4   | 實際應用 CNN | 優化 CNN 模型，介紹池化和 dropout 等正則化技術。在專案中實作基本 CNN。 | 4 |
| 5   | 超參數與優化 | 了解學習率、批量大小及優化器（如 SGD、Adam），調整專案中的 CNN 模型超參數。 | 4 |
| 6   | 模型評估與指標 | 學習精確度、召回率、F1 分數及 ROC 曲線。實作這些指標並評估 CNN 的效能。 | 4 |
| 7   | GitHub 設定與專案文檔撰寫 | 設定 GitHub 倉庫，撰寫 CNN 圖像分類器的 README 說明文件。 | 4 |

#### 第2週：進階神經網路與遷移學習
- **目標：** 探索進階架構並使用遷移學習提高效率。
- **專案：** 使用預訓練模型（ResNet、VGG）進行圖像分類的遷移學習。

| 天數 | 主題 | 任務 | 時間（小時） |
|-----|-------|------|--------------|
| 8   | 進階 CNN | 學習 ResNet、VGG、Inception 等進階 CNN 架構，修改專案中的 CNN 模型。 | 4 |
| 9   | 遷移學習概述 | 了解遷移學習和預訓練模型的概念。微調 ResNet 或 VGG 預訓練模型，應用於自定義資料集。 | 4 |
| 10  | 實作遷移學習 | 使用 Keras 或 PyTorch 中的預訓練模型，進行遷移學習。 | 4 |
| 11  | 資料增強技術 | 了解如何使用資料增強技術改善模型，並在專案中實作旋轉、翻轉、縮放等增強技術。 | 4 |
| 12  | 進階優化 | 探索權重衰減、學習率調整等進階優化技術，並應用於專案中。 | 4 |
| 13  | 微調模型 | 微調預訓練模型，提高準確性，探索凍結與解凍層技術。 | 4 |
| 14  | GitHub 更新與文檔撰寫 | 更新 GitHub 上的專案。 | 4 |

#### 第3週：物件檢測與 YOLO
- **目標：** 學習並應用 YOLO 進行物件檢測。
- **專案：** 實作 YOLOv5 進行即時物件檢測。

| 天數 | 主題 | 任務 | 時間（小時） |
|-----|-------|------|--------------|
| 15  | 物件檢測簡介 | 學習物件檢測基本概念（邊界框、IoU、錨點框），熟悉 YOLO 架構。 | 4 |
| 16  | YOLO 架構與設定 | 設置 YOLOv5，探索其結構與即時物件檢測的能力。 | 4 |
| 17  | 實作 YOLOv5 | 在自定義資料集或預先存在的資料集（如 COCO）上應用 YOLOv5，訓練模型並進行物件檢測。 | 4 |
| 18  | YOLO 微調 | 微調 YOLO 超參數以提高檢測準確性。 | 4 |
| 19  | 即時物件檢測 | 實作即時物件檢測，使用攝影機或影片流，並將其展示於專案中。 | 4 |
| 20  | 物件檢測模型評估 | 使用 mAP（平均準確度）等指標評估你的模型。 | 4 |
| 21  | GitHub 更新與文檔撰寫 | 更新 YOLO 專案，包括即時檢測腳本和評估指標。 | 4 |

#### 第4週：部署與面試準備
- **目標：** 部署模型。
- **專案：** 將 CNN 或 YOLO 模型部署成 API 或 Web 介面。

| 天數 | 主題 | 任務 | 時間（小時） |
|-----|-------|------|--------------|
| 22  | 部署 CNN 模型 | 將你的圖像分類 CNN 模型作為 API 使用 FastAPI 部署。 | 4 |
| 23  | 模型部署 | 學習使用 FastAPI 與建立 Docker 容器部署模型。 | 4 |
| 24  | 模型部署到AWS | 探索雲端平台 AWS 上部署模型的選項，並將模型部署到 AWS Elastic Beanstalk，應用工具包含 FastAPI 和 Docker。 | 4 |
| 25  | Grad-CAM 原理及 CNN 應用 | 將學習如何使用 Grad-CAM 來解釋 YOLO 模型對圖像的物件檢測過程，生成可視化的“熱圖”，以直觀展示模型的關注區域。 | 4 |
| 26  | SHAP 原理及應用 | 分析數據表中的結構化數據特徵，幫助解釋特徵對預測的影響。
| 27  | 模型監控與維護 | 了解如何監控部署後的模型效能，包括檢測模型漂移和性能瓶頸。學習自動化監控流程，以確保模型持續穩定運行並及時調整。 | 4 |
| 28  | 模型優化與效能提升 | 探索模型優化技術（如剪枝、量化、混合精度訓練），並嘗試應用在你之前的 YOLO 模型上，以提升效能。 | 4 |
| 29  | 面試準備：概念回顧 | 回顧所有深度學習概念：反向傳播、CNN、YOLO、遷移學習。練習解釋關鍵術語。 | 4 |
| 30  | 最後回顧 | 練習深度學習和 AI 工程師相關的常見面試問題。 | 4 |
