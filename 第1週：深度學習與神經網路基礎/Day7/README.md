```markdown
# CIFAR-10 CNN 圖像分類器

這是一個使用 TensorFlow 和 Keras 建立的卷積神經網路 (CNN) 模型，用於對 CIFAR-10 資料集進行圖像分類。CIFAR-10 資料集包含 10 種類別的彩色圖像，每個類別有 6000 張圖像。

## 環境設置

在開始之前，請確保已經安裝以下軟體包：

- TensorFlow
- NumPy

您可以使用以下命令安裝這些軟體包：

```sh
pip install tensorflow numpy
```

## 使用方法

### 1. 下載並預處理資料集

首先，下載並預處理 CIFAR-10 資料集。將圖像數據轉換為浮點數並正規化到 0-1 範圍。

### 2. 建立 CNN 模型

建立一個卷積神經網路模型，包括三個卷積層和兩個最大池化層，最後是全連接層和輸出層。

### 3. 編譯模型

使用 Adam 優化器，損失函數為 [`sparse_categorical_crossentropy`](command:_github.copilot.openSymbolFromReferences?%5B%22%22%2C%5B%7B%22uri%22%3A%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2FD%3A%2FLearning_Python%2F30-Day%20AI%20Deep%20Learning%20Plan%2F%E7%AC%AC1%E9%80%B1%EF%BC%9A%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E8%88%87%E7%A5%9E%E7%B6%93%E7%B6%B2%E8%B7%AF%E5%9F%BA%E7%A4%8E%2FDay7%2FCNN.py%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%2C%22pos%22%3A%7B%22line%22%3A34%2C%22character%22%3A20%7D%7D%5D%2C%228543192e-8c72-47d5-9b7f-7b3dd6bc8dcc%22%5D "Go to definition")，評估指標為準確率。

### 4. 訓練模型

使用訓練數據進行訓練，訓練 10 個 epoch，每批次大小為 64。

### 5. 評估模型

使用測試數據進行評估，並輸出測試準確率。

## 程式碼

以下是完整的程式碼：

```python
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

# 下載並預處理 CIFAR-10 資料集
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 建立 CNN 模型
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 編譯模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
model.fit(x_train, y_train, epochs=10, batch_size=64)

# 評估模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')
```

## 結果

訓練完成後，模型會輸出測試數據的準確率。這可以用來評估模型在未見過的數據上的表現。

## 參考資料

- [TensorFlow 官方網站](https://www.tensorflow.org/)
- [Keras 官方網站](https://keras.io/)
- [CIFAR-10 資料集](https://www.cs.toronto.edu/~kriz/cifar.html)