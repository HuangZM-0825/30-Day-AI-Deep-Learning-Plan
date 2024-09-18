import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10

# 下載並預處理 CIFAR-10 資料集
# CIFAR-10 是一個包含 10 種類別的彩色圖像資料集，每個類別有 6000 張圖像
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# 將圖像數據轉換為浮點數並正規化到 0-1 範圍
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# 建立 CNN 模型
model = models.Sequential([
    # 第一個卷積層，使用 32 個 3x3 的卷積核，激活函數為 ReLU，輸入形狀為 (32, 32, 3)
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    # 最大池化層，池化窗口大小為 2x2
    layers.MaxPooling2D((2, 2)),
    # 第二個卷積層，使用 64 個 3x3 的卷積核，激活函數為 ReLU
    layers.Conv2D(64, (3, 3), activation='relu'),
    # 最大池化層，池化窗口大小為 2x2
    layers.MaxPooling2D((2, 2)),
    # 第三個卷積層，使用 64 個 3x3 的卷積核，激活函數為 ReLU
    layers.Conv2D(64, (3, 3), activation='relu'),
    # 展開層，將多維數據展開為一維
    layers.Flatten(),
    # 全連接層，包含 64 個神經元，激活函數為 ReLU
    layers.Dense(64, activation='relu'),
    # 輸出層，包含 10 個神經元，對應 10 個類別，激活函數為 softmax
    layers.Dense(10, activation='softmax')
])

# 編譯模型
# 使用 Adam 優化器，損失函數為 sparse_categorical_crossentropy，評估指標為準確率
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 訓練模型
# 使用訓練數據進行訓練，訓練 10 個 epoch，每批次大小為 64
model.fit(x_train, y_train, epochs=10, batch_size=64)

# 評估模型
# 使用測試數據進行評估，並輸出測試準確率
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')
