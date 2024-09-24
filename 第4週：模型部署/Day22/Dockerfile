# 使用官方 Python 基礎映像
FROM python:3.9-slim

# 設置工作目錄
WORKDIR /app

# 複製當前目錄到容器中
COPY . .

# 安裝必要的 Python 套件
RUN pip install --no-cache-dir fastapi uvicorn torch pillow

# 暴露 8000 端口
EXPOSE 8000

# 運行 FastAPI 應用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
