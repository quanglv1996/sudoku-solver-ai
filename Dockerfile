# Sudoku Solver AI - image GPU (YOLO/PyTorch + PaddleOCR/Paddle trên CUDA 12.6)
# Thư viện CUDA/cuDNN được cài qua pip (gói nvidia-*) cùng torch và paddle,
# nên base image chỉ cần CUDA "base"; driver lấy từ máy host qua NVIDIA Container Toolkit.
ARG CUDA_IMAGE=nvidia/cuda:12.6.3-base-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgl1/libglib2.0-0: opencv-python; libgomp1: paddle
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.10 python3-pip libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Cài framework nặng trước để cache layer khi chỉ sửa code / requirements khác
ARG CUDA_TAG=cu126
RUN pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/${CUDA_TAG}
RUN pip install paddlepaddle-gpu==3.2.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/${CUDA_TAG}/ \
    --extra-index-url https://pypi.org/simple

# Bỏ các gói đã cài ở trên và gói chế độ "Màn hình PC" (container không có màn hình)
COPY requirements.txt .
RUN grep -vE '^(torch|torchvision|paddlepaddle-gpu|mss|pyautogui)[=<>]' requirements.txt > /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt

COPY . .

# Model PaddleOCR đã cache trong volume → bỏ bước kiểm tra kết nối tới nơi chứa model mỗi lần khởi động
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    IN_DOCKER=1
RUN mkdir -p /root/.config/Ultralytics

EXPOSE 5000 5443

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"HTTP_PORT\", 5000)}/api/health', timeout=4)"

CMD ["python", "run_web.py"]
