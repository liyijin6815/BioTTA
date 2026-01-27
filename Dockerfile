# BioTTA Docker镜像
# 使用PyTorch官方基础镜像，支持CUDA
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 安装ANTs (Advanced Normalization Tools)
# ANTspy需要从源码编译，但为了简化可以使用预编译版本或跳过
# 如果ANTs不可用，可以注释掉相关的图像配准功能
RUN pip install --no-cache-dir \
    antspyx \
    || echo "Warning: ANTspy installation failed, template registration may not work"

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 设置Python路径
ENV PYTHONPATH=/app:$PYTHONPATH

# 默认命令
CMD ["python", "main.py", "--help"]

