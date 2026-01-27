# BioTTA Docker 使用指南

## 概述

本指南介绍如何使用Docker容器运行BioTTA，确保环境一致性和易于复现。

## 前置要求

- Docker (版本 20.10+)
- NVIDIA Docker (如果使用GPU，需要安装 nvidia-docker2)
- 至少 20GB 可用磁盘空间

## 快速开始

### 1. 构建Docker镜像

```bash
cd /data/birth/lmx/work/Class_projects/lyj/work/fetal_localization_SFUDA/BioTTA
docker build -t biotta:latest .
```

### 2. 准备数据和模型

在运行容器前，确保你有以下文件：
- **模型文件**: `models/step1_length_MinValLoss.pth` 和 `models/step2_biometry_MinValLoss.pth`
- **模板文件**: `templates/template_image/` 和 `templates/template_label/`
- **输入数据**: 你的 `.nii.gz` 文件

### 3. 运行容器

#### GPU模式

```bash
docker run --rm --gpus all \
  -v /path/to/your/data:/data/input \
  -v /path/to/output:/data/output \
  -v /path/to/models:/app/models \
  -v /path/to/templates:/app/templates \
  biotta:latest \
  python main.py --input /data/input/image.nii.gz --output /data/output --gpu 0
```

#### 使用自定义配置文件

```bash
docker run --rm --gpus all \
  -v /path/to/your/data:/data/input \
  -v /path/to/output:/data/output \
  -v /path/to/models:/app/models \
  -v /path/to/templates:/app/templates \
  -v /path/to/config.yaml:/app/config.yaml \
  biotta:latest \
  python main.py --input /data/input/image.nii.gz --config /app/config.yaml --output /data/output
```

#### 指定年龄值

```bash
docker run --rm --gpus all \
  -v /path/to/your/data:/data/input \
  -v /path/to/output:/data/output \
  -v /path/to/models:/app/models \
  -v /path/to/templates:/app/templates \
  biotta:latest \
  python main.py --input /data/input/image.nii.gz --age 24 --output /data/output
```

## 数据卷说明

建议将以下目录通过数据卷挂载：

| 宿主机路径 | 容器内路径 | 说明 |
|-----------|-----------|------|
| `/path/to/models` | `/app/models` | 模型文件目录 |
| `/path/to/templates` | `/app/templates` | 模板文件目录 |
| `/path/to/input` | `/data/input` | 输入数据目录 |
| `/path/to/output` | `/data/output` | 输出结果目录 |

## 完整示例

```bash
# 1. 构建镜像
docker build -t biotta:latest .

# 2. 运行处理单个文件
docker run --rm --gpus all \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/templates:/app/templates:ro \
  -v /data/input:/data/input:ro \
  -v $(pwd)/results:/data/output \
  biotta:latest \
  python main.py \
    --input /data/input/sample.nii.gz \
    --age 24 \
    --output /data/output \
    --gpu 0

# 3. 批量处理文件夹
docker run --rm --gpus all \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/templates:/app/templates:ro \
  -v /data/input_folder:/data/input:ro \
  -v $(pwd)/results:/data/output \
  biotta:latest \
  python main.py \
    --input /data/input \
    --output /data/output \
    --gpu 0
```





