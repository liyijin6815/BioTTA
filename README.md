# BioTTA: 胎儿生物测量和端点定位工具

## 整体介绍

BioTTA 是一个整合的胎儿生物测量分析工具，能够自动预测11个解剖长度和22个端点的精确位置。
- **Step 1: 长度预测** - 使用深度学习模型粗精度预测11个解剖长度
- **Step 2: TTA端点定位** - 使用Test-Time Adaptation (TTA) 和图谱约束预测22个端点的精确坐标
- **Step 3: 可视化** - 可视化预测结果

## 功能特点

- **端到端运行** - 整合两个步骤，直接输出最终结果
- **可调整配置** - 通过YAML配置文件轻松调整参数
- **JSON格式输出** - 输出结构化的JSON格式结果
- **批量处理** - 支持单个文件或整个文件夹批量处理
- **灵活年龄输入** - 支持通过年龄文件或命令行参数直接指定年龄

## 安装依赖

### 方式1: 使用pip直接安装

```bash
pip install -r requirements.txt
```

或者手动安装：

```bash
pip install torch torchvision
pip install nibabel
pip install pandas numpy
pip install natsort
pip install pyyaml
pip install matplotlib
pip install antspyx  # 用于图像配准
pip install timm  # 用于网络结构
```

### 方式2: 使用Docker（推荐）

使用Docker可以确保环境一致性，便于复现。详细说明请参考 [DOCKER.md](DOCKER.md)。

**快速开始：**

```bash
# 构建镜像
docker build -t biotta:latest .

# 运行（GPU模式）
docker run --rm --gpus all \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/templates:/app/templates:ro \
  -v /path/to/input:/data/input:ro \
  -v $(pwd)/results:/data/output \
  biotta:latest \
  python main.py --input /data/input/image.nii.gz --output /data/output
```

## 目录结构

```
BioTTA/
├── main.py                  # 主运行脚本
├── config.yaml              # 配置文件
├── requirements.txt         # Python依赖包列表
├── Dockerfile               # Docker镜像构建文件
├── .dockerignore           # Docker忽略文件
├── DOCKER.md               # Docker使用文档
├── biotta_step1.py         # Step1模块
├── biotta_step2.py         # Step2模块
├── biotta_output.py        # 输出模块
├── lib/                     # 依赖库
│   ├── step1_length_prediction.py
│   ├── step2_source_network.py
│   └── step2_supp.py
├── models/                  # 模型文件（需要用户准备）
│   ├── step1_length_MinValLoss.pth
│   └── step2_biometry_MinValLoss.pth
├── templates/               # 模板文件（需要用户准备）
│   ├── template_image/
│   └── template_label/
└── results/                 # 输出结果
    ├── step1_lengths.csv    # Step1结果（如果启用）
    ├── results.json          # 最终JSON结果
    └── {image_name}/         # 每个样本的中间结果
        ├── landmarks.txt      # 端点坐标文本（如果启用）
        ├── landmarks.png      # 可视化图像（如果启用）
        ├── length_centile_web.png # 长度百分位图网页版（如果启用）
        ├── length_centile.csv # 长度百分位表格（如果启用）
        ├── length_centile.png # 长度百分位图pdf版（如果启用）
        └── tent_TTA.pth       # TTA模型（如果启用）
```

### 模型文件

需要准备以下模型文件：

1. **Step1模型** (`step1_model`): 长度预测模型 (.pth文件)
   - 默认路径: `./models/step1_length_MinValLoss.pth`

2. **Step2模型** (`step2_model`): TTA基础模型 (.pth文件)
   - 默认路径: `./models/step2_biometry_MinValLoss.pth`

### 模板文件

需要准备模板图像和标签文件（用于图谱约束）：

- **模板图像文件夹**: `./templates/template_image/`
  - 格式: `STA{age}.nii.gz` (例如: `STA23.nii.gz`, `STA24.nii.gz` ...)
- **模板标签文件夹**: `./templates/template_label/`
  - 格式: `{age}_m.nii.gz` (例如: `23_m.nii.gz`, `24_m.nii.gz` ...)

如果模板配准失败（通常是因为年龄不在22周以上），系统将不使用模板约束，仅使用TTA进行预测。

## 使用方法

### 1. 配置文件

首先，根据您的环境配置 `config.yaml`：

```yaml
paths:
  input_data: ""  # 如果为空，需要在命令行中指定
  output_dir: "./results"
  step1_model: "./models/step1_length_MinValLoss.pth"
  step2_model: "./models/step2_biometry_MinValLoss.pth"
  template:
    image_folder: "./templates/template_image"
    label_folder: "./templates/template_label"
```

### 2. 运行分析

#### 处理单个文件

```bash
python main.py --input /data/lyj/dataset/fetal_brain_localzation_dataset/FeTA_dataset/registered/test_2/sub-028.nii.gz --age 31 --registered_folder ./templates_registered/sub-028 --output ./test_results
```

#### 处理整个文件夹

```bash
python main.py --input /data/lyj/dataset/fetal_brain_localzation_dataset/WCB_cerebellar_abnormality/registered/vermian_ageneses --age_csv /data/lyj/dataset/fetal_brain_localzation_dataset/WCB_cerebellar_abnormality/WCB_vermian_ageneses_age.csv --age_file_format '{"name_column": "name", "age_column": "age"}' --registered_folder ./templates_registered/vermian_ageneses --output ./test_results/vermian_ageneses 
```

#### 使用自定义配置文件

```bash
python main.py --input path/to/sample.nii.gz --config my_config.yaml
```

#### 设置GPU

```bash
python main.py --input path/to/sample.nii.gz --gpu 0
```

### 3. 年龄文件格式（可选）

如果需要使用图谱约束，可以提供一个年龄文件（CSV格式）：

```csv
name,age
sample1,23.5
sample2,24.0
```

年龄的获取优先级：
1. **命令行参数 `--age`**（仅单个文件时有效）- 最高优先级
2. **年龄文件** - 从CSV文件中匹配图像名称对应的年龄
3. **默认行为** - 如果都未提供，系统将无法使用模板约束（仅使用TTA）

## 输出格式

### JSON输出

系统输出JSON格式文件（`results.json`），包含结构化的结果：

```json
[
  {
    "image_name": "sample1",
    "age": 24.0,
    "lengths": {
      "RLV_A": 12.5,
      "LLV_A": 12.3,
      "RLV_C": 8.2,
      ...
    },
    "landmarks": [
      [x1, y1, z1],
      [x2, y2, z2],
      ...
    ]
  }
]
```

每个条目的结构：
- `image_name`: 图像名称（字符串）
- `age`: 年龄值（浮点数，如果提供了年龄信息）
- `lengths`: 字典，包含11个长度测量值（RLV_A, LLV_A, RLV_C, LLV_C, BBD, CBD, TCD, FOD, AVD, VH, CCL）
- `landmarks`: 列表，包含22个端点的3D坐标 `[x, y, z]`

### 中间结果

根据配置文件中的 `save_intermediate` 设置，系统会保存以下中间结果：

- `step1_results`: Step1的长度预测结果CSV（如果 `step1_results: true`）
- `{image_name}/heatmaps`: TTA训练后的热图（如果 `step2_tta_heatmaps: true`）
- `{image_name}/tent_TTA.pth`: TTA训练后的模型（如果 `step2_tta_models: true`）
- `{image_name}/landmarks.txt`: 每个样本的端点坐标（文本格式，如果 `step2_landmarks_txt: true`）
- `{image_name}/landmarks.png`: 可视化图像（如果 `step2_landmarks_image: true`）


## 配置参数说明

### Step1 参数

- `batch_size`: 批次大小
- `model`: 模型配置（通道数、类别数等）

### Step2 参数

- `init_temp`: 温度参数（用于可微分地标检测）
- `topk`: TopK采样
- `learning_rate`: 学习率
- `num_epoch`: TTA训练轮数
- `lambda_entropy_loss`: 熵损失权重
- `lambda_length_loss`: 长度损失权重
- `lambda_boundarygrad_loss`: 边界梯度损失权重
- `radius`: 每个长度对应的搜索半径
- `template_label_scale`: 模板标签缩放
- `template_label_pan`: 模板标签平移

### 系统配置

- `gpu_id`: 使用的GPU ID
- `num_workers`: 数据加载器工作进程数
- `seed`: 随机种子
- `save_intermediate`: 中间结果保存配置
  - `step1_results`: 是否保存Step1的长度预测结果CSV
  - `step2_heatmaps`: 是否保存Step2的热图（会占用大量空间）
  - `step2_tta_models`: 是否保存每个样本的TTA模型
  - `step2_landmarks_txt`: 是否保存原始的landmarks.txt文件
  - `step2_landmarks_image`: 是否保存landmarks.png可视化图像

### 输出配置

- `format`: 输出格式（现在支持 `json`）
- `json.indent`: JSON输出的缩进（用于格式化）

## Docker部署（推荐）

为了确保环境一致性和便于复现，推荐使用Docker容器运行BioTTA。

详细的使用说明请参考 [DOCKER.md](DOCKER.md) 文件。

**快速示例：**

```bash
# 构建镜像
docker build -t biotta:latest .

# 运行（单个文件，指定年龄）
docker run --rm --gpus all \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/templates:/app/templates:ro \
  -v /path/to/input:/data/input:ro \
  -v $(pwd)/results:/data/output \
  biotta:latest \
  python main.py --input /data/input/sample.nii.gz --age 24 --output /data/output
```

