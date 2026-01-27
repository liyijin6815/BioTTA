# BioTTA 快速开始指南

## 项目结构

```
BioTTA/
├── main.py                    # 主入口脚本（推荐使用）
├── config.yaml                # 配置文件
├── biotta_step1.py            # Step1模块：长度预测
├── biotta_step2.py            # Step2模块：端点定位
├── biotta_output.py           # 输出模块
├── step1_length_prediction.py # Step1原始代码（参考）
├── step2_biometry_TTA.py      # Step2原始代码（参考）
└── lib/                       # 库文件
```

## 快速使用

### 1. 配置文件设置

编辑 `config.yaml`，主要设置：

```yaml
paths:
  step1_model: "./models/step1_length_MinValLoss.pth"
  step2_model: "./models/step2_biometry_MinValLoss.pth"
  template:
    image_folder: "./templates/template_image"
    label_folder: "./templates/template_label"
```

### 2. 单文件处理

```bash
python main.py --input image.nii.gz --config config.yaml
```

### 3. 批量处理

```bash
python main.py --input /path/to/images/ --config config.yaml
```

### 4. 自定义输出

```bash
python main.py --input image.nii.gz --config config.yaml --output ./my_results
```

## 输出结果

结果保存在输出目录中：
- `results.csv`: CSV格式结果（包含11个长度和22个端点坐标）
- `results.json`: JSON格式结果（结构化数据）

### CSV格式说明

- `image_name`: 图像名称
- `age`: 年龄（如果可用）
- `RLV_A`, `LLV_A`, ...: 11个生物测量长度
- `point_1_x`, `point_1_y`, `point_1_z`, ...: 22个端点坐标

### JSON格式说明

```json
[
  {
    "image_name": "image_001",
    "age": 28.5,
    "lengths": {"RLV_A": 12.34, ...},
    "landmarks": [[x1,y1,z1], [x2,y2,z2], ...]
  }
]
```

## 主要改进

1. **统一入口**: 使用 `main.py` 整合Step1和Step2
2. **YAML配置**: 所有参数通过配置文件管理
3. **批量处理**: 支持文件夹批量处理
4. **多格式输出**: 同时支持CSV和JSON
5. **模块化设计**: 每个步骤独立模块，易于维护

## 注意事项

1. 确保模型文件在 `models/` 目录下
2. 确保模板文件在 `templates/` 目录下
3. 如果提供年龄文件，需要配置 `paths.age_file` 和 `data.age_file_format`
4. GPU设置通过 `system.gpu_id` 或命令行 `--gpu` 参数

## 问题反馈

如有问题，请检查：
1. 模型文件路径是否正确
2. 模板文件是否存在
3. 输入文件格式是否正确（.nii.gz 或 .nii）
4. GPU是否可用（如果有CUDA错误）

