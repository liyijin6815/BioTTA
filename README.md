# BioTTA: Fetal Biometry and Landmark Localization Tool

## Overview

BioTTA is a source-free, unsupervised Test-Time Adaptation (TTA) framework designed for robust automatic fetal brain biometry.



Deep learning models often suffer from performance degradation when deployed in unseen clinical environments due to domain shifts caused by multi-center acquisition, different SRR methods and variable pathology. BioTTA addresses this challenge by adapting pre-trained models to out-of-distribution (OoD) target data during inference, without requiring manual annotations.



The framework operates in two stages:



**Pre-training**: A unified encoder-decoder architecture is trained, where a shared encoder feeds into two parallel decoders for landmark heatmap prediction and direct biometric measurement regression respectively.



**Test-Time Adaptation**: The model adapts to individual unlabeled test samples by minimizing entropy, length consistency, and boundary-gradient losses, while incorporating atlas-informed anatomical priors via registration .


![fig](https://github.com/user-attachments/assets/7ad4fa3f-72b3-48ea-9cec-57d56736ea1c)



To facilitate clinical translation, we further developed an automated web-based reporting system built upon the BioTTA framework. This end-to-end tool allows clinicians to upload 3D fetal MRI volumes (DICOM/NIfTI) and gestational age, automatically executing the full pipeline from preprocessing to biometry. The system generates an interactive HTML dashboard featuring a 3D viewer for slice navigation and quantitative analysis, where predicted measurements are dynamically mapped to standard growth trajectories to assist in risk stratification for developmental anomalies. Reports can be exported as standardized PDFs for clinical archiving, streamlining the diagnostic workflow and providing objective, consistent quantitative evidence for multi-center research.


https://github.com/user-attachments/assets/a05f2f41-c7ac-483d-a41b-3968ff314139


## Key Features

1. **Precise & Consistent Biometry**: Delivers highly accurate quantification for 11 standard clinical fetal brain biometric measurements. Comparison experiments demonstrate the superior consistency and performance compared to baselines.

2. **Robust Domain Generalization**: Mitigates domain shifts arising from diverse centers, scanner manufacturers, reconstruction methods (e.g., NiftyMIC, NeSVoR), and pathological conditions (e.g., GMH-IVH, VM, and cerebellar abnormalities).

3. **Source-Free & Unsupervised**: Performs adaptation on the target domain without accessing source data or requiring target ground-truth labels.

4. **End-to-End Clinical Reporting System**: Provides a unified framework designed to directly assist radiologists in clinical reading and diagnosis. By seamlessly integrating measurement with automated reporting, the system significantly streamlines the diagnostic workflow and supports objective decision-making.

   
## Installation

### Option 1: Install via pip

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install torch torchvision
pip install nibabel
pip install pandas numpy
pip install natsort
pip install pyyaml
pip install matplotlib
pip install antspyx  # For image registration
pip install timm     # For model architecture
```

### Option 2: Docker (Recommended)

Using Docker ensures environment consistency and reproducibility. For detailed instructions, please refer to DOCKER.md.

**Quick Start:**

```bash
# build docker
docker build -t biotta:latest .

# run with GPU
docker run --rm --gpus all \
  -v $(pwd)/models:/app/models:ro \
  -v $(pwd)/templates:/app/templates:ro \
  -v /path/to/input:/data/input:ro \
  -v $(pwd)/results:/data/output \
  biotta:latest \
  python main.py --input /data/input/image.nii.gz --output /data/output
```



## Directory Structure

```
BioTTA/
├── main.py                  # Main execution script
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies list
├── Dockerfile              # Docker image build file
├── .dockerignore           # Docker ignore file
├── DOCKER.md               # Docker usage documentation
├── biotta_step1.py         # Step1 module
├── biotta_step2.py         # Step2 module
├── biotta_output.py        # Output module
├── lib/                    # Dependency library
│   ├── step1_length_prediction.py
│   ├── step2_source_network.py
│   └── step2_supp.py
├── models/                 # Model files (to be prepared by user)
│   ├── step1_length_MinValLoss.pth
│   └── step2_biometry_MinValLoss.pth
├── templates/              # Template files (to be prepared by user)
│   ├── template_image/
│   └── template_label/
└── results/                # Output results
    ├── step1_lengths.csv   # Step1 results (if enabled)
    ├── results.json        # Final JSON results
    └── {image_name}/       # Intermediate results for each sample
        ├── landmarks.txt   # Endpoint coordinates text (if enabled)
        ├── landmarks.png   # Visualization image (if enabled)
        ├── length_centile_web.png # Length percentile chart web version (if enabled)
        ├── length_centile.csv # Length percentile table (if enabled)
        ├── length_centile.png # Length percentile chart PDF version (if enabled)
        └── tent_TTA.pth    # TTA model (if enabled)
```



## Usage

### 1. Configuration

First, configure `config.yaml` according to your environment:

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

### 2. Running Analysis

Processing a Single File:

```bash
python main.py --input /data/lyj/dataset/fetal_brain_localzation_dataset/FeTA_dataset/registered/test_2/sub-028.nii.gz --age 31 --registered_folder ./templates_registered/sub-028 --output ./test_results
```

Processing a Folder:

```bash
python main.py --input /data/lyj/dataset/fetal_brain_localzation_dataset/WCB_cerebellar_abnormality/registered/vermian_ageneses --age_csv /data/lyj/dataset/fetal_brain_localzation_dataset/WCB_cerebellar_abnormality/WCB_vermian_ageneses_age.csv --age_file_format '{"name_column": "name", "age_column": "age"}' --registered_folder ./templates_registered/vermian_ageneses --output ./test_results/vermian_ageneses 
```

Using a Custom Config:

```bash
python main.py --input path/to/sample.nii.gz --config my_config.yaml
```

Setting GPU:

```bash
python main.py --input path/to/sample.nii.gz --gpu 0
```

### 3. Age File Format (Optional)

To use Atlas constraints, you can provide an age file (CSV format):

```csv
name,age
sample1,23.5
sample2,24.0
```

**Age Priority Logic:**

1. Command Line Argument: `--age` (Single file only) - Highest priority.

2. Age CSV File: Matches the image name to the age column.

3. Default: If neither is provided, atlas constraints are disabled (TTA only).

   

## Output Format

### JSON Output

The system outputs a structured JSON file (`results.json`):

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

Entry structure:

- `image_name`: String.
- `age`: Float (if provided).
- `lengths`: Dictionary containing 11 biometric measurements (RLV_A, LLV_A, RLV_C, LLV_C, BBD, CBD, TCD, FOD, AVD, VH, CCL).
- `landmarks`: List of 22 3D coordinates `[x, y, z]`.

### Intermediate Results

Based on the `save_intermediate` settings in the config, the following can be saved:

- `step1_results`: Step 1 length prediction CSV.
- `{image_name}/heatmaps`: Heatmaps after TTA training.
- `{image_name}/tent_TTA.pth`: The adapted TTA model per sample.
- `{image_name}/landmarks.txt`: Raw landmark coordinates (text format).
- `{image_name}/landmarks.png`: Visualized result image.
