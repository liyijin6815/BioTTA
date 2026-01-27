# BioTTA: Fetal Biometry and Landmark Localization Tool

## Overview

BioTTA is an integrated tool for fetal biometry analysis, capable of automatically predicting 11 anatomical lengths and precise coordinates for 22 anatomical landmarks.

The workflow consists of three main steps:

- **Step 1: Length Prediction** - Uses deep learning models to perform coarse-level prediction of 11 anatomical lengths.
- **Step 2: TTA Landmark Localization** - Utilizes Test-Time Adaptation (TTA) and Atlas-based constraints to predict the precise coordinates of 22 landmarks.
- **Step 3: Visualization** - Visualizes the prediction results.

## Key Features

- **End-to-End Workflow** - Integrates length prediction and landmark localization into a single pipeline.
- **Configurable** - Easy parameter tuning via YAML configuration files.
- **Structured Output** - Generates results in structured JSON format.
- **Batch Processing** - Supports processing of single files or entire directories.
- **Flexible Age Input** - Supports age specification via external CSV files or direct command-line arguments.

## Installation

### Option 1: Install via pip

Bash

```
pip install -r requirements.txt
```

Or install manually:

Bash

```
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



## Directory Structure



### Model Files

You need to prepare the following model files:

1. **Step 1 Model** (`step1_model`): Length prediction model (`.pth` file).
   - Default path: `./models/step1_length_MinValLoss.pth`
2. **Step 2 Model** (`step2_model`): TTA base model (`.pth` file).
   - Default path: `./models/step2_biometry_MinValLoss.pth`

### Template Files

Template images and labels are required for Atlas constraints:

- **Template Image Folder**: `./templates/template_image/`
  - Format: `STA{age}.nii.gz` (e.g., `STA23.nii.gz`, `STA24.nii.gz` ...)
- **Template Label Folder**: `./templates/template_label/`
  - Format: `{age}_m.nii.gz` (e.g., `23_m.nii.gz`, `24_m.nii.gz` ...)

*Note: If template registration fails (usually because the gestational age is not within the supported range, e.g., >22 weeks), the system will proceed using TTA without atlas constraints.*

## Usage

### 1. Configuration

First, configure `config.yaml` according to your environment:



### 2. Running Analysis

#### Processing a Single File



#### Processing a Folder



#### Using a Custom Config



#### Setting GPU



### 3. Age File Format (Optional)

To use Atlas constraints, you can provide an age file (CSV format):



**Age Priority Logic:**

1. **Command Line Argument `--age`** (Single file only) - Highest priority.
2. **Age CSV File** - Matches the image name to the age column.
3. **Default** - If neither is provided, atlas constraints are disabled (TTA only).

## Output Format

### JSON Output

The system outputs a structured JSON file (`results.json`):



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

## Configuration Parameters

### Step 1 Parameters

- `batch_size`: Input batch size.
- `model`: Model configuration (channels, classes, etc.).

### Step 2 Parameters

- `init_temp`: Temperature parameter (for differentiable landmark detection).
- `topk`: Top-K sampling strategy.
- `learning_rate`: Learning rate for adaptation.
- `num_epoch`: Number of TTA training epochs.
- `lambda_entropy_loss`: Weight for entropy loss.
- `lambda_length_loss`: Weight for length loss.
- `lambda_boundarygrad_loss`: Weight for boundary gradient loss.
- `radius`: Search radius for each length.
- `template_label_scale`: Scale factor for template labels.
- `template_label_pan`: Translation factor for template labels.

### System Configuration

- `gpu_id`: Target GPU ID.
- `num_workers`: Number of data loader workers.
- `seed`: Random seed.
- `save_intermediate`: Controls saving of intermediate files (CSV, heatmaps, models, images, etc.).

### Output Configuration

- `format`: Output format (currently supports `json`).
- `json.indent`: Indentation level for JSON formatting.
