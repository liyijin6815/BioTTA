"""
Step1: 长度预测模块
预测11个生物测量长度
"""
import os
import sys
import pandas as pd
import numpy as np
import nibabel as nb
from natsort import natsorted
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Sequence, List, Dict, Optional
import warnings

# 导入lib目录下的模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
from lib.step1_length_prediction import (
    LocalAppearance, EnhancedLengthPredictor, 
    TestImageDataset, WeightedMSELoss,
    extract_brain, block_ind
)
# 默认生物测量列表
default_biometry_list = ['RLV_A', 'LLV_A', 'RLV_C', 'LLV_C', 'BBD', 'CBD', 'TCD', 'FOD', 'AVD', 'VH', 'CCL']


def predict_lengths(
    image_paths: List[str],
    model_path: str,
    device: torch.device,
    batch_size: int = 4,
    biometry_list: List[str] = None,
    length_weights: List[float] = None,
    ventricle_indices: List[int] = None,
    label_csv_path: Optional[str] = None
) -> pd.DataFrame:
    """
    预测多个图像的长度
    
    Args:
        image_paths: 图像文件路径列表
        model_path: 模型权重路径
        device: 计算设备
        batch_size: 批大小
        biometry_list: 生物测量列表
        length_weights: 长度权重列表
        ventricle_indices: 侧脑室索引列表
        label_csv_path: 标签CSV路径（可选，用于评估）
    
    Returns:
        包含预测结果的DataFrame
    """
    if biometry_list is None:
        biometry_list = default_biometry_list
    if length_weights is None:
        length_weights = [10.0, 10.0, 10.0, 10.0, 1.5, 1.5, 0.6, 1.0, 3.0, 3.0, 1.0]
    if ventricle_indices is None:
        ventricle_indices = [0, 1, 2, 3]
    
    # 创建临时数据集
    # 为了兼容TestImageDataset，我们需要创建临时目录或修改数据集类
    # 这里我们创建一个简化的数据集类
    class SimpleImageDataset(Dataset):
        def __init__(self, image_paths):
            self.image_paths = image_paths
            self.image_names = []
            for img_path in image_paths:
                img_name = os.path.basename(img_path)
                if img_name.endswith('.nii.gz'):
                    img_name = img_name[:-7]
                elif img_name.endswith('.nii'):
                    img_name = img_name[:-4]
                self.image_names.append(img_name)
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            img_name = self.image_names[idx]
            
            # 加载并预处理图像
            X_T1 = nb.load(img_path).get_fdata()
            mask = X_T1 > 0
            
            # 标准化
            X_T1_brain = X_T1[mask]
            mean = np.mean(X_T1_brain) if len(X_T1_brain) > 0 else 0
            std = np.std(X_T1_brain) if len(X_T1_brain) > 0 else 1
            normalized_brain = (X_T1_brain - mean) / std if len(X_T1_brain) > 0 else X_T1_brain
            X_T1[mask] = normalized_brain
            X_T1[~mask] = 0
            
            # 缩放
            ind_brain = block_ind(mask)
            X_T1 = extract_brain(X_T1, ind_brain, [128, 160, 128])
            data = X_T1.reshape((1,) + X_T1.shape)
            data = torch.tensor(data, dtype=torch.float32)
            
            # 标签占位符
            label = torch.full((len(biometry_list),), float('nan'), dtype=torch.float32)
            
            return data, label, img_name
    
    # 创建数据集和数据加载器
    dataset = SimpleImageDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 初始化模型
    base_model = LocalAppearance(in_channels=1, num_classes=22)
    model = EnhancedLengthPredictor(base_model, num_classes=11).to(device)
    
    # 加载模型权重
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint)
        print(f"✓ 模型权重已加载: {model_path}")
    else:
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    
    model.eval()
    
    # 预测
    results = []
    with torch.no_grad():
        for batch_idx, (images, labels, img_names) in enumerate(dataloader):
            images = images.to(device)
            outputs, _ = model(images)
            
            for i in range(len(img_names)):
                sample_result = {
                    'label_name': img_names[i],
                    **{biometry_list[j]: outputs[i][j].item() for j in range(11)}
                }
                results.append(sample_result)
            
            print(f"已处理 {batch_idx + 1}/{len(dataloader)} 批次")
    
    # 创建DataFrame
    df_results = pd.DataFrame(results)
    columns = ['label_name'] + [biometry_list[i] for i in range(11)]
    df_results = df_results[columns]
    
    return df_results


def predict_single_image(
    image_path: str,
    model_path: str,
    device: torch.device,
    biometry_list: List[str] = None,
    length_weights: List[float] = None,
    ventricle_indices: List[int] = None
) -> Dict[str, float]:
    """
    使用示例：预测单个图像的长度
    
    Args:
        image_path: 图像文件路径
        model_path: 模型权重路径
        device: 计算设备
        biometry_list: 生物测量列表
        length_weights: 长度权重列表
        ventricle_indices: 侧脑室索引列表
    
    Returns:
        包含预测结果的字典
    """
    results_df = predict_lengths(
        [image_path],
        model_path,
        device,
        batch_size=1,
        biometry_list=biometry_list,
        length_weights=length_weights,
        ventricle_indices=ventricle_indices
    )
    
    # 转换为字典格式
    result_dict = results_df.iloc[0].to_dict()
    return result_dict

