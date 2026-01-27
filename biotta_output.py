"""
输出模块
将22个端点位置保存为CSV/JSON格式
"""
import os
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Optional
from pathlib import Path


def save_results(
    results: List[Dict],
    output_dir: str,
    output_format: str = "json",
    biometry_list: Optional[List[str]] = None
):
    """
    保存结果到JSON文件
    
    Args:
        results: 结果列表，每个元素包含：
            - image_name: 图像名称
            - age: 年龄（可选）
            - lengths: 11个长度值列表或字典
            - landmarks: 22个端点坐标 (22, 3) 或列表
        output_dir: 输出目录
        output_format: 输出格式（现在只支持"json"）
        biometry_list: 生物测量列表
    """
    if biometry_list is None:
        biometry_list = ['RLV_A', 'LLV_A', 'RLV_C', 'LLV_C', 'BBD', 'CBD', 'TCD', 'FOD', 'AVD', 'VH', 'CCL']
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备JSON数据
    json_data = []
    
    for result in results:
        image_name = result['image_name']
        age = result.get('age', None)
        
        # 处理长度数据
        lengths = result.get('lengths', {})
        if isinstance(lengths, dict):
            length_dict = lengths
        elif isinstance(lengths, (list, np.ndarray)):
            length_dict = {biometry_list[i]: float(lengths[i]) for i in range(len(lengths))}
        else:
            length_dict = {}
        
        # 处理端点坐标
        landmarks = result.get('landmarks', None)
        if landmarks is not None:
            if isinstance(landmarks, np.ndarray):
                landmarks = landmarks.tolist()
        else:
            landmarks = [[0.0, 0.0, 0.0]] * 22
        
        # JSON格式数据
        json_entry = {
            'image_name': image_name,
            'lengths': length_dict,
            'landmarks': landmarks
        }
        
        if age is not None:
            json_entry['age'] = float(age)
        
        json_data.append(json_entry)
    
    # 保存JSON
    json_path = os.path.join(output_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"✓ 结果已保存到: {json_path}")
    
    return json_data


def format_landmarks_for_display(landmarks: np.ndarray) -> str:
    """
    格式化端点坐标用于显示
    
    Args:
        landmarks: 端点坐标 (22, 3)
    
    Returns:
        格式化后的字符串
    """
    if landmarks is None:
        return "No landmarks"
    
    if isinstance(landmarks, list):
        landmarks = np.array(landmarks)
    
    lines = []
    lines.append("端点坐标 (22个点):")
    lines.append("-" * 50)
    
    point_names = [
        "RLV_A_start", "RLV_A_end",
        "LLV_A_start", "LLV_A_end",
        "RLV_C_start", "RLV_C_end",
        "LLV_C_start", "LLV_C_end",
        "BBD_start", "BBD_end",
        "CBD_start", "CBD_end",
        "TCD_start", "TCD_end",
        "FOD_start", "FOD_end",
        "AVD_start", "AVD_end",
        "VH_start", "VH_end",
        "CCL_start", "CCL_end"
    ]
    
    for i in range(min(22, len(landmarks))):
        point = landmarks[i]
        point_name = point_names[i] if i < len(point_names) else f"Point_{i+1}"
        lines.append(f"{point_name:20s}: ({point[0]:8.2f}, {point[1]:8.2f}, {point[2]:8.2f})")
    
    return "\n".join(lines)

