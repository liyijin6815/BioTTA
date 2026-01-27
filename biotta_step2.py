"""
Step2: TTA端点定位模块
使用测试时自适应（TTA）预测22个端点的精确位置
"""
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from natsort import natsorted
import nibabel as nb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Optional, Tuple
from matplotlib import gridspec
import matplotlib.pyplot as plt
import scipy.stats as stats

# 导入lib目录下的模块
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
from lib.step2_source_network import LocalAppearance
from lib.step2_supp import (
    TENTWrapper, TestDataset, DifferentiableLandmarkDetector,
    EntropyLoss, LengthLoss, BoundaryGradientLoss,
    register_template_label, constrain_heatmap, get_landmark_from_heatmap,
    extract_brain, block_ind, crop_template_label,
    plot_slice, rotate_coordinate, convert_coords_to_original
)


def run_tta_and_predict(
    image_path: str,
    pred_lengths: pd.Series,
    model_path: str,
    device: torch.device,
    config: Dict,
    template_paths: Optional[Dict] = None,
    age: Optional[float] = None,
    skip_tta: bool = False,
    tta_model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    curve_dir: Optional[str] = None,
    save_intermediate: Optional[Dict] = None
) -> Tuple[np.ndarray, Dict]:
    """
    对单个图像运行TTA并预测端点位置
    
    Args:
        image_path: 目标图像路径
        pred_lengths: 预测的长度值（11个长度）
        model_path: 预训练模型路径
        device: 计算设备
        config: Step2配置字典
        template_paths: 模板路径字典
        age: 年龄（用于匹配模板）
        skip_tta: 是否跳过TTA训练
        tta_model_path: 已有TTA模型路径（如果跳过训练）
    
    Returns:
        landmarks: 22个端点坐标 (22, 3)
        metadata: 元数据字典
    """
    # 解析配置
    batch_size = config.get('batch_size', 1)
    channel = config.get('channel', 22)
    init_temp = config.get('init_temp', 0.5)
    topk = config.get('topk', 50)
    learning_rate = config.get('learning_rate', 0.0005)
    num_epoch = config.get('num_epoch', 5)
    lambda_entropy_loss = config.get('lambda_entropy_loss', 0.5)
    lambda_length_loss = config.get('lambda_length_loss', 1.5)
    lambda_boundarygrad_loss = config.get('lambda_boundarygrad_loss', 300)
    point_pairs = config.get('point_pairs', list(range(1, 12)))
    radius = config.get('radius', [20, 20, 20, 20, 20, 20, 5, 5, 5, 5, 5])
    template_label_scale = config.get('template_label_scale', 1)
    template_label_pan = config.get('template_label_pan', 0.25)
    orientations = config.get('orientations', [2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    template_heatmap_number = config.get('template_heatmap_number', 
        [1, 2, 3, 4, 9, 10, 11, 12, 13, 14, 13, 14, 17, 18, 23, 24, 25, 26, 27, 28, 29, 30])
    
    # 创建临时DataFrame用于TestDataset
    image_name = os.path.basename(image_path)
    if image_name.endswith('.nii.gz'):
        image_name = image_name[:-7]
    elif image_name.endswith('.nii'):
        image_name = image_name[:-4]
    
    # 构建pred_length_df
    biometry_list = ['RLV_A', 'LLV_A', 'RLV_C', 'LLV_C', 'BBD', 'CBD', 'TCD', 'FOD', 'AVD', 'VH', 'CCL']
    pred_length_dict = {'label_name': image_name}
    for i, biometry in enumerate(biometry_list):
        pred_length_dict[biometry] = pred_lengths.iloc[i] if isinstance(pred_lengths, pd.Series) else pred_lengths[i]
    pred_length_df = pd.DataFrame([pred_length_dict])
    
    # 创建数据集
    dataset_target = TestDataset(image_path, pred_length_df)
    target_loader = DataLoader(
        dataset_target, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # 避免多进程问题
        drop_last=True,
        pin_memory=True
    )
    
    # 初始化网络
    net = LocalAppearance(1, channel).to(device)
    
    if not skip_tta and tta_model_path is None:
        # 加载预训练模型
        if os.path.exists(model_path):
            pretrained_model = torch.load(model_path, map_location=device, weights_only=True)
            net.load_state_dict(pretrained_model.get('net', pretrained_model), strict=False)
            print(f"✓ 预训练模型已加载: {model_path}")
        else:
            raise FileNotFoundError(f"预训练模型文件未找到: {model_path}")
        
        # 封装为TENT网络
        tent_net = TENTWrapper(net)
        
        # TTA训练设置
        detector = DifferentiableLandmarkDetector(init_temp=init_temp, topk=topk)
        entropy_loss_f = EntropyLoss()
        length_loss_f = LengthLoss()
        boundarygrad_loss_f = BoundaryGradientLoss(
            point_pairs=point_pairs,
            kernel_type='scharr',
            smooth_sigma=1.0,
            grad_sigma=0.9,
            adaptive_threshold=False,
            min_grad=0.4
        )
        
        torch.manual_seed(1)
        optim = torch.optim.Adam(
            list(tent_net.parameters()) + list(length_loss_f.parameters()),
            lr=learning_rate
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=num_epoch)
        
        # TTA训练
        tent_net.train()
        min_loss = float('inf')
        best_model_state = None
        
        print(f"开始TTA训练 (epochs={num_epoch})...")
        for epoch in range(num_epoch):
            epoch_losses = []
            for i, batch_data in enumerate(target_loader):
                image = batch_data['image'].to(device)
                length_pseudo_label = batch_data['length'].to(device)
                
                # 前向传播
                _, HLA = tent_net(image)
                
                # 计算坐标位置
                all_keypoints = []
                for c in range(channel):
                    channel_heatmap = HLA[:, c:c+1, :, :, :]
                    keypoints = detector(channel_heatmap)
                    all_keypoints.append(keypoints.squeeze(1))
                keypoints = torch.stack(all_keypoints, dim=1)
                
                # 计算损失
                entropy_loss = entropy_loss_f(HLA)
                length_loss = length_loss_f(keypoints, length_pseudo_label)
                boundarygrad_loss = boundarygrad_loss_f(image, keypoints)
                loss = (lambda_entropy_loss * entropy_loss + 
                       lambda_length_loss * length_loss + 
                       lambda_boundarygrad_loss * boundarygrad_loss)
                
                # 反向传播
                optim.zero_grad()
                loss.backward()
                optim.step()
                
                epoch_losses.append(loss.item())
                if i == 0:  # 只打印第一个batch
                    print(f'Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}')
            
            avg_loss = np.mean(epoch_losses)
            if avg_loss < min_loss:
                min_loss = avg_loss
                best_model_state = tent_net.state_dict().copy()
        
        # 加载最佳模型
        if best_model_state is not None:
            tent_net.load_state_dict(best_model_state)
        print("✓ TTA训练完成")
        
        # 保存TTA模型（如果配置允许）
        if save_intermediate is not None and save_intermediate.get('step2_tta_models', False):
            if output_dir is not None:
                tta_model_path = os.path.join(output_dir, 'tent_TTA.pth')
                torch.save({
                    'net': tent_net.state_dict(),
                    'optim': optim.state_dict()
                }, tta_model_path)
                print(f"✓ TTA模型已保存到: {tta_model_path}")
        
    elif tta_model_path is not None and os.path.exists(tta_model_path):
        # 直接加载已有TTA模型
        TTA_model = torch.load(tta_model_path, map_location=device)
        tent_net = TENTWrapper(net)
        tent_net.load_state_dict(TTA_model.get('net', TTA_model), strict=False)
        print(f"✓ 已有TTA模型已加载: {tta_model_path}")
    else:
        # 跳过TTA，直接使用预训练模型
        pretrained_model = torch.load(model_path, map_location=device)
        net.load_state_dict(pretrained_model.get('net', pretrained_model), strict=False)
        tent_net = TENTWrapper(net)
        print("⚠ 跳过TTA训练，使用预训练模型")
    
    # 预测
    for batch_data in target_loader:
        image = batch_data['image'].to(device)
        _, HLA = tent_net(image)
        heatmap = HLA.detach().cpu().numpy().astype(np.float32).squeeze()
        break  # 只有一个batch
    
    # 加载原始图像用于处理
    data = nb.load(image_path).get_fdata()
    data_copy = data.copy()
    data_expand = np.expand_dims(data_copy, -1)
    mask = data_expand > 0
    ind_brain = block_ind(mask)
    sized_data = extract_brain(data, ind_brain, [128, 160, 128])
    sized_mask = sized_data > 0
    img_affine = nb.load(image_path).affine
    max_value = np.max(sized_data)
    min_value = np.min(sized_data)
    
    # 模板约束（如果有）
    cropped_template_heatmap_data = None
    if template_paths and age is not None:
        age = round(age)
        if age > 22:
            template_image_path = os.path.join(template_paths['image_folder'], f'STA{age}.nii.gz')
            template_label_path = os.path.join(template_paths['label_folder'], f'{age}_m.nii.gz')
            registered_folder = template_paths.get('registered_folder', None)
            print("registered_folder: ", registered_folder)
            
            if registered_folder:
                os.makedirs(registered_folder, exist_ok=True)
                registered_template_path = os.path.join(registered_folder, f'{image_name}_registered_{age}_m.nii.gz')
                if not os.path.exists(registered_template_path):
                    # 执行配准
                    register_template_label(template_image_path, template_label_path, image_path, registered_template_path)
                    print(f"✓ 模板配准完成，胎龄: {age}, 反配准后的模板标签文件路径: {registered_folder}")
                else:
                    print(f"✓ 使用已有配准结果，胎龄: {age}, 反配准后的模板标签文件路径: {registered_folder}")
                
                registered_template_label = nb.load(registered_template_path).get_fdata()
                cropped_template_heatmap_data = crop_template_label(registered_template_label, data)
            # else:
            #     # 临时配准
            #     import tempfile
            #     with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp_file:
            #         tmp_path = tmp_file.name
            #     register_template_label(template_image_path, template_label_path, image_path, tmp_path)
            #     registered_template_label = nb.load(tmp_path).get_fdata()
            #     cropped_template_heatmap_data = crop_template_label(registered_template_label, data)
            #     os.remove(tmp_path)
    
    # 提取端点坐标
    coordinates = []
    if cropped_template_heatmap_data is not None:
        # 使用模板约束
        for i in range(11):
            heatmap_data_1 = constrain_heatmap(
                cropped_template_heatmap_data, heatmap[2*i], 
                template_heatmap_number[2*i], radius[i], 
                template_label_scale, template_label_pan
            )
            heatmap_data_2 = constrain_heatmap(
                cropped_template_heatmap_data, heatmap[2*i+1], 
                template_heatmap_number[2*i+1], radius[i], 
                template_label_scale, template_label_pan
            )
            
            coordinate_1, _ = get_landmark_from_heatmap(heatmap_data_1, orientations[i], init_temp=init_temp, topk=topk)
            coordinate_2, _ = get_landmark_from_heatmap(heatmap_data_2, orientations[i], init_temp=init_temp, topk=topk)
            coordinate = np.array([coordinate_1, coordinate_2])
            coordinates.append(coordinate)
    else:
        # 不使用模板约束
        for i in range(11):
            coordinate_1, _ = get_landmark_from_heatmap(heatmap[2*i], orientations[i], init_temp=init_temp, topk=topk)
            coordinate_2, _ = get_landmark_from_heatmap(heatmap[2*i+1], orientations[i], init_temp=init_temp, topk=topk)
            coordinate = np.array([coordinate_1, coordinate_2])
            coordinates.append(coordinate)
    
    # 转换为numpy数组 (22, 3)
    landmarks = np.vstack(coordinates)
    
    # 转换到原始图像坐标系
    landmarks_original = convert_coords_to_original(landmarks, ind_brain, [128, 160, 128])
    
    # 保存landmarks.txt（如果配置允许）- 在提取后的坐标系中
    if save_intermediate is not None and save_intermediate.get('step2_landmarks_txt', False):
        if output_dir is not None:
            landmarks_txt_path = os.path.join(output_dir, 'landmarks.txt')
            with open(landmarks_txt_path, 'w') as f:
                # 将22个端点按11个点对写入（每行6个坐标：x1,y1,z1,x2,y2,z2）
                for i in range(11):
                    coordinate = np.array([landmarks[2*i], landmarks[2*i+1]])
                    # 展平为一行6个坐标
                    merged_coordinates = coordinate.flatten()
                    coordinates_str = ' '.join(map(str, merged_coordinates))
                    f.write(f"{coordinates_str}\n")
            print(f"✓ landmarks.txt已保存到: {landmarks_txt_path}")
    
    # 保存原始图像坐标系的landmarks.txt（如果配置允许）
    if save_intermediate is not None and save_intermediate.get('step2_landmarks_txt_original', False):
        if output_dir is not None:
            landmarks_original_txt_path = os.path.join(output_dir, 'landmarks_original.txt')
            with open(landmarks_original_txt_path, 'w') as f:
                # 将22个端点按11个点对写入（每行6个坐标：x1,y1,z1,x2,y2,z2）
                for i in range(11):
                    coordinate = np.array([landmarks_original[2*i], landmarks_original[2*i+1]])
                    # 展平为一行6个坐标
                    merged_coordinates = coordinate.flatten()
                    coordinates_str = ' '.join(map(str, merged_coordinates))
                    f.write(f"{coordinates_str}\n")
            print(f"✓ landmarks_original.txt已保存到: {landmarks_original_txt_path}")
    
    # 保存可视化图像（如果配置允许）
    if save_intermediate is not None and save_intermediate.get('step2_landmarks_image', False):
        if output_dir is not None:
            save_landmarks_visualization(
                image_path, landmarks, sized_mask, sized_data, 
                max_value, min_value, config, output_dir
            )
    
    # 计算长度和分位数，并保存结果
    if age is not None and curve_dir is not None and output_dir is not None:
        # 计算11个长度
        calculated_lengths = calculate_lengths_from_landmarks(landmarks)
        
        # 计算分位数
        save_csv = save_intermediate is not None and save_intermediate.get('step2_length_centile_csv', False)
        centile_df = calculate_centiles(calculated_lengths, age, curve_dir, output_dir, save_csv=save_csv)
        
        # 绘制发育曲线图（如果配置允许）
        if save_intermediate is not None and save_intermediate.get('step2_length_centile_plot', False):
            plot_centile_curves(calculated_lengths, age, curve_dir, output_dir, save_intermediate)
    
    metadata = {
        'image_name': image_name,
        'age': age,
        'heatmap_shape': heatmap.shape,
        'landmarks_shape': landmarks.shape
    }
    
    return landmarks, metadata


def calculate_lengths_from_landmarks(landmarks: np.ndarray, resolution: float = 0.8) -> np.ndarray:
    """
    从22个端点坐标计算11个长度
    
    Args:
        landmarks: 22个端点坐标 (22, 3)
        resolution: 图像分辨率，默认0.8mm
    
    Returns:
        lengths: 11个长度值
    """
    lengths = []
    for i in range(11):
        # 获取一对端点
        point1 = landmarks[2*i]
        point2 = landmarks[2*i+1]
        
        # 计算欧氏距离
        distance = np.sqrt(np.sum((point1 - point2)**2))
        
        # 乘以分辨率得到实际长度
        actual_length = distance * resolution
        lengths.append(actual_length)
    
    return np.array(lengths)


def calculate_centiles(lengths: np.ndarray, age: float, curve_dir: str, output_dir: str, save_csv: bool = True) -> pd.DataFrame:
    """
    计算11个长度对应的分位数
    
    Args:
        lengths: 11个长度值
        age: 胎龄
        curve_dir: 曲线文件目录
        output_dir: 输出目录
        save_csv: 是否保存CSV文件，默认True
    
    Returns:
        centile_df: 包含长度和分位数的DataFrame
    """
    biometry_list = ['RLV-A', 'LLV-A', 'RLV-C', 'LLV-C', 'BBD', 'CBD', 'TCD', 'FOD', 'AVD', 'VH', 'CCL']
    centile_data = []
    
    for i, (length_name, length_value) in enumerate(zip(biometry_list, lengths)):
        # 读取对应的CSV文件
        csv_path = os.path.join(curve_dir, f"{length_name}.csv")
        
        if not os.path.exists(csv_path):
            print(f"⚠ 曲线文件不存在: {csv_path}")
            centile_value = np.nan
        else:
            # 读取曲线数据
            curve_df = pd.read_csv(csv_path)
            
            # 找到最接近的胎龄行
            closest_idx = (curve_df['GA'] - age).abs().idxmin()
            closest_row = curve_df.iloc[closest_idx]
            
            # 获取该胎龄下的50th百分位数
            p50_value = closest_row['50th']
            
            # 计算分位数（假设正态分布）
            # 使用5th和95th来估计标准差
            p5_value = closest_row['5th']
            p95_value = closest_row['95th']
            
            # 估计标准差 (95th - 5th ≈ 3.29 * std)
            estimated_std = (p95_value - p5_value) / 3.29
            
            if estimated_std > 0:
                # 计算Z-score
                z_score = (length_value - p50_value) / estimated_std
                # 计算分位数
                centile_value = stats.norm.cdf(z_score) * 100
            else:
                centile_value = 50.0  # 如果标准差为0，默认中位数
        
        centile_data.append({
            'Biometry': length_name,
            'Length_mm': length_value,
            'Centile': centile_value
        })
    
    # 创建DataFrame
    centile_df = pd.DataFrame(centile_data)
    
    # 保存到CSV文件（如果允许）
    if save_csv:
        output_path = os.path.join(output_dir, 'length_centile.csv')
        centile_df.to_csv(output_path, index=False)
        print(f"✓ 长度和分位数已保存到: {output_path}")
    
    return centile_df


def plot_centile_curves(lengths: np.ndarray, age: float, curve_dir: str, output_dir: str, save_intermediate: Optional[Dict] = None):
    """
    绘制11个长度的发育曲线图
    
    Args:
        lengths: 11个长度值
        age: 胎龄
        curve_dir: 曲线文件目录
        output_dir: 输出目录
    """
    biometry_list = ['RLV-A', 'LLV-A', 'RLV-C', 'LLV-C', 'BBD', 'CBD', 'TCD', 'FOD', 'AVD', 'VH', 'CCL']
    
    # Y轴设置
    y_axis_settings = {
        "RLV-A": (2, 18, 4),
        "LLV-A": (2, 18, 4),
        "RLV-C": (2, 18, 4),
        "LLV-C": (2, 18, 4),
        "TCD": (18, 58, 10),
        "FOD": (40, 120, 20),
        "BBD": (40, 100, 15),
        "CBD": (35, 95, 15),
        "AVD": (4, 20, 4),
        "VH": (8, 24, 4),
        "CCL": (18, 50, 8),
    }
    
    # 定义绘制函数
    def draw_plot(is_dark_mode=False):
        # 创建2x6的画布
        fig, axes = plt.subplots(2, 6, figsize=(36, 12))
        axes_flat = axes.flatten()
        
        # 设置背景颜色
        if is_dark_mode:
            fig.patch.set_facecolor('#1a1a1a')
            bg_color = '#1a1a1a'
            text_color = 'white'
            grid_color = 'gray'
            spine_color = 'white'
        else:
            fig.patch.set_facecolor('white')
            bg_color = 'white'
            text_color = 'black'
            grid_color = 'gray'
            spine_color = 'black'
        
        # 罗马数字标记
        roman_numerals = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x', 'xi']
        
        for i, length_name in enumerate(biometry_list):
            ax = axes_flat[i]
            
            # 设置子图背景颜色
            ax.set_facecolor(bg_color)
            
            # 添加罗马数字标记
            y_pos = -0.1 if i < 6 else -0.18
            ax.text(0.5, y_pos, f'({roman_numerals[i]})', transform=ax.transAxes, 
                    fontsize=20, fontweight='bold', ha='center', va='top', color=text_color)
            
            # 设置网格
            ax.grid(True, linestyle="-", linewidth=1.5, color=grid_color, alpha=0.5)
            ax.tick_params(axis="both", direction="in", width=1.5, length=6, color=text_color, labelcolor=text_color)
            ax.tick_params(axis="x", labelsize=18, pad=10)
            ax.tick_params(axis="y", labelsize=18)
            
            # 读取曲线数据
            csv_path = os.path.join(curve_dir, f"{length_name}.csv")
            if os.path.exists(csv_path):
                curve_df = pd.read_csv(csv_path)
                
                # 绘制曲线
                ax.plot(
                    curve_df["GA"],
                    curve_df["50th"],
                    color="red",
                    linewidth=4.5,
                    alpha=1.0,
                    zorder=5,
                )
                ax.plot(
                    curve_df["GA"],
                    curve_df["5th"],
                    color="darkblue" if not is_dark_mode else "lightblue",
                    linewidth=4.5,
                    linestyle="--",
                    alpha=1.0,
                    zorder=5,
                )
                ax.plot(
                    curve_df["GA"],
                    curve_df["95th"],
                    color="darkblue" if not is_dark_mode else "lightblue",
                    linewidth=4.5,
                    linestyle="--",
                    alpha=1.0,
                    zorder=5,
                )
                
                # 计算x轴范围
                x_min = np.nanmin(curve_df["GA"].to_numpy())
                x_max = np.nanmax(curve_df["GA"].to_numpy())
            else:
                # 如果没有曲线数据，使用默认范围
                x_min, x_max = 20.0, 40.0
            
            # 绘制当前case的散点
            ax.scatter(
                [age],
                [lengths[i]],
                color="black" if not is_dark_mode else "white",
                alpha=1,
                s=150,
                zorder=10,
                linewidth=2
            )
            
            # 设置坐标轴范围
            y_start, y_end, y_gap = y_axis_settings.get(length_name, (0, 50, 10))
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_start, y_end)
            ax.set_xticks(np.arange(np.floor(x_min / 5) * 5, np.ceil(x_max / 5) * 5 + 0.1, 5))
            ax.set_yticks(np.arange(y_start, y_end + 0.1, y_gap))
            
            # 设置边框
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_linewidth(2)
            ax.spines["bottom"].set_linewidth(2)
            ax.spines["left"].set_color(spine_color)
            ax.spines["bottom"].set_color(spine_color)
            
            # 添加标题
            ax.text(
                0.02,
                0.95,
                length_name,
                transform=ax.transAxes,
                fontsize=22,
                fontweight="bold",
                va="top",
                color=text_color
            )
            
            # 设置轴标签
            if length_name in ["RLV-A", "TCD"]:
                ax.set_ylabel("Length (mm)", fontsize=22, color=text_color)
            if length_name in ["TCD", "FOD", "AVD", "VH", "CCL"]:
                ax.set_xlabel("Gestational Age (weeks)", fontsize=22, color=text_color)
        
        # 隐藏第12个多余子图
        if len(axes_flat) > len(biometry_list):
            axes_flat[len(biometry_list)].set_visible(False)
        
        plt.tight_layout()
        
        return fig
    
    # 生成并保存两个版本的图像
    # 1. 白色背景版本（用于PDF）
    fig = draw_plot(is_dark_mode=False)
    output_path_pdf = os.path.join(output_dir, 'length_centile.png')
    fig.savefig(output_path_pdf, bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✓ 发育曲线图(PDF版)已保存到: {output_path_pdf}")
    
    # 2. 深色背景版本（用于网页）
    fig = draw_plot(is_dark_mode=True)
    output_path_web = os.path.join(output_dir, 'length_centile_web.png')
    fig.savefig(output_path_web, bbox_inches="tight", dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"✓ 发育曲线图(网页版)已保存到: {output_path_web}")


def save_landmarks_visualization(
    image_path: str,
    landmarks: np.ndarray,
    sized_mask: np.ndarray,
    sized_data: np.ndarray,
    max_value: float,
    min_value: float,
    config: Dict,
    output_dir: str
):
    """
    保存landmarks可视化图像，参考step2_biometry_TTA.py的画图方式
    
    Args:
        image_path: 原始图像路径
        landmarks: 22个端点坐标 (22, 3)
        sized_mask: 缩放后的mask (128, 160, 128)
        sized_data: 缩放后的数据 (128, 160, 128)
        max_value: 数据最大值
        min_value: 数据最小值
        config: Step2配置字典
        output_dir: 输出目录
    """
    # 解析配置
    init_temp = config.get('init_temp', 0.5)
    topk = config.get('topk', 50)
    orientations = config.get('orientations', [2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    radius = config.get('radius', [20, 20, 20, 20, 20, 20, 5, 5, 5, 5, 5])
    template_label_scale = config.get('template_label_scale', 1)
    template_label_pan = config.get('template_label_pan', 0.25)
    location_image_names = ['1+2', '3+4', '5+6', '7+8', '9+10', '11+12', '13+14', '15+16', '17+18', '19+20', '21+22']
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 准备数据和缓存
    coordinates = []
    images_ups = []
    aspect_ratios = []
    cross_positions_list = []
    
    # 第一次遍历：计算并缓存所有需要的数据
    for i in range(11):
        # 每个通道有两个点
        coordinate_1 = landmarks[2*i]
        coordinate_2 = landmarks[2*i+1]
        coordinate = np.array([coordinate_1, coordinate_2])
        coordinates.append(coordinate)
        
        # 计算平均切片索引
        orientation = orientations[i]
        if orientation == 0:
            slice_sum_max_index_1 = int(round(coordinate_1[0]))
            slice_sum_max_index_2 = int(round(coordinate_2[0]))
        elif orientation == 1:
            slice_sum_max_index_1 = int(round(coordinate_1[1]))
            slice_sum_max_index_2 = int(round(coordinate_2[1]))
        else:  # orientation == 2
            slice_sum_max_index_1 = int(round(coordinate_1[2]))
            slice_sum_max_index_2 = int(round(coordinate_2[2]))
        
        slice_sum_max_index = round((slice_sum_max_index_1 + slice_sum_max_index_2) / 2)
        
        # 缓存切片图像
        img_ups, cross_positions = plot_slice(
            sized_mask, sized_data, slice_sum_max_index, 
            orientations[i], max_value, min_value, coordinate
        )
        
        print(location_image_names[i], coordinate, slice_sum_max_index)
        
        # 旋转图片
        img_ups = np.rot90(img_ups, k=1)
        # 旋转坐标
        rotated_cross_positions = []
        for (x, y) in cross_positions:
            x_new, y_new = rotate_coordinate(
                x, y, k=3, 
                img_width=img_ups.shape[1], 
                img_height=img_ups.shape[0]
            )
            rotated_cross_positions.append((x_new, y_new))
        cross_positions = rotated_cross_positions
        
        # 左右削边
        non_zero_indices = np.nonzero(img_ups)
        if len(non_zero_indices[0]) > 0 and len(non_zero_indices[1]) > 0:
            leftmost = np.min(non_zero_indices[1])  # 最小列索引
            rightmost = np.max(non_zero_indices[1])  # 最大列索引
            new_width = rightmost - leftmost + 20  # 两侧各留出 10 像素
            h, w = img_ups.shape[:2]  # 获取原图像的高度和宽度
            # 创建新的空白图像（黑色背景）
            new_img = np.zeros((h, new_width, 3), dtype=img_ups.dtype)
            # 将原图像粘贴到新图像的中间部分
            paste_left = 10  # 左侧留出 10 像素
            new_img[:, paste_left:paste_left + (rightmost - leftmost)] = img_ups[:, leftmost:rightmost]
            # 更新图像数据
            img_ups = new_img
            # 更新坐标位置
            cross_positions = [(x, y-leftmost+paste_left) for (x, y) in cross_positions]
        
        # 缓存
        images_ups.append(img_ups)
        cross_positions_list.append(cross_positions)
        # 计算旋转后的宽高比（宽度/高度）
        h, w = img_ups.shape[:2]
        aspect_ratios.append(w / h)
    
    # 设置全局参数
    total_width_ratio = sum(aspect_ratios)  # 总宽度（相对单位）
    fig_height = 3  # 固定高度（英寸）
    wspace = -0.05  # 可调整的间距参数（相对单位）
    # 创建GridSpec布局
    fig = plt.figure(figsize=(fig_height * (total_width_ratio + wspace * 10), fig_height))
    gs = gridspec.GridSpec(1, 11, width_ratios=aspect_ratios, wspace=wspace)
    
    # 第二次遍历：使用缓存的数据绘制子图
    for i in range(11):
        # 获取缓存的图像
        img_ups = images_ups[i]
        cross_positions = cross_positions_list[i]
        # 检查并调整图像数据
        if np.max(img_ups) > 1 or np.min(img_ups) < 0:  # 如果值范围不在 [0, 1]
            img_ups = (img_ups - np.min(img_ups)) / (np.max(img_ups) - np.min(img_ups))  # 归一化到 [0, 1]
        img_ups = img_ups.astype(np.float32)  # 确保数据类型正确
        # 创建子图
        ax = fig.add_subplot(gs[i])
        ax.imshow(img_ups, aspect='auto')
        # 在图像上绘制黑叉
        for (x, y) in cross_positions:
            ax.plot(y, x, 'kx', markersize=7, markeredgewidth=4)  # 绘制黑色加号
        # 在图像上绘制红叉
        for (x, y) in cross_positions:
            ax.plot(y, x, 'rx', markersize=6, markeredgewidth=2)  # 绘制红色加号
        ax.axis('off')
        ax.set_adjustable('datalim')
    
    # 保存图像
    landmarks_image_path = os.path.join(output_dir, 'landmarks.png')
    plt.savefig(landmarks_image_path, bbox_inches='tight', dpi=300, pad_inches=0)
    plt.close()
    print(f"✓ 可视化图像已保存到: {landmarks_image_path}")