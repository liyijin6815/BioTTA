#!/usr/bin/env python3
"""
BioTTA: 胎儿生物测量和端点定位分析工具
整合Step1（长度预测）和Step2（TTA端点定位）流程

用法:
    python run_analysis.py --input <nii_file_or_folder> [--config config.yaml]
    python run_analysis.py --input data/sample.nii.gz
    python run_analysis.py --input data/ --config my_config.yaml
"""

import os
import sys
import argparse
import torch
import yaml
import nibabel as nb
import pandas as pd
import numpy as np
from pathlib import Path
from natsort import natsorted
import json
from typing import List, Dict, Tuple, Optional

# 添加lib路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

# 导入步骤模块
from work.fetal_localization_SFUDA.BioTTA.lib.step1_length_prediction import (
    EnhancedLengthPredictor, LocalAppearance, TestImageDataset,
    WeightedMSELoss
)

# 导入Step2模块（从lib目录）
from lib.step2_source_network import LocalAppearance as Step2LocalAppearance
from lib.step2_supp import (
    TENTWrapper, extract_brain, block_ind, register_template_label, 
    get_landmark_from_heatmap, constrain_heatmap, crop_template_label, 
    plot_slice, rotate_coordinate, TestDataset as Step2TestDataset, 
    DifferentiableLandmarkDetector, EntropyLoss, LengthLoss, BoundaryGradientLoss
)


class BioTTAPipeline:
    """BioTTA分析流程主类"""
    
    def __init__(self, config_path: str):
        """初始化配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 设置GPU
        gpu_id = self.config['system']['gpu_id']
        if gpu_id:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 设置随机种子
        torch.manual_seed(self.config['system']['seed'])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 创建输出目录
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run_step1(self, input_path: str) -> pd.DataFrame:
        """运行Step1: 长度预测"""
        print("\n" + "="*60)
        print("Step 1: 长度预测")
        print("="*60)
        
        # 准备输入
        input_path = Path(input_path)
        if input_path.is_file():
            image_folder = input_path.parent
            image_files = [input_path.name]
        else:
            image_folder = input_path
            image_files = natsorted([
                f for f in os.listdir(image_folder) 
                if f.endswith('.nii') or f.endswith('.nii.gz')
            ])
        
        if not image_files:
            raise ValueError(f"在 {input_path} 中未找到.nii.gz文件")
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        # 创建数据集
        dataset = TestImageDataset(
            str(image_folder),
            self.config['data'].get('label_csv_path', None)
        )
        
        # 过滤只包含目标文件
        if input_path.is_file():
            filtered_paths = []
            filtered_names = []
            for i, img_path in enumerate(dataset.image_paths):
                img_name = dataset.image_names[i]
                if Path(img_path).name in image_files:
                    filtered_paths.append(img_path)
                    filtered_names.append(img_name)
            dataset.image_paths = filtered_paths
            dataset.image_names = filtered_names
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config['step1']['batch_size'],
            shuffle=False,
            num_workers=self.config['system']['num_workers']
        )
        
        # 加载模型
        base_model = LocalAppearance(
            in_channels=self.config['step1']['model']['in_channels'],
            num_classes=22  # heatmap channels
        )
        model = EnhancedLengthPredictor(
            base_model,
            num_classes=self.config['step1']['model']['num_classes']
        ).to(self.device)
        
        model_path = self.config['paths']['step1_model']
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Step1模型文件未找到: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"已加载Step1模型: {model_path}")
        
        # 预测
        results = []
        biometry_list = self.config['data']['biometry_list']
        length_weights = self.config['data']['length_weights']
        ventricle_indices = self.config['data']['ventricle_indices']
        criterion = WeightedMSELoss(weights=length_weights, ventricle_indices=ventricle_indices)
        
        with torch.no_grad():
            for batch_idx, (images, labels, img_names) in enumerate(dataloader):
                images = images.to(self.device)
                labels = labels.to(self.device) if labels[0] is not None else None
                
                outputs, _ = model(images)
                
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    pred_lengths = outputs[i].cpu().numpy()
                    
                    result = {'label_name': img_name}
                    for j, biometry in enumerate(biometry_list):
                        result[biometry] = float(pred_lengths[j])
                    
                    # 如果有真实标签，计算MSE
                    if labels is not None and not torch.isnan(labels[i]).any():
                        weighted_mse = criterion(outputs[i:i+1], labels[i:i+1]).item()
                        result['weighted_mse'] = weighted_mse
                    
                    results.append(result)
                    
                print(f"已处理 {batch_idx+1}/{len(dataloader)} 批次")
        
        # 保存Step1结果
        df_results = pd.DataFrame(results)
        step1_output = self.output_dir / 'step1_lengths.csv'
        df_results.to_csv(step1_output, index=False)
        print(f"\nStep1结果已保存: {step1_output}")
        
        return df_results
    
    def run_step2(self, input_path: str, length_df: pd.DataFrame) -> pd.DataFrame:
        """运行Step2: TTA端点定位"""
        print("\n" + "="*60)
        print("Step 2: TTA端点定位")
        print("="*60)
        
        # 准备输入
        input_path = Path(input_path)
        if input_path.is_file():
            image_files = [input_path.name]
            image_folder = input_path.parent
        else:
            image_folder = input_path
            image_files = natsorted([
                f for f in os.listdir(image_folder) 
                if f.endswith('.nii') or f.endswith('.nii.gz')
            ])
        
        # 加载年龄文件
        age_df = None
        age_file = self.config['data'].get('age_file', '')
        if age_file and os.path.exists(age_file):
            age_df = pd.read_csv(age_file, dtype={'name': str})
            name_col = self.config['data']['age_file_format']['name_column']
            age_col = self.config['data']['age_file_format']['age_column']
            print(f"已加载年龄文件: {age_file}")
        
        # 配置参数
        step2_cfg = self.config['step2']
        
        # 加载Step2模型
        net = Step2LocalAppearance(1, step2_cfg['channel']).to(self.device)
        step2_model_path = self.config['paths']['step2_model']
        if os.path.exists(step2_model_path):
            pretrained_model = torch.load(step2_model_path, map_location=self.device)
            if 'net' in pretrained_model:
                net.load_state_dict(pretrained_model['net'], strict=False)
            else:
                net.load_state_dict(pretrained_model, strict=False)
            print(f"已加载Step2基础模型: {step2_model_path}")
        
        # 处理每个图像
        all_results = []
        
        for file in image_files:
            if not (file.endswith('.nii') or file.endswith('.nii.gz')):
                continue
                
            image_name = file[:-7] if file.endswith('.nii.gz') else file[:-4]
            print(f"\n处理图像: {image_name}")
            
            # 检查是否已有该图像的长度预测
            if image_name not in length_df['label_name'].values:
                print(f"警告: {image_name} 未在Step1结果中找到，跳过")
                continue
            
            # 获取该图像的预测长度
            image_lengths = length_df[length_df['label_name'] == image_name].iloc[0]
            length_values = [image_lengths[biom] for biom in self.config['data']['biometry_list']]
            
            # 创建临时DataFrame用于Step2
            temp_length_df = pd.DataFrame({
                'label_name': [image_name],
                **{biom: [length_values[i]] for i, biom in enumerate(self.config['data']['biometry_list'])}
            })
            
            # 图像路径
            image_path = os.path.join(image_folder, file)
            
            # 运行Step2
            landmarks = self._process_single_image_step2(
                image_path, image_name, temp_length_df, 
                age_df, net, step2_cfg
            )
            
            # 保存结果
            result = {
                'image_name': image_name,
                **{f'length_{biom}': length_values[i] for i, biom in enumerate(self.config['data']['biometry_list'])},
            }
            
            # 添加22个端点坐标（扁平化）
            for i in range(22):
                result[f'landmark_{i+1}_x'] = float(landmarks[i][0])
                result[f'landmark_{i+1}_y'] = float(landmarks[i][1])
                result[f'landmark_{i+1}_z'] = float(landmarks[i][2])
            
            # 添加结构化坐标
            result['landmarks'] = landmarks.tolist()
            
            all_results.append(result)
        
        return pd.DataFrame(all_results)
    
    def _process_single_image_step2(
        self, image_path: str, image_name: str, 
        length_df: pd.DataFrame, age_df: Optional[pd.DataFrame],
        net: torch.nn.Module, step2_cfg: dict
    ) -> np.ndarray:
        """处理单个图像的Step2流程"""
        
        # 创建样本输出目录
        sample_output_dir = self.output_dir / image_name
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 准备数据
        dataset = Step2TestDataset(image_path, length_df)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=step2_cfg['batch_size'], 
            shuffle=True, 
            num_workers=self.config['system']['num_workers'],
            drop_last=True,
            pin_memory=True
        )
        
        # TTA包装
        tent_net = TENTWrapper(net)
        
        # 检查是否需要训练或直接使用已有模型
        use_existing = self.config['step2'].get('use_existing_tta', False)
        tta_model_path = sample_output_dir / 'tent_TTA.pth'
        
        if use_existing and tta_model_path.exists():
            # 直接加载已有模型
            TTA_model = torch.load(tta_model_path, map_location=self.device)
            tent_net.load_state_dict(TTA_model['net'], strict=False)
            print(f"  使用已有TTA模型: {tta_model_path}")
        elif not self.config['step2'].get('skip_tta', False):
            # 进行TTA训练
            print(f"  开始TTA训练...")
            tent_net.train()
            
            detector = DifferentiableLandmarkDetector(
                init_temp=step2_cfg['init_temp'], 
                topk=step2_cfg['topk']
            )
            entropy_loss_f = EntropyLoss()
            length_loss_f = LengthLoss()
            boundarygrad_loss_f = BoundaryGradientLoss(
                point_pairs=step2_cfg['point_pairs'],
                kernel_type='scharr',
                smooth_sigma=1.0,
                grad_sigma=0.9,
                adaptive_threshold=False,
                min_grad=0.4
            )
            
            optim = torch.optim.Adam(
                list(tent_net.parameters()) + list(length_loss_f.parameters()),
                lr=step2_cfg['learning_rate']
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=step2_cfg['num_epoch']
            )
            
            min_loss = float('inf')
            num_epoch = step2_cfg['num_epoch']
            
            for epoch in range(num_epoch):
                for i, batch_data in enumerate(dataloader):
                    image = batch_data['image'].to(self.device)
                    length_pseudo_label = batch_data['length'].to(self.device)
                    
                    _, HLA = tent_net(image)
                    
                    all_keypoints = []
                    for c in range(step2_cfg['channel']):
                        channel_heatmap = HLA[:, c:c+1, :, :, :]
                        keypoints = detector(channel_heatmap)
                        all_keypoints.append(keypoints.squeeze(1))
                    keypoints = torch.stack(all_keypoints, dim=1)
                    
                    entropy_loss = entropy_loss_f(HLA)
                    length_loss = length_loss_f(keypoints, length_pseudo_label)
                    boundarygrad_loss = boundarygrad_loss_f(image, keypoints)
                    loss = (
                        step2_cfg['lambda_entropy_loss'] * entropy_loss +
                        step2_cfg['lambda_length_loss'] * length_loss +
                        step2_cfg['lambda_boundarygrad_loss'] * boundarygrad_loss
                    )
                    
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                
                scheduler.step()
                current_loss = loss.item()
                
                if current_loss < min_loss:
                    min_loss = current_loss
                    torch.save({
                        'net': tent_net.state_dict(),
                        'optim': optim.state_dict()
                    }, str(tta_model_path))
                
                print(f"  Epoch [{epoch+1}/{num_epoch}], Loss: {current_loss:.4f}")
        
        # 推理得到热图
        tent_net.eval()
        with torch.no_grad():
            for batch_data in dataloader:
                image = batch_data['image'].to(self.device)
                _, HLA = tent_net(image)
                heatmap = HLA.detach().cpu().numpy().astype(np.float32).squeeze()
                break
        
        # 加载图像用于图谱配准
        data = nb.load(image_path).get_fdata()
        data_copy = data.copy()
        data_expand = np.expand_dims(data_copy, -1)
        mask = data_expand > 0
        ind_brain = block_ind(mask)
        sized_data = extract_brain(data, ind_brain, [128, 160, 128])
        sized_mask = np.zeros([128, 160, 128])
        sized_mask = sized_data > 0
        
        img_affine = nb.load(image_path).affine
        
        # 图谱配准
        cropped_template_heatmap_data = None
        age = None
        if age_df is not None:
            name_col = self.config['data']['age_file_format']['name_column']
            age_col = self.config['data']['age_file_format']['age_column']
            
            if image_name in age_df[name_col].values:
                age = age_df.loc[age_df[name_col] == image_name, age_col].values[0]
                age = round(age)
                
                if age > 22:
                    template_image_path = os.path.join(
                        self.config['paths']['template']['image_folder'],
                        f'STA{age}.nii.gz'
                    )
                    template_label_path = os.path.join(
                        self.config['paths']['template']['label_folder'],
                        f'{age}_m.nii.gz'
                    )
                    
                    registered_folder = self.config['paths']['template'].get('registered_folder', '')
                    if registered_folder:
                        template_label_register_path = os.path.join(
                            registered_folder,
                            f'{image_name}_registered_{age}_m.nii.gz'
                        )
                    else:
                        template_label_register_path = sample_output_dir / f'template_registered_{age}_m.nii.gz'
                    
                    if not os.path.exists(template_label_register_path):
                        register_template_label(
                            template_image_path,
                            template_label_path,
                            image_path,
                            str(template_label_register_path)
                        )
                    
                    registered_template_label = nb.load(str(template_label_register_path)).get_fdata()
                    cropped_template_heatmap_data = crop_template_label(registered_template_label, data)
        
            # 提取22个端点坐标
            radius = step2_cfg['radius']
            orientations = step2_cfg['orientations']
            template_heatmap_number = step2_cfg['template_heatmap_number']
            template_label_scale = step2_cfg['template_label_scale']
            template_label_pan = step2_cfg['template_label_pan']
            
            coordinates = []
            
            # 保存landmarks.txt（如果配置要求）
            landmarks_txt_path = sample_output_dir / 'landmarks.txt'
            save_txt = self.config['system']['save_intermediate'].get('step2_landmarks_txt', True)
            
            if save_txt:
                landmarks_file = open(landmarks_txt_path, 'w')
            
            for i in range(11):
                if cropped_template_heatmap_data is not None:
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
                else:
                    heatmap_data_1 = heatmap[2*i]
                    heatmap_data_2 = heatmap[2*i+1]
                
                coordinate_1, _ = get_landmark_from_heatmap(
                    heatmap_data_1, orientations[i],
                    init_temp=step2_cfg['init_temp'],
                    topk=step2_cfg['topk']
                )
                coordinate_2, _ = get_landmark_from_heatmap(
                    heatmap_data_2, orientations[i],
                    init_temp=step2_cfg['init_temp'],
                    topk=step2_cfg['topk']
                )
                
                coordinates.append(coordinate_1)
                coordinates.append(coordinate_2)
                
                # 写入landmarks.txt
                if save_txt:
                    merged_coords = np.array([coordinate_1, coordinate_2]).flatten()
                    coords_str = ' '.join(map(str, merged_coords))
                    landmarks_file.write(f"{coords_str}\n")
            
            if save_txt:
                landmarks_file.close()
            
            # 生成可视化（如果配置要求）
            if self.config['output'].get('save_visualization', True):
                try:
                    self._save_visualization(
                        image_name, sized_mask, sized_data, coordinates,
                        orientations, sample_output_dir
                    )
                except Exception as e:
                    print(f"  警告: 可视化生成失败: {e}")
            
            return np.array(coordinates)
    
    def _save_visualization(
        self, image_name: str, sized_mask: np.ndarray, sized_data: np.ndarray,
        coordinates: np.ndarray, orientations: List[int], output_dir: Path
    ):
        """保存可视化图像"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib import gridspec
            
            max_value = np.max(sized_data)
            min_value = np.min(sized_data)
            
            images_ups = []
            aspect_ratios = []
            cross_positions_list = []
            
            # 处理每个通道的可视化
            for i in range(11):
                coordinate = np.array([coordinates[2*i], coordinates[2*i+1]])
                slice_sum_max_index = int((coordinate[0][orientations[i]] + coordinate[1][orientations[i]]) / 2)
                
                img_ups, cross_positions = plot_slice(
                    sized_mask, sized_data, slice_sum_max_index,
                    orientations[i], max_value, min_value, coordinate
                )
                
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
                if len(non_zero_indices[1]) > 0:
                    leftmost = np.min(non_zero_indices[1])
                    rightmost = np.max(non_zero_indices[1])
                    new_width = rightmost - leftmost + 20
                    h, w = img_ups.shape[:2]
                    new_img = np.zeros((h, new_width, 3), dtype=img_ups.dtype)
                    paste_left = 10
                    new_img[:, paste_left:paste_left + (rightmost - leftmost)] = img_ups[:, leftmost:rightmost]
                    img_ups = new_img
                    cross_positions = [(x, y-leftmost+paste_left) for (x, y) in cross_positions]
                
                images_ups.append(img_ups)
                cross_positions_list.append(cross_positions)
                h, w = img_ups.shape[:2]
                aspect_ratios.append(w / h)
            
            # 创建GridSpec布局
            total_width_ratio = sum(aspect_ratios)
            fig_height = 3
            wspace = -0.05
            fig = plt.figure(
                figsize=(fig_height * (total_width_ratio + wspace * 10), fig_height)
            )
            gs = gridspec.GridSpec(1, 11, width_ratios=aspect_ratios, wspace=wspace)
            
            # 绘制子图
            for i in range(11):
                img_ups = images_ups[i]
                cross_positions = cross_positions_list[i]
                
                if np.max(img_ups) > 1 or np.min(img_ups) < 0:
                    img_ups = (img_ups - np.min(img_ups)) / (np.max(img_ups) - np.min(img_ups))
                img_ups = img_ups.astype(np.float32)
                
                ax = fig.add_subplot(gs[i])
                ax.imshow(img_ups, aspect='auto')
                for (x, y) in cross_positions:
                    ax.plot(y, x, 'kx', markersize=7, markeredgewidth=4)
                for (x, y) in cross_positions:
                    ax.plot(y, x, 'rx', markersize=6, markeredgewidth=2)
                ax.axis('off')
                ax.set_adjustable('datalim')
            
            # 保存图像
            vis_path = output_dir / 'landmarks.png'
            plt.savefig(vis_path, bbox_inches='tight', dpi=300, pad_inches=0)
            plt.close()
            print(f"  可视化已保存: {vis_path}")
        except Exception as e:
            print(f"  警告: 可视化保存失败: {e}")
    
    def save_results(self, results_df: pd.DataFrame):
        """保存最终结果"""
        print("\n" + "="*60)
        print("保存结果")
        print("="*60)
        
        output_format = self.config['output']['format']
        filename = self.config['output']['filename']
        
        # 保存CSV
        if output_format in ['csv', 'both']:
            csv_path = self.output_dir / f"{filename}.csv"
            results_df.to_csv(csv_path, index=False)
            print(f"CSV结果已保存: {csv_path}")
        
        # 保存JSON
        if output_format in ['json', 'both']:
            json_path = self.output_dir / f"{filename}.json"
            
            # 转换为字典列表
            results_dict = results_df.to_dict('records')
            
            # 处理landmarks列，确保是列表格式
            for record in results_dict:
                if 'landmarks' in record:
                    if isinstance(record['landmarks'], str):
                        # 如果是字符串，尝试解析
                        try:
                            import ast
                            record['landmarks'] = ast.literal_eval(record['landmarks'])
                        except:
                            record['landmarks'] = []
                    # 确保landmarks是列表格式
                    if isinstance(record['landmarks'], (list, np.ndarray)):
                        record['landmarks'] = [[float(c) for c in point] for point in record['landmarks']]
                
                # 转换numpy类型为Python原生类型
                for key, value in record.items():
                    if isinstance(value, (np.integer, np.floating)):
                        record[key] = float(value)
            
            indent = self.config['output']['json'].get('indent', 2)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results_dict, f, indent=indent, ensure_ascii=False)
            print(f"JSON结果已保存: {json_path}")
        
        print(f"\n总共处理 {len(results_df)} 个样本")
        print(f"结果保存在: {self.output_dir}")
    
    def run(self, input_path: str):
        """运行完整流程"""
        print(f"\n开始BioTTA分析流程")
        print(f"输入: {input_path}")
        print(f"输出目录: {self.output_dir}")
        
        # Step1: 长度预测
        length_df = self.run_step1(input_path)
        
        # Step2: 端点定位
        results_df = self.run_step2(input_path, length_df)
        
        # 保存结果
        self.save_results(results_df)
        
        print("\n" + "="*60)
        print("分析完成！")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='BioTTA: 胎儿生物测量和端点定位分析工具'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='输入路径（单个.nii.gz文件或包含多个文件的文件夹）'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='配置文件路径（默认: config.yaml）'
    )
    
    args = parser.parse_args()
    
    # 检查输入路径
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")
    
    # 检查配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    # 运行流程
    pipeline = BioTTAPipeline(str(config_path))
    pipeline.run(str(input_path))


if __name__ == "__main__":
    main()

