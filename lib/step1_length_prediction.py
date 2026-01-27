import os
import pandas as pd
import numpy as np
import nibabel as nb
from natsort import natsorted
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from collections import OrderedDict
from typing import Sequence


# 测试参数设置
# test_image_folder = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/FeTA_dataset/registered/test_2'  
# label_csv_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/FeTA_dataset/registered/lyj_measure_2/length_label.csv'  
# test_model_path = '/data/birth/lmx/work/Class_projects/lyj/work/fetal_localization_SFUDA/try_and_validation/length/checkpoints/v1_copy_modify_/best_model_epoch_val.pth' #####
# output_csv_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/FeTA_dataset/registered/lyj_measure_2/test_results_v1.csv' #####
test_image_folder = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/WC_BTFE_dataset/registered/test'  
label_csv_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/WC_BTFE_dataset/registered/test_hxt_measure/length_label.csv'  
test_model_path = '/data/birth/lmx/work/Class_projects/lyj/work/fetal_localization_SFUDA/try_and_validation/length/checkpoints/v1_copy_modify_/best_model_epoch_val.pth' #####
output_csv_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/WC_BTFE_dataset/registered/test_hxt_measure/test_results_v1.csv' #####
# test_image_folder = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/LFC_dataset/registered/test'  
# label_csv_path = ''
# test_model_path = '/data/birth/lmx/work/Class_projects/lyj/work/fetal_localization_SFUDA/try_and_validation/length/checkpoints/v1_copy_modify_/best_model_epoch_val.pth' #####
# output_csv_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/LFC_dataset/registered/test_json_measure/test_results_v1.csv' #####test_image_folder = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/WC_BTFE_dataset/registered/test'  
test_image_folder = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/WCUMS_dataset/WCT_VM_test' 
label_csv_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/WCUMS_dataset/WCT_VM_test_Jia_measure/length_label.csv'  
test_model_path = '/data/birth/lmx/work/Class_projects/lyj/work/fetal_localization_SFUDA/try_and_validation/length/checkpoints/v1_copy_modify_/best_model_epoch_val.pth' #####
output_csv_path = '/data/birth/lmx/work/Class_projects/lyj/dataset/fetal_brain_localzation_dataset/WCUMS_dataset/WCT_VM_test_Jia_measure/test_results_v1.csv' #####

batch_size = 4 #####
biometry_list = ['RLV_A', 'LLV_A', 'RLV_C', 'LLV_C', 'BBD', 'CBD', 'TCD', 'FOD', 'AVD', 'VH', 'CCL'] 
length_weights =  [10.0, 10.0, 10.0, 10.0, 1.5, 1.5, 0.6, 1.0, 3.0, 3.0, 1.0]
ventricle_indices = [0, 1, 2, 3]


# 定义相关函数（与训练文件相同）
def extract_brain(data, inds, sz_brain):
    if isinstance(sz_brain, int):
        sz_brain = [sz_brain, sz_brain, sz_brain]
    xsz_brain = inds[1] - inds[0] + 1
    ysz_brain = inds[3] - inds[2] + 1
    zsz_brain = inds[5] - inds[4] + 1
    brain = np.zeros((sz_brain[0], sz_brain[1], sz_brain[2]))
    x_start = int((sz_brain[0] - xsz_brain) / 2)
    y_start = int((sz_brain[1] - ysz_brain) / 2)
    z_start = int((sz_brain[2] - zsz_brain) / 2)
    brain[x_start:x_start+xsz_brain, y_start:y_start+ysz_brain,
          z_start:z_start+zsz_brain] = data[inds[0]:inds[1]+1, inds[2]:inds[3]+1, inds[4]:inds[5]+1]
    return brain

def block_ind(mask):
    tmp = np.nonzero(mask)
    xmin, xmax = np.min(tmp[0]), np.max(tmp[0])
    ymin, ymax = np.min(tmp[1]), np.max(tmp[1])
    zmin, zmax = np.min(tmp[2]), np.max(tmp[2])
    return [xmin, xmax, ymin, ymax, zmin, zmax]


# 主任务的热图预测网络（与训练文件相同）
class LocalAppearance(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        filters: int = 64,
        dropout: float = 0.,
        mode: str = 'add',
    ):
        super().__init__()
        self.mode = mode
        self.pool = nn.AvgPool3d(2, 2, ceil_mode=True)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.in_conv = self.Block(in_channels, filters)
        self.out_conv = nn.Conv3d(filters, num_classes, 1, bias=False)
        self.enc1 = self.Block(filters, filters, dropout)
        self.enc2 = self.Block(filters, filters, dropout)
        self.enc3 = self.Block(filters, filters, dropout)
        self.enc4 = self.Block(filters, filters, dropout)
        if mode == 'add':
            self.dec3 = self.Block(filters, filters, dropout)
            self.dec2 = self.Block(filters, filters, dropout)
            self.dec1 = self.Block(filters, filters, dropout)
        else:
            self.dec3 = self.Block(2*filters, filters, dropout)
            self.dec2 = self.Block(2*filters, filters, dropout)
            self.dec1 = self.Block(2*filters, filters, dropout)
        nn.init.trunc_normal_(self.out_conv.weight, 0, 1e-4)

    def Block(self, in_channels, out_channels, dropout=0):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.Dropout3d(dropout, True),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.Dropout3d(dropout, True),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        x0 = self.in_conv(x)
        e1 = self.enc1(x0)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        if self.mode == 'add':
            d3 = self.dec3(self.up(e4)+e3)
            d2 = self.dec2(self.up(d3)+e2)
            d1 = self.dec1(self.up(d2)+e1)
        else:
            d3 = self.dec3(torch.cat([self.up(e4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        out = self.out_conv(d1)
        return d1, out

class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)"""
    def __init__(self, channels, reduction_ratio=8):
        super().__init__()
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // reduction_ratio, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // reduction_ratio, channels, 1),
            nn.Sigmoid()
        )
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv3d(channels, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        channel_att = self.channel_attention(x)
        spatial_att = self.spatial_attention(x)
        return x * channel_att * spatial_att


class FPNFusion(nn.Module):
    """特征金字塔网络(FPN)式特征融合"""
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        
        for i, in_channels in enumerate(in_channels_list):
            self.lateral_convs.append(nn.Conv3d(in_channels, out_channels, 1))
            self.fpn_convs.append(nn.Conv3d(out_channels, out_channels, 3, padding=1))
        
    def forward(self, features):
        # 自顶向下路径
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]
        
        # 上采样并相加
        for i in range(len(laterals)-1, 0, -1):
            laterals[i-1] += F.interpolate(
                laterals[i], scale_factor=2, mode='trilinear', align_corners=True
            )
        
        # 应用3x3x3卷积
        return [conv(lateral) for conv, lateral in zip(self.fpn_convs, laterals)]


class EnhancedLengthPredictor(nn.Module):
    def __init__(
        self,
        pretrained_model: nn.Module,  # 预训练模型实例
        num_classes: int,
        filters: int = 64,
        dropout: float = 0.4
    ):
        super().__init__()
        # 复用预训练模型的编码器部分
        self.pool = pretrained_model.pool  # 下采样层
        self.in_conv = pretrained_model.in_conv
        self.enc1 = pretrained_model.enc1
        self.enc2 = pretrained_model.enc2
        self.enc3 = pretrained_model.enc3
        self.enc4 = pretrained_model.enc4

        # FPN式多尺度特征融合
        self.fpn = FPNFusion(
            in_channels_list=[filters, filters, filters, filters],
            out_channels=filters // 2
        )
        
        # 侧脑室专用注意力机制 - 使用CBAM
        self.ventricle_attention = CBAM(filters * 2)  # FPN输出通道数
        
        # 侧脑室特征提取器
        self.ventricle_feature_extractor = nn.Sequential(
            nn.Conv3d(filters * 2, filters, 3, padding=1),
            nn.BatchNorm3d(filters),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(filters, filters, 3, padding=1),
            nn.BatchNorm3d(filters),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 改进的池化方式 - 使用空间金字塔池化(SPP)
        self.spp = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.AdaptiveAvgPool3d((2, 2, 2)),
            nn.AdaptiveAvgPool3d((4, 4, 4))
        )
        self.spp_proj = nn.Linear(filters * 2 * (1 + 8 + 64), filters * 2)
        
        # 简化的回归头
        self.main_fc = nn.Sequential(
            nn.Linear(filters * 2, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # 侧脑室专用回归头
        self.ventricle_fc = nn.Sequential(
            nn.Linear(filters, 64),  # 修正：ventricle_features的通道数是filters，不是filters*2
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 4)  # 只预测前4个侧脑室长度
        )
        
        # 加权融合参数
        self.fusion_weights = nn.Parameter(torch.ones(2, 4))  # 2个回归头，前4个长度
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 编码器前向传播
        x0 = self.in_conv(x)
        e1 = self.enc1(x0)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # FPN式多尺度特征融合
        fpn_features = self.fpn([e1, e2, e3, e4])
        
        # 将所有FPN特征上采样到最大分辨率并拼接
        fpn_features_upsampled = []
        for i, feat in enumerate(fpn_features):
            scale_factor = 2 ** i
            if scale_factor > 1:
                feat = F.interpolate(feat, scale_factor=scale_factor, mode='trilinear', align_corners=True)
            fpn_features_upsampled.append(feat)
        
        fused_features = torch.cat(fpn_features_upsampled, dim=1)
        
        # 侧脑室注意力
        ventricle_attn_features = self.ventricle_attention(fused_features)
        
        # 侧脑室特征增强
        ventricle_features = self.ventricle_feature_extractor(ventricle_attn_features)
        
        # 改进的池化方式 - 空间金字塔池化
        global_features = []
        for pool in self.spp:
            pooled = pool(fused_features)
            global_features.append(pooled.view(pooled.size(0), -1))
        
        global_features_flat = torch.cat(global_features, dim=1)
        global_features_flat = self.spp_proj(global_features_flat)
        
        # 侧脑室特征池化
        ventricle_pooled = F.adaptive_avg_pool3d(ventricle_features, 1)
        ventricle_features_flat = ventricle_pooled.view(ventricle_pooled.size(0), -1)
        
        # 回归预测
        main_lengths = self.main_fc(global_features_flat)
        ventricle_lengths = self.ventricle_fc(ventricle_features_flat)
        
        # 加权融合结果
        # 对前4个长度使用加权融合
        ventricle_weights = torch.sigmoid(self.fusion_weights[0])
        main_weights = torch.sigmoid(self.fusion_weights[1])
        
        # 归一化权重
        total_weights = ventricle_weights + main_weights
        ventricle_weights = ventricle_weights / total_weights
        main_weights = main_weights / total_weights
        
        # 应用加权融合
        fused_ventricle_lengths = (
            ventricle_weights * ventricle_lengths + 
            main_weights * main_lengths[:, :4]
        )
        
        # 合并结果
        combined_lengths = torch.cat([fused_ventricle_lengths, main_lengths[:, 4:]], dim=1)
        
        # 返回注意力图用于可视化
        return combined_lengths, ventricle_attn_features
    

# 测试数据集类
class TestImageDataset(Dataset):
    def __init__(self, image_dir, label_csv_path):
        self.image_paths = []
        self.image_names = []
        self.label_mapping = {}
        
        # 获取所有NIfTI文件
        img_files = natsorted([f for f in os.listdir(image_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        print(f"Found {len(img_files)} images in test folder")
        
        # 添加图像路径
        for img_file in img_files:
            img_name = img_file[:-7] if img_file.endswith('.nii.gz') else img_file[:-4]
            self.image_paths.append(os.path.join(image_dir, img_file))
            self.image_names.append(img_name)
        
        # 添加标签加载逻辑
        if label_csv_path and os.path.exists(label_csv_path):
            df = pd.read_csv(label_csv_path, dtype={'label_name':str})
            for _, row in df.iterrows():
                img_name = row['label_name']
                # 提取11个生物测量值
                lengths = row[biometry_list].values.astype(float).tolist()
                self.label_mapping[img_name] = lengths
        else:
            print("Warning: Label CSV not provided or not found. Loss will not be computed.")
        
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
        X_T1 = extract_brain(X_T1, ind_brain, [128,160,128])
        data = X_T1.reshape((1,)+X_T1.shape)
        data = torch.tensor(data, dtype=torch.float32)

        # 获取标签
        label = self.label_mapping.get(img_name, None)
        if label is not None:
            label = torch.tensor(label, dtype=torch.float32)
        else:
            # 创建一个长度为11的NaN张量作为占位符
            label = torch.full((len(biometry_list),), float('nan'), dtype=torch.float32)
            
        return data, label, img_name


class WeightedMSELoss(nn.Module):
    def __init__(self, weights=None, ventricle_indices=None):
        super().__init__()
        # 将权重列表转换为张量
        self.weights = torch.tensor(weights, dtype=torch.float32) if weights is not None else torch.ones(11)
        self.ventricle_indices = ventricle_indices if ventricle_indices is not None else []
    
    def forward(self, pred, target):
        # 计算每个特征的MSE
        per_feature_loss = (pred - target)**2
        
        # 移动权重到pred所在的设备
        weights_tensor = self.weights.to(pred.device)
        
        # 应用权重
        weighted_loss = per_feature_loss * weights_tensor
        
        # 额外加强侧脑室损失 - 修改索引方式
        if self.ventricle_indices:
            # 使用单维索引代替二维索引
            ventricle_loss = per_feature_loss[self.ventricle_indices] * 5.0
            weighted_loss[self.ventricle_indices] = ventricle_loss
        
        return weighted_loss.mean()


# 主测试函数
def main():
    # 设备设置
    os.environ["CUDA_VISIBLE_DEVICES"] = "7" 
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建测试数据集
    test_dataset = TestImageDataset(test_image_folder, label_csv_path)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 初始化模型
    # 注意：需要先创建LocalAppearance作为基础模型
    base_model = LocalAppearance(in_channels=1, num_classes=22)  # heatmap_channel=22
    model = EnhancedLengthPredictor(base_model, num_classes=11).to(device)  # length_channel=11
    
    # 加载训练好的模型权重
    if os.path.exists(test_model_path):
        print(f"Loading model weights from {test_model_path}")
        checkpoint = torch.load(test_model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("Model weights loaded successfully")
    else:
        print(f"Error: Model file not found at {test_model_path}")
        return
    
    # 设置为评估模式
    model.eval()
    
    # 准备结果存储
    results = [] 
    # 测试循环
    with torch.no_grad():
        for batch_idx, (images, labels, img_names) in enumerate(test_loader):
            images = images.to(device)
            labels = labels.to(device) if labels[0] is not None else [None] * len(images)
            criterion = WeightedMSELoss(weights=length_weights, ventricle_indices=ventricle_indices)
            
            # 预测
            outputs, attention_map = model(images)
            print(attention_map.shape)
            
            # 处理每个样本
            for i in range(len(img_names)):
                # 计算当前样本的加权MSE
                weighted_mse = criterion(outputs[i], labels[i]).item()
                # 添加到结果
                sample_result = {
                    'label_name': img_names[i],
                    'weighted_mse': weighted_mse,  
                    **{biometry_list[j]: outputs[i][j].item() for j in range(11)}
                }
                results.append(sample_result)
            
            # 打印进度
            print(f"Processed batch {batch_idx+1}/{len(test_loader)} - Saved attention maps for {len(img_names)} images")
    
    # 创建DataFrame并保存为CSV
    df_results = pd.DataFrame(results)
    
    # 重新排列列，将每个长度放在单独列
    columns = ['label_name'] + [biometry_list[i] for i in range(11)] + ['weighted_mse']
    df_results = df_results[columns]
    
    # 保存结果
    df_results.to_csv(output_csv_path, index=False)
    print(f"Test results saved to {output_csv_path}")
    
    # 打印部分结果
    print("\nSample predictions:")
    print(df_results.head())


if __name__ == "__main__":
    main()