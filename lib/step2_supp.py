import os
import ants
import numpy as np
import nibabel as nb
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F


### ————————————————网络拼接———————————————— ###
class TENTWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
        # ----------- 初始化配置 -----------
        # 1. 保持BN层的weight/beta可学习，其他参数冻结
        self._freeze_non_bn_params()
        
        # 2. 配置BN层使用当前批次的统计量
        self._configure_bn_layers()
        
        # 3. 保持Dropout等层为评估模式
        self._set_special_layers_eval()

    def _freeze_non_bn_params(self):
        """冻结所有非BN层参数"""
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 解冻BN层的weight/bias (alpha/beta)
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                if m.weight is not None:
                    m.weight.requires_grad_(True)
                if m.bias is not None:
                    m.bias.requires_grad_(True)

    def _configure_bn_layers(self):
        """配置BN层使用当前batch统计量，且不累积历史"""
        for m in self.model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                # 禁用历史统计量累积
                m.track_running_stats = False
                # 清空现有统计量
                m.reset_running_stats()
                # 强制使用当前batch统计量
                m.momentum = 0.0  # 动量=0表示完全使用当前batch

    def _set_special_layers_eval(self):
        """将Dropout等层设为评估模式"""
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.eval()

    def forward(self, x):
        # 前向传播时强制模型处于"训练模式"以使用当前batch统计量
        self.model.train()  
        return self.model(x)


### ————————————————数据导入———————————————— ###
def extract_brain(data, inds, sz_brain):
    if isinstance(sz_brain, int):
        sz_brain = [sz_brain, sz_brain, sz_brain]
    xsz_brain = inds[1] - inds[0] + 1
    ysz_brain = inds[3] - inds[2] + 1
    zsz_brain = inds[5] - inds[4] + 1
    # print(data.shape)
    # print(inds)
    brain = np.zeros((sz_brain[0], sz_brain[1], sz_brain[2]))
    x_start = int((sz_brain[0] - xsz_brain) / 2)
    y_start = int((sz_brain[1] - ysz_brain) / 2)
    z_start = int((sz_brain[2] - zsz_brain) / 2)
    brain[x_start:x_start+xsz_brain,y_start:y_start+ysz_brain,
          z_start:z_start+zsz_brain] = data[inds[0]:inds[1]+1,inds[2]:inds[3]+1,inds[4]:inds[5]+1]
    return brain

def convert_coords_to_original(coords, inds, sz_brain):
    """
    将extract_brain处理后的坐标系中的坐标转换回原始图像坐标系
    
    Args:
        coords: 在sz_brain坐标系中的坐标，形状为 (N, 3) 或 (N,)，其中N是点的数量
                坐标顺序为 (x, y, z) 对应 (D, H, W)
        inds: 原始图像中脑区的边界索引 [xmin, xmax, ymin, ymax, zmin, zmax]
        sz_brain: 目标尺寸 [D, H, W] 或单个整数
    
    Returns:
        original_coords: 在原始图像坐标系中的坐标，形状与coords相同
    """
    if isinstance(sz_brain, int):
        sz_brain = [sz_brain, sz_brain, sz_brain]
    
    # 确保coords是2D数组，记录原始形状
    coords = np.asarray(coords)
    was_1d = coords.ndim == 1
    if was_1d:
        coords = coords.reshape(1, -1)
    
    # 计算原始脑区的尺寸
    xsz_brain = inds[1] - inds[0] + 1
    ysz_brain = inds[3] - inds[2] + 1
    zsz_brain = inds[5] - inds[4] + 1
    
    # 计算在sz_brain坐标系中的起始偏移（居中偏移）
    x_start = int((sz_brain[0] - xsz_brain) / 2)
    y_start = int((sz_brain[1] - ysz_brain) / 2)
    z_start = int((sz_brain[2] - zsz_brain) / 2)
    
    # 将坐标从sz_brain坐标系转换到原始脑区坐标系
    # coords的坐标顺序是 (x, y, z) 对应 (D, H, W)
    # 即 coords[:, 0] 对应 D维度，coords[:, 1] 对应 H维度，coords[:, 2] 对应 W维度
    original_coords = coords.copy()
    original_coords[:, 0] = coords[:, 0] - x_start + inds[0]  # D维度 -> X维度
    original_coords[:, 1] = coords[:, 1] - y_start + inds[2]  # H维度 -> Y维度
    original_coords[:, 2] = coords[:, 2] - z_start + inds[4]  # W维度 -> Z维度
    
    # 如果输入是1D，返回1D
    if was_1d:
        return original_coords[0]
    
    return original_coords

def block_ind(mask):
    tmp = np.nonzero(mask); # 元组(array([0, 0, 1, 1, 1]), array([0, 1, 0, 0, 1]), array([1, 0, 0, 1, 0]))
    xind = tmp[0]
    yind = tmp[1]
    zind = tmp[2]
    xmin = np.min(xind); xmax = np.max(xind)
    ymin = np.min(yind); ymax = np.max(yind)
    zmin = np.min(zind); zmax = np.max(zind)
    ind_brain = [xmin, xmax, ymin, ymax, zmin, zmax]
    return ind_brain

class TestDataset(torch.utils.data.Dataset): # 给一个文件路径而非目录
    def __init__(self, target_image_path, pred_length_df):
        self.target_image_path = target_image_path
        self.target_image = [nb.load(self.target_image_path).get_fdata()]
        self.affine = [nb.load(self.target_image_path).affine]
        self.target_image_name = target_image_path.split("/")[-1][:-7]
        self.pred_length_df = pred_length_df
        
    def __len__(self):
        return len(self.target_image)

    def detect_orientation(self, affine_matrix):
        """
        通过仿射矩阵检测图像方向
        返回：'neurological'(神经学) 或 'radiological'(放射学)
        """
        # 提取方向向量 (仿射矩阵的前3列)
        orientation_vec = affine_matrix[:3, :3]
        
        # 检查X轴方向 (第一列)
        if orientation_vec[0, 0] > 0:
            # print('RAS')
            return 'radiological'  # RAS 坐标系
        else:
            # print('LPS')
            return 'neurological'  # LPS 坐标系
    
    def __getitem__(self, idx):
        image_data = self.target_image[idx]
        X_T1  = image_data.copy()
        img_affine = self.affine[idx]
        # 标准化
        mask = X_T1 > 0
        X_T1_brain = X_T1[mask]
        mean = np.mean(X_T1_brain)
        std = np.std(X_T1_brain)
        normalized_brain = (X_T1_brain - mean) / std
        X_T1[mask] = normalized_brain
        X_T1[~mask] = 0
        # 缩放
        ind_brain = block_ind(mask)
        X_T1 = extract_brain(X_T1, ind_brain, [128,160,128])
        
        # # 方向变换
        # orientation = self.detect_orientation(img_affine)
        # if orientation == 'neurological': 
        #     # 交换x轴和z轴
        #     X_T1 = np.swapaxes(X_T1, 0, 2)  # 交换x(0)和z(2)轴
        #     # 翻转y轴
        #     X_T1 = X_T1[:, ::-1, :].copy()  # 关键修改：添加.copy()确保连续存储
        #     # 更新仿射矩阵
        #     swap_matrix = np.eye(4)
        #     swap_matrix[:3, :3] = img_affine[:3, [2, 1, 0]]  # 交换X和Z列
        #     swap_matrix[:3, 3] = img_affine[:3, 3]  # 保持平移不变
        #     y_vector = swap_matrix[:3, 1].copy()   # 获取y轴方向向量
        #     swap_matrix[:3, 1] = -y_vector         # 翻转y轴方向
        #     swap_matrix[:3, 3] += y_vector * (X_T1.shape[1] - 1)  # 调整平移项
        #     img_affine = swap_matrix

        data = X_T1.reshape((1,)+X_T1.shape)
        data = torch.tensor(data, dtype=torch.float32) # torch.Size([1, 128, 160, 128])
        # 长度伪标签
        row = self.pred_length_df.loc[
            self.pred_length_df['label_name'] == self.target_image_name, 
            self.pred_length_df.columns != 'label_name'  # 显式排除label列
        ].values.astype(np.float32)
        pred_length = torch.tensor(row.squeeze(), dtype=torch.float32) # torch.Size([11])
        # 数据和伪标签组装成字典格式
        data_pair = {'image': data, 'length': pred_length}
        return data_pair


### ————————————————损失函数———————————————— ###
class EntropyLoss(nn.Module):
    def __init__(self, epsilon=1e-12):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C = x.shape[:2]
        x_flat = x.view(B * C, -1)  # 展平空间维度[B*C, D*H*W]
        # 数值稳定的Softmax计算
        max_vals = x_flat.max(dim=1, keepdim=True).values  # 每个通道的最大值[B*C, 1]
        data_exp = torch.exp(x_flat - max_vals)            # 减去最大值[B*C, N]
        sum_exp = data_exp.sum(dim=1, keepdim=True)        # 分母[B*C, 1]
        data_softmax = data_exp / sum_exp                  # Softmax概率[B*C, N]
        # 数值稳定性处理
        data_clipped = torch.clamp(data_softmax, min=self.epsilon)
        # 计算熵项：-p*log(p)
        p_logp = - data_clipped * torch.log(data_clipped)  # [B*C, N]
        # 对每个通道求和得到熵 [B*C]
        channel_entropies = p_logp.sum(dim=1)
        # 恢复形状为 [B, C] 并取平均
        return channel_entropies.view(B, C).mean(dim=[0,1])  # 先通道平均再batch平均
    

class DifferentiableLandmarkDetector(nn.Module):
    def __init__(self, init_temp, topk):
        """
        可微分地标检测器 - 稀疏优化版
        """
        super().__init__()
        # self.temperature = nn.Parameter(torch.tensor(init_temp))
        self.temperature = init_temp
        self.topk = topk
        
    def forward(self, heatmap):
        """
        heatmap: 输入热图张量，形状为 [B, C, D, H, W]
        返回: 预测坐标，形状为 [B, C, 3] (顺序: x, y, z)
        """
        # 创建坐标网格
        B, C, D, H, W = heatmap.shape
        # print(torch.max(heatmap),torch.min(heatmap))
        device = heatmap.device
        x, y, z = torch.meshgrid(
            torch.arange(D, device=device),  # x: 深度方向
            torch.arange(H, device=device),  # y: 高度方向
            torch.arange(W, device=device),  # z: 宽度方向
            indexing='ij'  # 矩阵索引 (i,j,k)
        )
        # 堆叠坐标
        coords = torch.stack((x, y, z), dim=-1).float()  # [D, H, W, 3]
        coords = coords.permute(3, 0, 1, 2)  # 调整为 [3, D, H, W] 以匹配热图形状
        
        # 展平坐标以便处理
        coords_flat = coords.reshape(3, -1).permute(1, 0)  # [D*H*W, 3]
        heat_flat = heatmap.view(B, C, -1)  # [B, C, D*H*W]
        # 找出每个通道的前k个值
        actual_topk = min(self.topk, heat_flat.size(-1))
        values, indices = torch.topk(heat_flat, actual_topk, dim=-1)
        
        # 数值稳定的权重计算
        weights = values / self.temperature 
        weights = weights - weights.max(dim=-1, keepdim=True).values
        exp_weights = torch.exp(weights)
        probs = exp_weights / (exp_weights.sum(dim=-1, keepdim=True) + 1e-20)

        # 收集对应的坐标
        selected_coords = coords_flat[indices.view(-1)].view(B, C, actual_topk, 3)
        # 计算加权坐标
        weighted_coords = torch.sum(probs.unsqueeze(-1) * selected_coords, dim=2)

        return weighted_coords


class LengthLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, keypoints, length_pseudo_label):
        """
        keypoints: 输入从热图中提取的坐标，形状为 [B, 22, 3]
        length_pseudo_label: 伪长度标签，形状为 [B, 11]
        """
        C = keypoints.shape[1]  # 点数
        # 计算11个长度 (相邻点对: 0-1, 2-3, ..., 20-21)
        pred_lengths = []
        for i in range(0, C, 2):
            # 计算每对点的欧氏距离 [B]
            dist = torch.norm(keypoints[:, i, :] - keypoints[:, i+1, :], dim=-1)
            pred_lengths.append(dist)
        # 组合成预测长度张量 [B, 11]
        pred_lengths = torch.stack(pred_lengths, dim=1)
        # print(pred_lengths)
        # print(length_pseudo_label)
        
        # 计算MSE损失
        loss = F.mse_loss(pred_lengths, length_pseudo_label)
        return loss

class BoundaryGradientLoss(nn.Module):
    """
    Args:
        point_pairs: 需要应用损失的点对索引列表 (从1开始计数)
        kernel_type: 梯度算子类型 ('sobel' 或 'scharr')
        smooth_sigma: 高斯平滑的sigma值
        grad_sigma: 梯度核的高斯权重sigma
        adaptive_threshold: 是否使用自适应梯度阈值
        min_grad: 最小梯度阈值 (当不使用自适应阈值时)
    """
    
    def __init__(self, 
                 point_pairs: List[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                 kernel_type: str = 'scharr',
                 smooth_sigma: float = 1.0,
                 grad_sigma: float = 1.0,
                 adaptive_threshold: bool = True,
                 min_grad: float = 0.3):
        super().__init__()
        
        # 参数校验
        if kernel_type not in ['sobel', 'scharr']:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")
        
        # 将点对索引转换为点索引 (每对点有2个点)
        self.point_indices = []
        for pair_idx in point_pairs:
            if pair_idx < 1:
                raise ValueError("Point pair indices start from 1")
            # 计算点索引: (pair_idx-1)*2 和 (pair_idx-1)*2 + 1
            start_idx = (pair_idx - 1) * 2
            self.point_indices.extend([start_idx, start_idx + 1])
        
        self.kernel_type = kernel_type
        self.smooth_sigma = smooth_sigma
        self.grad_sigma = grad_sigma
        self.adaptive_threshold = adaptive_threshold
        self.min_grad = min_grad
        
        # 创建高斯平滑核
        self.smooth_kernel = self._create_smooth_kernel()

        # 创建梯度核
        self.gradient_kernel = self._create_gradient_kernel()
    
    def _create_smooth_kernel(self) -> torch.Tensor:
        """创建1D高斯平滑核"""
        # 自动确定核大小
        kernel_size = max(3, int(2 * 2 * self.smooth_sigma + 1))
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # 创建1D高斯核
        k = torch.arange(kernel_size) - kernel_size//2
        k = torch.exp(-k**2 / (2 * self.smooth_sigma**2))
        return k / k.sum()
    
    def _gaussian_smooth(self, volume: torch.Tensor) -> torch.Tensor:
        """使用可分离卷积进行3D高斯平滑"""
        smoothed = volume.clone()
        kernel = self.smooth_kernel.to(volume.device)
        
        # 三个方向分别卷积
        for dim, padding_size in zip([2, 3, 4], 
                                [kernel.size(0)//2, kernel.size(0)//2, kernel.size(0)//2]):
            # 创建适合当前维度的卷积核
            kernel_shape = [1, 1, 1, 1, 1]
            kernel_shape[dim] = -1
            conv_kernel = kernel.view(*kernel_shape)
            
            # 手动添加padding以保持尺寸
            pad_size = kernel.size(0) - 1
            pad_before = pad_size // 2
            pad_after = pad_size - pad_before
            
            # 应用填充
            if dim == 2:  # D维度 (深度)
                smoothed = F.pad(smoothed, (0, 0, 0, 0, pad_before, pad_after), mode='replicate')
            elif dim == 3:  # H维度 (高度)
                smoothed = F.pad(smoothed, (0, 0, pad_before, pad_after, 0, 0), mode='replicate')
            else:  # W维度 (宽度)
                smoothed = F.pad(smoothed, (pad_before, pad_after, 0, 0, 0, 0), mode='replicate')
            
            # 执行卷积 (使用padding=0)
            smoothed = F.conv3d(
                smoothed, 
                conv_kernel,
                padding=0
            )
        
        return smoothed
    
    def _create_gradient_kernel(self) -> torch.Tensor:
        """创建3D梯度核 (Z,Y,X 方向)"""
        if self.kernel_type == 'sobel':
            # 标准3D Sobel核 - 所有核形状统一为3x3x3
            kx = torch.tensor([
                [[1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]],
                
                [[2, 0, -2],
                [4, 0, -4],
                [2, 0, -2]],
                
                [[1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]]
            ], dtype=torch.float32)
            
            ky = torch.tensor([
                [[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]],
                
                [[2, 4, 2],
                [0, 0, 0],
                [-2, -4, -2]],
                
                [[1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]]
            ], dtype=torch.float32)
            
            kz = torch.tensor([
                [[1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]],
                
                [[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
                
                [[-1, -2, -1],
                [-2, -4, -2],
                [-1, -2, -1]]
            ], dtype=torch.float32)
        else:  # scharr
            # Scharr核 - 所有核形状统一为3x3x3
            kx = torch.tensor([
                [[3, 0, -3],
                [10, 0, -10],
                [3, 0, -3]],
                
                [[10, 0, -10],
                [30, 0, -30],
                [10, 0, -10]],
                
                [[3, 0, -3],
                [10, 0, -10],
                [3, 0, -3]]
            ], dtype=torch.float32)
            
            ky = torch.tensor([
                [[3, 10, 3],
                [0, 0, 0],
                [-3, -10, -3]],
                
                [[10, 30, 10],
                [0, 0, 0],
                [-10, -30, -10]],
                
                [[3, 10, 3],
                [0, 0, 0],
                [-3, -10, -3]]
            ], dtype=torch.float32)
            
            kz = torch.tensor([
                [[3, 10, 3],
                [10, 30, 10],
                [3, 10, 3]],
                
                [[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]],
                
                [[-3, -10, -3],
                [-10, -30, -10],
                [-3, -10, -3]]
            ], dtype=torch.float32)
        
        # 确保所有核形状一致 (3, 3, 3)
        kx = kx.reshape(3, 3, 3)
        ky = ky.reshape(3, 3, 3)
        kz = kz.reshape(3, 3, 3)
        
        # 组合核 [3, 1, 3, 3, 3]
        kernel = torch.stack([kz, ky, kx])[:, None, ...]
        
        # 应用高斯权重
        if self.grad_sigma > 0:
            z, y, x = torch.meshgrid(
                torch.arange(3), 
                torch.arange(3), 
                torch.arange(3), 
                indexing='ij'
            )
            center = torch.tensor([1.0, 1.0, 1.0])
            dist = torch.sqrt((z-center[0])**2 + (y-center[1])**2 + (x-center[2])**2)
            gauss = torch.exp(-dist**2 / (2 * self.grad_sigma**2))
            kernel = kernel * gauss[None, None, ...]
        
        return kernel / kernel.abs().sum(dim=(2, 3, 4), keepdim=True)
    
    def _compute_gradient_magnitude(self, volume: torch.Tensor) -> torch.Tensor:
        """计算3D梯度幅值图"""
        # 1. 高斯平滑预处理
        smoothed = self._gaussian_smooth(volume)
        
        # 2. 计算梯度
        gradients = F.conv3d(
            smoothed, 
            self.gradient_kernel.to(volume.device), 
            padding=1  # 3x3x3核的padding
        )
        
        # 3. 计算梯度幅值 (L2范数)
        grad_magnitude = torch.sqrt(torch.sum(gradients**2, dim=1, keepdim=True) + 1e-8)
        
        return grad_magnitude
    
    def _sample_points(self, 
                      volume: torch.Tensor, 
                      points: torch.Tensor) -> torch.Tensor:
        """在三线性插值采样体积值"""
        # 归一化点坐标到[-1,1]
        if volume.dim() == 4:
            volume = volume.unsqueeze(1)
        D, H, W = volume.shape[2:]
        grid = points.clone()
        grid[..., 0] = 2.0 * points[..., 0] / (D - 1) - 1.0  # Z
        grid[..., 1] = 2.0 * points[..., 1] / (H - 1) - 1.0  # Y
        grid[..., 2] = 2.0 * points[..., 2] / (W - 1) - 1.0  # X
        
        # 添加batch维度 [B, N, 1, 1, 3]
        grid = grid.unsqueeze(2).unsqueeze(2)
        
        # 采样 [B, C, N, 1, 1]
        return F.grid_sample(volume, grid, 
                            mode='bilinear', 
                            align_corners=True).squeeze(-1).squeeze(-1)
    
    def forward(self,
               volume: torch.Tensor, 
               pred_points: torch.Tensor) -> torch.Tensor:
        """
        计算边界梯度损失
        
        Args:
            volume: 3D MRI体积 [B, C, D, H, W]
            pred_points: 预测的点坐标 [B, C, 3] 
            
        Returns:
            loss: 边界梯度损失
        """
        B, C, D, H, W = volume.shape
        
        # 1. 选择需要计算损失的点
        selected_points = pred_points[:, self.point_indices]  # [B, M, 3]
        
        # 2. 计算全图的梯度幅值
        grad_map = self._compute_gradient_magnitude(volume)  # [B, 1, D, H, W]
        
        # 3. 在点位置采样梯度值
        sampled_grads = self._sample_points(grad_map, selected_points)  # [B, M]
        # print(sampled_grads)
        
        # 4. 确定梯度阈值
        if self.adaptive_threshold:
            with torch.no_grad():
                # 使用梯度图的中位数作为阈值
                threshold = torch.median(grad_map.view(B, -1), dim=1)[0]
                threshold = threshold.view(B, 1)
        else:
            threshold = self.min_grad  # 固定阈值
            
        # 5. 计算边界损失 - 鼓励点位于高梯度区域
        # 损失 = max(0, threshold - 实际梯度值)
        loss_per_point = F.relu(threshold - sampled_grads)
        
        # 6. 平均所有点的损失
        return loss_per_point.mean()
    

### —————————————图谱配准，解剖先验约束热图———————————— ###
def register_template_label(template_image_path, template_label_path, target_image_path, output_path): # 预测热图已读入
    # 1.图片配准至模板然后label反配准
        # 读入模板图片
    template_data = ants.image_read(template_image_path)    
        # 读入数据集图片
    image_data = ants.image_read(target_image_path)
        # 图片配准至模板
    reg_result = ants.registration(
        fixed = template_data,
        moving = image_data,
        type_of_transform = 'SyN', 
        reg_iterations = (150, 100, 70), # 粗、中、 细尺度配准中每个分辨率阶段的迭代次数
        syn_sampling = 16, # 控制变形场采样率，数值小，采样更密集，变形场更精细
        verbose=False
    )
        # label反配准至数据集图片
    template_label = ants.image_read(template_label_path)
    registered_label = ants.apply_transforms( # apply_transforms
        fixed  = image_data,
        moving = template_label,  
        transformlist = reg_result['invtransforms'], # 使用逆向变换
        interpolator = 'nearestNeighbor' # 插值方式
    )
        # 保存配准后的label
    ants.image_write(registered_label, output_path)
        # 清理临时文件
    if 'fwdtransforms' in reg_result:
        for file in reg_result['fwdtransforms']:
            if os.path.exists(file):
                os.unlink(file)  # 删除正向变换文件
    if 'invtransforms' in reg_result:
        for file in reg_result['invtransforms']:
            if os.path.exists(file):
                os.unlink(file)  # 删除逆向变换文件
    
def crop_template_label(registered_label, image_data):
    # 2.匹配后模板缩放
        # 提取ind
    mask = image_data > 0
    ind_brain = block_ind(mask)
        # crop
    cropped_template_label = extract_brain(registered_label,ind_brain,[128,160,128]) 
    return cropped_template_label

def constrain_heatmap(cropped_template_label, data, label_value, radius, template_label_scale, template_label_pan):
    heatmap_mask = np.zeros_like(cropped_template_label, dtype=np.int32)
    # 对模板热图约束进行缩放处理
    voxels = np.argwhere(cropped_template_label == label_value) # 找到模板热图中等于当前标签值的体素
    if len(voxels) > 0:  # 确保当前标签值存在
        # 计算几何中心
        geometric_center = np.mean(voxels, axis=0)
        # 创建距离场
        z, y, x = np.indices(cropped_template_label.shape)
        distances = np.sqrt(
            (z - geometric_center[0])**2 +
            (y - geometric_center[1])**2 +
            (x - geometric_center[2])**2
        )
        # 标记距离几何中心template_label_scale像素以内的体素
        heatmap_mask[distances < radius] = 1
    # 对 data 进行约束
    original_data = data.copy()
    data[heatmap_mask == 1] = template_label_scale * original_data[heatmap_mask == 1] + template_label_pan
    data[heatmap_mask == 0] = -1000  # 其他区域置为 -1000
    return data



### ————————————————提取坐标———————————————— ###
def get_landmark_from_heatmap(path_or_data, slice_index, detector=None, init_temp=0.05, topk=50):
    # 创建检测器（如果未提供）
    if detector is None:
        detector = DifferentiableLandmarkDetector(init_temp=init_temp, topk=topk)
        detector = detector.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    if isinstance(path_or_data, str):
        # 是路径，加载热图数据
        path = path_or_data
        img = nb.load(path)
        heatmap_data = img.get_fdata()  # [D, H, W]
    elif isinstance(path_or_data, np.ndarray):
        # 已经是数据，直接用
        heatmap_data = path_or_data
    
    # 转换为PyTorch张量并添加批次和通道维度
    heatmap = torch.tensor(heatmap_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [B=1, C=1, D, H, W]
    # 移动到GPU（如果可用）
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    heatmap = heatmap.to(device)
    detector = detector.to(device)
    
    # 使用检测器提取坐标
    with torch.no_grad():
        peak_coord = detector(heatmap)  # [B=1, C=1, 3]
    # 获取点的坐标
    coordinate = np.round(peak_coord[0, 0].cpu().numpy())
    # 根据方向获取层数
    slice_sum_max_index = coordinate[slice_index].astype(int) 
    
    return coordinate, slice_sum_max_index


### ————————————————可视化结果———————————————— ###
def plot_slice(mask, img, slice_index, orientation, up_bound, low_bound, coordinate):
    # 根据扫描方向提取对应切片（二维），方向用0，1，2表示
    if orientation == 2:
        mask_slc = mask[:, :, slice_index]
        img_slc = img[:, :, slice_index]
    elif orientation == 0:
        mask_slc = mask[slice_index, :, :]
        img_slc = img[slice_index, :, :]
    elif orientation == 1:
        mask_slc = mask[:, slice_index, :]
        img_slc = img[:, slice_index, :]
    else:
        raise ValueError("Invalid orientation value. Must be 0, 1, or 2.")

    # 归一化图像数据到 [0, 255] 范围
    img_ups = ((img_slc - low_bound) / (up_bound - low_bound) * 255).astype(np.uint8)
    mask_ups = mask_slc

    # 创建RGB图像
    image_rgb = np.zeros((*img_ups.shape, 3), dtype=np.uint8)
    image_rgb[..., 0] = img_ups  # R
    image_rgb[..., 1] = img_ups  # G
    image_rgb[..., 2] = img_ups  # B
    image_rgb[~mask_ups] = [0, 0, 0]  # 提取mask部分

    # 收集所有需要绘制红叉的坐标
    cross_positions = []
    for ii in range(coordinate.shape[0]):
        if orientation == 2:
            x = coordinate[ii, 0]
            y = coordinate[ii, 1]
        elif orientation == 0:
            x = coordinate[ii, 1]
            y = coordinate[ii, 2]
        elif orientation == 1:
            x = coordinate[ii, 0]
            y = coordinate[ii, 2]
        else:
            continue

        # 确保坐标在图像范围内
        if 0 <= x < image_rgb.shape[0] and 0 <= y < image_rgb.shape[1]:
            cross_positions.append((x, y))

    # 返回图像、叉的位置列表以及图像的横纵大小
    return image_rgb, cross_positions

def rotate_coordinate(x, y, k, img_width, img_height):
    """
    根据旋转参数 k 和图片尺寸，计算旋转后的坐标。
    :param x: 原始横坐标 (列索引)
    :param y: 原始纵坐标 (行索引)
    :param k: 旋转次数 (1, 2, 或 3)
    :param img_width: 图片宽度
    :param img_height: 图片高度
    :return: 旋转后的 (x_new, y_new)
    """
    if k == 1:  # 逆时针旋转 90°
        x_new = y
        y_new = img_width - 1 - x
    elif k == 2:  # 旋转 180°
        x_new = img_height - 1 - x
        y_new = img_width - 1 - y
    elif k == 3:  # 顺时针旋转 90°
        x_new = img_height - 1 - y
        y_new = x
    else:
        raise ValueError("Invalid rotation parameter k. Must be 1, 2, or 3.")
    return x_new, y_new