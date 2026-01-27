"""
网络：SCN空间、CNN外观
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence
from functools import partial
from torch.utils.checkpoint import checkpoint
from timm.models.layers import drop_path, trunc_normal_


ACT = {'relu': nn.ReLU, 'leaky': nn.LeakyReLU, 'prelu': nn.PReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid}


class SCN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        filters: int = 64,
        factor: int = 8,
        dropout: float = 0.,
        mode: str = 'add',
        local_act: str = None,
        spatial_act: str = 'relu',
    ):
        super().__init__() # 父类nn.Module的初始化
        self.HLA = LocalAppearance(in_channels, num_classes, filters, dropout, mode)
        self.down = nn.AvgPool3d(factor, factor, ceil_mode=True)
        self.up = nn.Upsample(scale_factor=factor, mode='trilinear', align_corners=True)
        self.local_act = ACT[local_act]() if local_act else None # 局部激活函数
        self.spatial_act = ACT[spatial_act]() # 局部激活函数
        self.HSC = nn.Sequential( # 不改变特征图大小和通道数(filters)
            nn.Conv3d(filters, filters, 5, 1, 2, bias=False),
            nn.Conv3d(filters, filters, 5, 1, 2, bias=False),
            nn.Conv3d(filters, filters, 5, 1, 2, bias=False),
            nn.Conv3d(filters, num_classes, 5, 1, 2 , bias=False),
        )
        nn.init.trunc_normal_(self.HSC[-1].weight, 0, 1e-4) # 对 HSC 模块的最后一层卷积的权重进行截断正态分布初始化

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]: # x是输入
        d1, HLA = self.HLA(x)
        if self.local_act:
            HLA = self.local_act(HLA)
        HSC = self.up(self.spatial_act(self.HSC(self.down(d1)))) # d1->池化->4次卷积->上采样恢复尺寸
        #print(HSC.shape)
        heatmap = HLA * HSC
        return heatmap, HLA, HSC


class LocalAppearance(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        filters: int = 64,
        dropout: float = 0.,
        mode: str = 'add',
    ):
        super().__init__() # 父类nn.Module的初始化
        self.mode = mode
        self.pool = nn.AvgPool3d(2, 2, ceil_mode=True)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.in_conv = self.Block(in_channels, filters) # 通道由in_channels到filters,大小不变,卷积
        self.out_conv = nn.Conv3d(filters, num_classes, 1, bias=False) # 通道由in_channels到filters,大小不变,不卷积
        # 修改处：添加 Sigmoid 激活层
        # self.out_conv = nn.Sequential(
        #     nn.Conv3d(filters, num_classes, 1, bias=False),
        #     nn.Sigmoid()  # 新增激活函数
        # )
        self.enc1 = self.Block(filters, filters, dropout) # 不停卷积
        self.enc2 = self.Block(filters, filters, dropout)
        self.enc3 = self.Block(filters, filters, dropout)
        self.enc4 = self.Block(filters, filters, dropout)
        if mode == 'add':
            self.dec3 = self.Block(filters, filters, dropout) # 现在的图与原来的enc求和再卷积
            self.dec2 = self.Block(filters, filters, dropout)
            self.dec1 = self.Block(filters, filters, dropout)
        else:
            self.dec3 = self.Block(2*filters, filters, dropout)
            self.dec2 = self.Block(2*filters, filters, dropout)
            self.dec1 = self.Block(2*filters, filters, dropout)
        nn.init.trunc_normal_(self.out_conv.weight, 0, 1e-4)
        # 修改处：调整初始化对象（针对 Sequential 中的 Conv3d）
        # nn.init.trunc_normal_(self.out_conv[0].weight, 0, 1e-4)  # 初始化第一个子模块（Conv3d）

    def Block(self, in_channels, out_channels, dropout=0): # 不改变特征图大小,主要是用卷积核糊化,CNN
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
        #print(e1.shape)
        e2 = self.enc2(self.pool(e1)) # self.pool,特征图大小减半
        #print(e2.shape)
        e3 = self.enc3(self.pool(e2))
        #print(e3.shape)
        e4 = self.enc4(self.pool(e3))
        #print(e4.shape)
        if self.mode == 'add':
            d3 = self.dec3(self.up(e4)+e3)
            #print(d3.shape)
            d2 = self.dec2(self.up(d3)+e2)
            #print(d2.shape)
            d1 = self.dec1(self.up(d2)+e1)
            #print('d1:',d1.shape)
        else: # 理解为 mode == 'cat'
            d3 = self.dec3(torch.cat([self.up(e4), e3], dim=1))
            d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))

        # out和d1的区别:out_conv卷积核为1,只是通道数由filters变为num_classes
        out = self.out_conv(d1)
        return d1, out # torch.sigmoid(out)

class PointPredictor(nn.Module):
    def __init__(self, channel: int):
        super().__init__()
        # 共享的特征提取网络（输入为单通道）
        self.shared_conv = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),  # 保持空间维度
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # 共享的回归头
        self.shared_regressor = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),  # 全局空间平均池化
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 每个通道输出6个坐标
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: # x是heatmap
        B, C, D, H, W = x.shape
        # 将通道维度合并到批次维度
        # 新形状: (B*C, 1, D, H, W)
        x = x.view(B*C, 1, D, H, W)
        # 共享卷积特征提取
        features = self.shared_conv(x)  # (B*C, 64, D, H, W)
        # 共享回归头
        coords = self.shared_regressor(features)  # (B*C, 6)
        # 恢复原始维度
        coords = coords.view(B, C, 6)  # (B, C=11, 6)
        return coords


class CombinedModel(nn.Module):
    def __init__(self, scn_model, point_predictor):
        super().__init__()
        self.scn_model = scn_model
        self.point_predictor = point_predictor

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        heatmap, HLA, HSC = self.scn_model(x)
        points = self.point_predictor(heatmap)
        return heatmap, HLA, HSC, points


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        filters: int = 64,
    ) -> None:
        super().__init__()
        self.in_conv = self.Block(in_channels, filters)
        self.enc1 = self.Block(filters, filters)
        self.enc2 = self.Block(filters, filters)
        self.enc3 = self.Block(filters, filters)
        self.enc4 = self.Block(filters, filters)
        self.enc5 = self.Block(filters, filters)
        self.dec4 = self.Block(2*filters, filters)
        self.dec3 = self.Block(2*filters, filters)
        self.dec2 = self.Block(2*filters, filters)
        self.dec1 = self.Block(2*filters, filters)
        self.out_conv = nn.Conv3d(filters, num_classes, 1, bias=False)
        self.pool = nn.AvgPool3d(2, 2, ceil_mode=True)
        self.up = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.in_conv(x)
        e1 = self.enc1(x0)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        e5 = self.enc5(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up(e5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))

        out = self.out_conv(d1)
        return out

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

# -------------------------
# Utilities
# -------------------------
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # shape: [B, 1, 1, ...] broadcasting
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0:
        random_tensor = random_tensor.div(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    def extra_repr(self):
        return f"p={self.drop_prob}"


# -------------------------
# MLP / Attention / Block
# -------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.fc2(self.act(self.fc1(x))))
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = (attn_head_dim or (dim // num_heads))
        all_head_dim = head_dim * num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 初始化权重以防止数值不稳定
        nn.init.xavier_uniform_(self.qkv.weight)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, N, C = x.shape
        
        # 检查输入是否包含NaN或无穷大
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: Input contains NaN or Inf values in Attention")
            print(f"x shape: {x.shape}, x min: {x.min()}, x max: {x.max()}")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        
        # 检查q, k, v是否包含NaN或无穷大
        if torch.isnan(q).any() or torch.isinf(q).any():
            print(f"Warning: q contains NaN or Inf values")
            q = torch.nan_to_num(q, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(k).any() or torch.isinf(k).any():
            print(f"Warning: k contains NaN or Inf values")
            k = torch.nan_to_num(k, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(v).any() or torch.isinf(v).any():
            print(f"Warning: v contains NaN or Inf values")
            v = torch.nan_to_num(v, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # 计算注意力权重
        attn_weights = q @ k.transpose(-2, -1)
        
        # 检查注意力权重是否包含NaN或无穷大
        if torch.isnan(attn_weights).any() or torch.isinf(attn_weights).any():
            print(f"Warning: Attention weights contain NaN or Inf values")
            print(f"attn_weights shape: {attn_weights.shape}")
            print(f"attn_weights min: {attn_weights.min()}, max: {attn_weights.max()}")
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=1.0, neginf=-1.0)
        
        attn = attn_weights.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj_drop(self.proj(x))
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path_rate=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, attn_head_dim=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_scale, attn_drop, drop, attn_head_dim)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(dim, mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# -------------------------
# Patch Embedding (3D)
# -------------------------
class PatchEmbed(nn.Module):
    """3D Image to Patch Embedding"""
    def __init__(self, img_size=(128,160,128), patch_size=8, in_chans=1, embed_dim=64):
        super().__init__()
        if isinstance(img_size, int):
            H = W = D = img_size
        else:
            H, W, D = img_size
        assert H % patch_size == 0 and W % patch_size == 0 and D % patch_size == 0, \
            "img_size must be divisible by patch_size for simple conv embedding."

        self.grid_size = (H // patch_size, W // patch_size, D // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x):
        # x: [B, C, H, W, D]
        x = self.proj(x)  # [B, E, Hp, Wp, Dp]
        Hp, Wp, Dp = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)  # [B, N, E]
        return x, (Hp, Wp, Dp)


# -------------------------
# DoubleConv Block
# -------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=pad, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


# -------------------------
# ViTPose3D (Fixed)
# -------------------------
class ViTPose3D(nn.Module):
    def __init__(self,
                 img_size=(128, 160, 128),
                 patch_size=8,
                 in_chans=1,
                 num_classes=22,
                 embed_dim=64,
                 depth=8,
                 num_heads=8,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.1,
                 norm_layer=None,
                 out_activation: str = 'sigmoid'  # 'none' or 'sigmoid'
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.out_activation = out_activation
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # Patch Embedding + Positional Embedding（无 cls token）
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # 轻量级低层特征（skip）：将输入下采样 8×，对齐到 (Hp,Wp,Dp)
        # 3 次 stride=2 的 3x3x3 卷积： 1 -> 16 -> 32 -> embed_dim
        self.low_level = nn.Sequential(
            nn.Conv3d(in_chans, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(16), nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            nn.Conv3d(32, embed_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm3d(embed_dim), nn.ReLU(inplace=True),
        )

        # Transformer Encoder
        dpr = torch.linspace(0, drop_path_rate, steps=depth).tolist()
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias, qk_scale,
                  drop_rate, attn_drop_rate, dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.last_norm = norm_layer(embed_dim)

        # 解码器：3 次上采样回到原分辨率
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.fuse1 = DoubleConv(embed_dim * 2, embed_dim)   # 与低层特征融合
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv2 = DoubleConv(embed_dim, embed_dim)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        self.conv3 = DoubleConv(embed_dim, embed_dim)

        self.head = nn.Conv3d(embed_dim, num_classes, kernel_size=1, bias=True)

        self.apply(self._init_weights)
        trunc_normal_(self.pos_embed, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            trunc_normal_(m.weight, std=1e-4)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm3d, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0.)
            nn.init.constant_(m.weight, 1.)

    def forward_features(self, x):
        B, C, H, W, D = x.shape
        tokens, (Hp, Wp, Dp) = self.patch_embed(x)     # [B, N, E], N=Hp*Wp*Dp
        tokens = tokens + self.pos_embed               # 正确：仅加一次 positional embedding

        for blk in self.blocks:
            tokens = blk(tokens)

        tokens = self.last_norm(tokens)
        # -> feature map
        feat = tokens.transpose(1, 2).reshape(B, self.embed_dim, Hp, Wp, Dp).contiguous()
        return feat

    def forward(self, x):
        B, C, H, W, D = x.shape

        # Transformer path
        feat = self.forward_features(x)                       # [B, E, H/8, W/8, D/8]

        # Low-level skip (aligned at 1/8)
        low = self.low_level(x)                               # [B, E, H/8, W/8, D/8]

        # Fuse and upsample x3
        y = torch.cat([feat, low], dim=1)
        y = self.fuse1(y)
        y = self.up1(y)                                       # /4
        y = self.conv2(y)
        y = self.up2(y)                                       # /2
        y = self.conv3(y)
        y = self.up3(y)                                       # /1

        # 最后确保与输入完全同尺寸（避免边界尺寸误差）
        if y.shape[2:] != (H, W, D):
            y = F.interpolate(y, size=(H, W, D), mode='trilinear', align_corners=False)

        logits = self.head(y)                                 # [B, 22, H, W, D]

        if self.out_activation == 'sigmoid':
            return torch.sigmoid(logits)
        else:
            return logits