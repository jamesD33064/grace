# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# ... (Keeping license header)

from __future__ import annotations
from collections.abc import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep

# --- 1. 高速 Channel-First RMSNorm ---
class RMSNormCH(nn.Module):
    """
    針對 (B, C, D, H, W) 優化的 RMSNorm。
    避免 permute 到 (B, D, H, W, C) 再轉回來。
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        # 參數形狀設為 (1, C, 1, 1, 1) 以便廣播
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        # x: (B, C, D, H, W)
        # 在 Channel 維度 (dim=1) 計算 norm
        norm = torch.norm(x, dim=1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.g

# --- 2. Turbo AFT Layer (全卷積實作) ---
class TurboAFTLayer(nn.Module):
    def __init__(self, dim, mode="global"):
        super().__init__()
        self.mode = mode
        
        # 使用 1x1 Conv 代替 Linear，避免 Reshape
        self.conv_q = nn.Conv3d(dim, dim, 1)
        self.conv_k = nn.Conv3d(dim, dim, 1)
        self.conv_v = nn.Conv3d(dim, dim, 1)
        self.proj = nn.Conv3d(dim, dim, 1)
        
        self.act = nn.Sigmoid()

        if mode == "local":
            # Depthwise Conv3d (Groups=dim)
            self.local_mixer = nn.Conv3d(dim, dim, kernel_size=3, padding=1, groups=dim)

    def forward(self, x):
        # x shape: (B, C, D, H, W) - 保持這個形狀，不要變！
        
        # Q 投影
        q = self.conv_q(x)
        q_sig = self.act(q)

        if self.mode == "global":
            # [AFT-Global 優化版]
            # 目標：計算 Global Context = Sum(Softmax(K) * V)
            
            k = self.conv_k(x)
            v = self.conv_v(x)
            
            B, C, D, H, W = k.shape
            N = D * H * W
            
            # 1. 空間維度展平 (B, C, N) - 這是唯一需要的 view，且內存連續
            k_flat = k.view(B, C, N)
            v_flat = v.view(B, C, N)
            
            # 2. 空間 Softmax (沿著 N)
            # 這裡計算每個 Channel 的空間注意力分佈
            attn = F.softmax(k_flat, dim=-1) # (B, C, N)
            
            # 3. 加權求和 (Global Context Aggregation)
            # (B, C, N) * (B, C, N) -> element-wise -> sum over N -> (B, C, 1)
            # 這一步極快，因為沒有矩陣乘法，只有並行的乘加
            global_ctx = torch.sum(attn * v_flat, dim=-1, keepdim=True) # (B, C, 1)
            
            # 4. Reshape 回 (B, C, 1, 1, 1) 以便廣播
            global_ctx = global_ctx.view(B, C, 1, 1, 1)
            
            # 5. Gating
            out = q_sig * global_ctx # Broadcasting 自動處理
            
        elif self.mode == "local":
            # AFT-Local: 只是 Q * DepthwiseConv(X)
            # 在全卷積模式下，我們直接對 x 做卷積作為 V 的替代 (ResNet style)
            local_feat = self.local_mixer(x)
            out = q_sig * local_feat

        # Projection
        out = self.proj(out)
        return out

# --- 3. Turbo AFT Block ---
class TurboAFTBlock(nn.Module):
    def __init__(self, dim, mlp_dim, dropout_rate, mode="global"):
        super().__init__()
        self.norm1 = RMSNormCH(dim)
        self.attn = TurboAFTLayer(dim, mode=mode)
        
        self.norm2 = RMSNormCH(dim)
        self.mlp = nn.Sequential(
            nn.Conv3d(dim, mlp_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Conv3d(mlp_dim, dim, 1),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        # x: (B, C, D, H, W)
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# --- 4. Turbo AFT Encoder (無 Patch Flatten) ---
class TurboAFTEncoder(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, hidden_size, mlp_dim, num_layers, num_heads, dropout_rate, spatial_dims=3):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        
        # Patch Embedding (Strided Conv)
        self.patch_embed = nn.Conv3d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        
        # Absolute Position Embedding
        # 因為不 Flatten，我們需要一個與特徵圖大小匹配的參數
        # 假設輸入大小固定，計算特徵圖尺寸
        feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, patch_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, hidden_size, *feat_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.dropout = nn.Dropout(dropout_rate)

        # Layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            mode = "local" if i < 3 else "global"
            # 注意：不再需要傳入 feat_size 或 num_heads，因為我們做的是 Channel-wise 操作
            layer = TurboAFTBlock(
                dim=hidden_size,
                mlp_dim=mlp_dim,
                dropout_rate=dropout_rate,
                mode=mode
            )
            self.layers.append(layer)
            
        self.norm = RMSNormCH(hidden_size)

    def forward(self, x):
        # x: (B, C_in, D, H, W)
        x = self.patch_embed(x) # -> (B, Hidden, D/P, H/P, W/P)
        
        # Add Pos Embed (Broadcasting handles batch dim)
        if x.shape[2:] == self.pos_embed.shape[2:]:
            x = x + self.pos_embed
        else:
            # 如果輸入尺寸動態變化，使用插值調整 Pos Embed
            pos_embed_resized = F.interpolate(self.pos_embed, size=x.shape[2:], mode='trilinear', align_corners=False)
            x = x + pos_embed_resized
            
        x = self.dropout(x)

        hidden_states_out = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if (i + 1) % 3 == 0:
                hidden_states_out.append(self.norm(x))
        
        return x, hidden_states_out

# --- 5. 主模型 AFTUNET (介面適配) ---
class AFTUNET(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] | int,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12, # 保留參數但不使用 (全卷積自適應)
        norm_name: tuple | str = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
    ) -> None:
        super().__init__()

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        
        # 使用 TurboAFTEncoder
        self.aft_encoder = TurboAFTEncoder(
            img_size=img_size,
            patch_size=self.patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims
        )

        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)

    def forward(self, x_in):
        # Encoder 直接輸出 3D 特徵圖，不需要 reshape
        x, hidden_states_out = self.aft_encoder(x_in)
        
        enc1 = self.encoder1(x_in)
        
        x2 = hidden_states_out[0]
        enc2 = self.encoder2(x2) 
        
        x3 = hidden_states_out[1]
        enc3 = self.encoder3(x3)
        
        x4 = hidden_states_out[2]
        enc4 = self.encoder4(x4)
        
        dec4 = x 
        
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        
        return self.out(out)