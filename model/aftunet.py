# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# ... (Keeping license header)

from __future__ import annotations
from collections.abc import Sequence
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.utils import ensure_tuple_rep

# --- 1. 穩定性組件: RMSNorm ---
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    比 LayerNorm 更穩定且計算效率更高，有助於 AFT 訓練收斂。
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / (norm + self.eps) * self.g

# --- 2. 3D 相對位置偏置 ---
class RelativePositionBias3D(nn.Module):
    """
    計算 3D 體積數據的相對位置偏置。
    """
    def __init__(self, num_heads, h, w, d):
        super().__init__()
        self.num_heads = num_heads
        # 定義相對距離表 (Table)
        self.rel_pos_table = nn.Parameter(
            torch.zeros((2 * h - 1) * (2 * w - 1) * (2 * d - 1), num_heads)
        )
        nn.init.trunc_normal_(self.rel_pos_table, std=0.02)
        
        # 建立相對座標索引
        coords_h = torch.arange(h)
        coords_w = torch.arange(w)
        coords_d = torch.arange(d)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w, coords_d], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        
        relative_coords[:, :, 0] += h - 1
        relative_coords[:, :, 1] += w - 1
        relative_coords[:, :, 2] += d - 1
        
        relative_coords[:, :, 0] *= (2 * w - 1) * (2 * d - 1)
        relative_coords[:, :, 1] *= (2 * d - 1)
        
        self.relative_position_index = relative_coords.sum(-1) # (N, N)

    def forward(self):
        N = self.relative_position_index.size(0)
        # 取出 bias 並 reshape 為 (1, Heads, N, N)
        rel_pos_bias = self.rel_pos_table[self.relative_position_index.view(-1)].view(
            N, N, -1
        )  # (N, N, nH)
        return rel_pos_bias.permute(2, 0, 1).unsqueeze(0) # (1, nH, N, N)

# --- 3. AFT 模塊 (Global & Local) ---
class AFTLayer(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        feat_size,
        mode="global", # 'global' or 'local'
        qkv_bias=False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.mode = mode
        self.scale = self.head_dim ** -0.5
        
        self.h, self.w, self.d = feat_size

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        
        self.proj = nn.Linear(dim, dim)

        if mode == "global":
            self.act = nn.Sigmoid()
            self.rel_pos = RelativePositionBias3D(num_heads, self.h, self.w, self.d)
        
        elif mode == "local":
            self.act = nn.Sigmoid()
            self.local_mixer = nn.Conv3d(
                dim, dim, kernel_size=3, padding=1, groups=dim, bias=True
            )

    def forward(self, x):
        B, N, C = x.shape
        
        q = self.to_q(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.to_k(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.to_v(x).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Sigmoid on Q
        q_sig = self.act(q)

        if self.mode == "global":
            # pos_bias shape: (1, nH, N, N)
            pos_bias = self.rel_pos().to(k.device)
            
            # [修正邏輯開始]
            # 目標: 計算 sum_j ( softmax(K_j + w_{ij}) * V_j )
            # 我們需要構造 (B, nH, N_i, N_j, d) 的張量來進行計算
            
            # 1. 擴展 K 以廣播到 "目標位置 i" (dim 2)
            # K shape: (B, nH, N, d) -> (B, nH, 1, N, d)
            k_expand = k.unsqueeze(2)
            
            # 2. 擴展 pos_bias 以廣播到 "特徵維度 d" (dim 4)
            # pos_bias shape: (1, nH, N, N) -> (1, nH, N, N, 1)
            pos_bias_expand = pos_bias.unsqueeze(-1)
            
            # 3. 相加 (Broadcasting 會自動處理 B 和 1 的對齊)
            # k_with_bias shape: (B, nH, N, N, d)
            k_with_bias = k_expand + pos_bias_expand
            
            # 4. 在 "來源位置 j" (dim 3) 進行 Softmax
            attn = F.softmax(k_with_bias, dim=3)
            
            # 5. 擴展 V 以便相乘
            # V shape: (B, nH, N, d) -> (B, nH, 1, N, d)
            v_expand = v.unsqueeze(2)
            
            # 6. 加權求和: 沿著 j (dim 3) 加總
            # (B, nH, N, N, d) * (B, nH, 1, N, d) -> Sum dim 3 -> (B, nH, N, d)
            global_ctx = torch.sum(attn * v_expand, dim=3)
            # [修正邏輯結束]

            out = q_sig * global_ctx
            
        elif self.mode == "local":
            # AFT-Local: Depthwise Conv
            v_img = v.reshape(B * self.num_heads, self.d, self.h, self.w, self.head_dim).permute(0, 4, 1, 2, 3)
            
            # 用 Conv3d 處理原始 x 或 v
            # 這裡為了簡單且高效，我們對 x 做 mixer (如同一開始的設計)
            x_img = x.permute(0, 2, 1).view(B, C, self.d, self.h, self.w)
            local_ctx = self.local_mixer(x_img).view(B, C, N).permute(0, 2, 1)
            local_ctx = local_ctx.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
            
            out = q_sig * local_ctx

        out = out.permute(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)
        return out

class AFTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout_rate, feat_size, mode="global"):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = AFTLayer(dim, num_heads, feat_size, mode=mode)
        self.norm2 = RMSNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

# --- 4. AFT Encoder (取代 ViT) ---
class AFTEncoder(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, hidden_size, mlp_dim, num_layers, num_heads, dropout_rate, spatial_dims=3):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        
        # 計算特徵圖大小 (D/P, H/P, W/P)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, patch_size))
        n_patches = int(torch.tensor(self.feat_size).prod())

        # Patch Embedding
        self.patch_embed = nn.Conv3d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.dropout = nn.Dropout(dropout_rate)

        # AFT Layers
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # 2.1 淺層 (前 3 層) 使用 Local，深層使用 Global
            mode = "local" if i < 3 else "global"
            layer = AFTBlock(
                dim=hidden_size,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout_rate=dropout_rate,
                feat_size=self.feat_size,
                mode=mode
            )
            self.layers.append(layer)
            
        self.norm = RMSNorm(hidden_size)

    def forward(self, x):
        # Patch Embedding
        x = self.patch_embed(x) # (B, C, D, H, W)
        x = x.flatten(2).transpose(1, 2) # (B, N, C)
        x = x + self.pos_embed
        x = self.dropout(x)

        hidden_states_out = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # 收集第 3, 6, 9, 12 層的輸出 (index 2, 5, 8, 11)
            if (i + 1) % 3 == 0:
                hidden_states_out.append(self.norm(x)) # Apply norm before returning for skip connections
        
        return x, hidden_states_out

# --- 5. 主模型 AFTUNET ---
class AFTUNET(nn.Module):
    """
    Refactored UNETR using AFT (Attention Free Transformer) Encoder.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Sequence[int] | int,
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
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
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size

        # [修改] 替換 ViT 為 AFTEncoder
        self.aft_encoder = AFTEncoder(
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

        # Encoder 1 (直接對輸入處理)
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        
        # Skip Connections (Projecting from Transformer Hidden Size)
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
        
        # Decoders
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
        
        # Projection helper attributes
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, x_in):
        # 使用 AFT Encoder 獲取特徵
        x, hidden_states_out = self.aft_encoder(x_in)
        
        enc1 = self.encoder1(x_in)
        
        # 提取各層特徵 (Indices 對應 AFT 輸出列表的順序)
        # hidden_states_out[0] -> Layer 3
        # hidden_states_out[1] -> Layer 6
        # hidden_states_out[2] -> Layer 9
        # hidden_states_out[3] -> Layer 12 (x)
        
        x2 = hidden_states_out[0]
        enc2 = self.encoder2(self.proj_feat(x2))
        
        x3 = hidden_states_out[1]
        enc3 = self.encoder3(self.proj_feat(x3))
        
        x4 = hidden_states_out[2]
        enc4 = self.encoder4(self.proj_feat(x4))
        
        dec4 = self.proj_feat(x) # Bottleneck
        
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        
        return self.out(out)