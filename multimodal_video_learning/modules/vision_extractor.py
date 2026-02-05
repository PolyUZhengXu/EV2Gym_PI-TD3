"""视觉特征提取器 - Vision Transformer and 3D CNN"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import torchvision.models as models
from einops import rearrange


class VisionTransformerExtractor(nn.Module):
    """基于Vision Transformer的视觉特征提取器"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config['vision']['output_dim']
        
        # 使用轻量级的ViT模型
        model_name = "google/vit-base-patch16-224"
        self.vit = AutoModel.from_pretrained(model_name)
        
        # 冻结预训练层以节省内存
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # 适配器层用于时空特征融合
        self.temporal_adapter = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.output_dim)
        )
        
        self.frame_reduce = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, frames):
        """
        Args:
            frames: (B, T, C, H, W) 视频帧序列
        Returns:
            features: (B, T, output_dim) 时空特征
        """
        batch_size, num_frames, c, h, w = frames.shape
        
        # 重塑为 (B*T, C, H, W)
        frames_flat = rearrange(frames, 'b t c h w -> (b t) c h w')
        
        # 使用ViT提取特征
        outputs = self.vit(frames_flat, output_hidden_states=True)
        # CLS token特征: (B*T, 768)
        cls_features = outputs.last_hidden_state[:, 0, :]
        
        # 应用适配器层
        features = self.temporal_adapter(cls_features)  # (B*T, output_dim)
        
        # 重塑回 (B, T, output_dim)
        features = rearrange(features, '(b t) d -> b t d', b=batch_size, t=num_frames)
        
        return features


class R3DExtractor(nn.Module):
    """基于3D CNN (R3D)的视觉特征提取器"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config['vision']['output_dim']
        
        # 使用预训练的3D ResNet
        self.r3d = models.video.r3d_18(pretrained=True)
        
        # 冻结预训练层
        for param in self.r3d.parameters():
            param.requires_grad = False
        
        # 替换分类头为特征输出
        in_features = self.r3d.fc.in_features
        self.r3d.fc = nn.Identity()
        
        self.projection = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.output_dim)
        )
        
    def forward(self, frames):
        """
        Args:
            frames: (B, T, C, H, W) 或 (B, C, T, H, W)
        Returns:
            features: (B, output_dim)
        """
        # 转换为 (B, C, T, H, W) 格式
        if frames.shape[1] != 3:
            frames = rearrange(frames, 'b t c h w -> b c t h w')
        
        # 使用3D CNN提取特征
        features = self.r3d(frames)  # (B, in_features)
        
        # 投影到输出维度
        features = self.projection(features)  # (B, output_dim)
        
        return features.unsqueeze(1)  # (B, 1, output_dim)


class MultiScaleVisionExtractor(nn.Module):
    """多尺度视觉特征提取器"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config['vision']['output_dim']
        
        # 多个尺度的特征提取器
        self.vit_extractor = VisionTransformerExtractor(config)
        
        # 多尺度投影
        self.scale_fusion = nn.Sequential(
            nn.Linear(self.output_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.output_dim)
        )
        
    def forward(self, frames):
        """
        Args:
            frames: (B, T, C, H, W)
        Returns:
            features: (B, T, output_dim)
        """
        # 提取ViT特征
        vit_features = self.vit_extractor(frames)  # (B, T, output_dim)
        
        # 多尺度融合
        fused_features = self.scale_fusion(vit_features)  # (B, T, output_dim)
        
        return fused_features


class VisionFeatureExtractor(nn.Module):
    """统一的视觉特征提取器接口"""
    
    def __init__(self, config, model_type='vit'):
        super().__init__()
        self.config = config
        self.model_type = model_type
        
        if model_type == 'vit':
            self.extractor = VisionTransformerExtractor(config)
        elif model_type == 'r3d':
            self.extractor = R3DExtractor(config)
        elif model_type == 'multi_scale':
            self.extractor = MultiScaleVisionExtractor(config)
        else:
            raise ValueError(f"Unknown vision model type: {model_type}")
    
    def forward(self, frames):
        """
        Args:
            frames: (B, T, C, H, W) 视频帧序列
        Returns:
            features: (B, T, output_dim) 或 (B, output_dim)
        """
        return self.extractor(frames)
    
    def freeze_backbone(self):
        """冻结主干网络"""
        for param in self.extractor.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """解冻主干网络"""
        for param in self.extractor.parameters():
            param.requires_grad = True
