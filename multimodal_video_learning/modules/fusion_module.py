"""动态多模态融合模块 - 权重生成和融合"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightGeneratorNetwork(nn.Module):
    """轻量级权重生成网络 - 根据内容动态计算模态权重"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        input_dim = config['text']['output_dim']
        hidden_dim = config['fusion']['weight_generator_hidden_dim']
        num_modalities = len(config['fusion']['modalities'])
        dropout = config['fusion']['weight_generator_dropout']
        
        # 轻量级MLP网络
        self.weight_generator = nn.Sequential(
            nn.Linear(input_dim * num_modalities, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_modalities),
        )
        
        self.normalization = config['fusion']['use_normalization']
        self.num_modalities = num_modalities
    
    def forward(self, concatenated_features):
        """
        Args:
            concatenated_features: (B, D * num_modalities)
        Returns:
            weights: (B, num_modalities)
        """
        logits = self.weight_generator(concatenated_features)  # (B, num_modalities)
        
        if self.normalization == 'softmax':
            weights = F.softmax(logits, dim=-1)
        elif self.normalization == 'sigmoid':
            weights = torch.sigmoid(logits)
        else:
            weights = logits
        
        return weights


class GatingMechanism(nn.Module):
    """门控机制 - 控制各模态的信息流"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        input_dim = config['text']['output_dim']
        hidden_dim = config['fusion']['weight_generator_hidden_dim']
        num_modalities = len(config['fusion']['modalities'])
        
        # 门控控制
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
                nn.Sigmoid()
            )
            for _ in range(num_modalities)
        ])
    
    def forward(self, features_list):
        """
        Args:
            features_list: List[(B, D)]
        Returns:
            gated_features: List[(B, D)]
        """
        gated_features = []
        for i, feat in enumerate(features_list):
            gated = feat * self.gates[i](feat)
            gated_features.append(gated)
        
        return gated_features


class FeatureFusionModule(nn.Module):
    """特征融合模块 - 整合多个模态的特征"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        input_dim = config['text']['output_dim']
        hidden_dim = config['fusion']['fusion_hidden_dim']
        num_modalities = len(config['fusion']['modalities'])
        
        # 权重生成器
        self.weight_generator = WeightGeneratorNetwork(config)
        
        # 门控机制（可选）
        if config['fusion']['use_gating']:
            self.gating = GatingMechanism(config)
        else:
            self.gating = None
        
        # 融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(input_dim * num_modalities, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, input_dim),
        )
        
        # 剩差连接的投影
        self.residual_proj = nn.Linear(input_dim * num_modalities, input_dim) if num_modalities > 1 else None
        
        self.num_modalities = num_modalities
    
    def forward(self, vision_feat, audio_feat, text_feat):
        """
        Args:
            vision_feat: (B, D) 或 (B, T, D)
            audio_feat: (B, D) 或 (B, T, D)
            text_feat: (B, D) 或 (B, T, D)
        Returns:
            fused_features: (B, D) 或 (B, T, D)
        """
        # 处理时间维度
        has_temporal = vision_feat.dim() == 3
        
        if has_temporal:
            batch_size, time_steps, feat_dim = vision_feat.shape
            
            # 处理音频和文本维度
            if audio_feat.dim() == 2:
                audio_feat = audio_feat.unsqueeze(1).expand(-1, time_steps, -1)
            if text_feat.dim() == 2:
                text_feat = text_feat.unsqueeze(1).expand(-1, time_steps, -1)
            
            # 逐时间步融合
            fused_list = []
            for t in range(time_steps):
                feat_t = self.forward(
                    vision_feat[:, t, :],
                    audio_feat[:, t, :],
                    text_feat[:, t, :]
                )
                fused_list.append(feat_t)
            
            return torch.stack(fused_list, dim=1)  # (B, T, D)
        
        # 将特征连接
        concatenated = torch.cat([vision_feat, audio_feat, text_feat], dim=-1)  # (B, D*3)
        
        # 生成权重
        weights = self.weight_generator(concatenated)  # (B, 3)
        
        # 应用权重
        weighted_vision = vision_feat * weights[:, 0:1]
        weighted_audio = audio_feat * weights[:, 1:2]
        weighted_text = text_feat * weights[:, 2:3]
        
        # 应用门控（如果启用）
        if self.gating is not None:
            gated = self.gating([weighted_vision, weighted_audio, weighted_text])
            weighted_vision, weighted_audio, weighted_text = gated
        
        # 融合加权特征
        weighted_concat = torch.cat([weighted_vision, weighted_audio, weighted_text], dim=-1)  # (B, D*3)
        
        fused = self.fusion_net(weighted_concat)  # (B, D)
        
        # 剩余连接
        if self.residual_proj is not None:
            residual = self.residual_proj(concatenated)
            fused = fused + residual
        
        return fused


class AttentionFusionModule(nn.Module):
    """基于注意力机制的融合模块"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        input_dim = config['text']['output_dim']
        hidden_dim = config['fusion']['fusion_hidden_dim']
        num_modalities = len(config['fusion']['modalities'])
        num_heads = 4
        
        # 多头交叉注意力
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(input_dim * num_modalities, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
        )
        
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        
        self.num_modalities = num_modalities
    
    def forward(self, vision_feat, audio_feat, text_feat):
        """
        Args:
            vision_feat: (B, D)
            audio_feat: (B, D)
            text_feat: (B, D)
        Returns:
            fused_features: (B, D)
        """
        # 拼接所有特征
        features = torch.stack([vision_feat, audio_feat, text_feat], dim=1)  # (B, 3, D)
        
        # 自注意力
        attn_output, _ = self.multihead_attn(features, features, features)
        features = self.norm1(features + attn_output)
        
        # 融合
        batch_size = features.shape[0]
        concat_feat = features.reshape(batch_size, -1)  # (B, D*3)
        fused = self.fusion_net(concat_feat)  # (B, D)
        
        return self.norm2(fused + vision_feat)  # 剩余连接


class DynamicMultimodalFusion(nn.Module):
    """统一的动态多模态融合接口"""
    
    def __init__(self, config, fusion_type='weighted'):
        super().__init__()
        self.config = config
        self.fusion_type = fusion_type
        
        if fusion_type == 'dynamic_weighted':
            self.fusion_module = FeatureFusionModule(config)
        elif fusion_type == 'attention':
            self.fusion_module = AttentionFusionModule(config)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(self, vision_feat, audio_feat, text_feat):
        """
        Args:
            vision_feat: (B, D) 或 (B, T, D)
            audio_feat: (B, D) 或 (B, T, D)
            text_feat: (B, D) 或 (B, T, D)
        Returns:
            fused_features: (B, D) 或 (B, T, D)
        """
        return self.fusion_module(vision_feat, audio_feat, text_feat)
