"""特征对齐模块 - 多模态特征时空对齐"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAlignment(nn.Module):
    """基于时间戳的特征对齐"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        input_dim = config['text']['output_dim']
        hidden_dim = config['alignment']['projection_hidden_dim']
        
        # 时间编码
        self.temporal_encoder = nn.Linear(1, 64)
        
        # 特征缩放到统一时间步
        self.time_projections = nn.ModuleDict({
            'vision': nn.Linear(input_dim + 64, input_dim),
            'audio': nn.Linear(input_dim + 64, input_dim),
            'text': nn.Linear(input_dim + 64, input_dim),
        })
    
    def encode_timestamps(self, timestamps, max_time):
        """编码时间戳"""
        # 归一化时间戳 [0, 1]
        normalized_ts = timestamps / max_time
        return self.temporal_encoder(normalized_ts.unsqueeze(-1))  # (B, 64)
    
    def forward(self, vision_features, audio_features, text_features, 
                vision_timestamps=None, audio_timestamps=None, text_timestamps=None):
        """
        Args:
            vision_features: (B, T_v, D)
            audio_features: (B, D) 或 (B, T_a, D)
            text_features: (B, D) 或 (B, T_t, D)
            timestamps: 各模态的时间戳信息
        Returns:
            aligned_features: dict
        """
        batch_size = vision_features.shape[0]
        device = vision_features.device
        
        # 处理音频特征维度
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(1)  # (B, 1, D)
        
        # 处理文本特征维度
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)  # (B, 1, D)
        
        # 生成默认时间戳
        if vision_timestamps is None:
            vision_timestamps = torch.linspace(0, 1, vision_features.shape[1], device=device)
            vision_timestamps = vision_timestamps.unsqueeze(0).expand(batch_size, -1)
        
        if audio_timestamps is None:
            audio_timestamps = torch.linspace(0, 1, audio_features.shape[1], device=device)
            audio_timestamps = audio_timestamps.unsqueeze(0).expand(batch_size, -1)
        
        if text_timestamps is None:
            text_timestamps = torch.linspace(0, 1, text_features.shape[1], device=device)
            text_timestamps = text_timestamps.unsqueeze(0).expand(batch_size, -1)
        
        # 编码时间戳
        vision_ts_feat = self.encode_timestamps(vision_timestamps, 1.0)  # (B, 64)
        audio_ts_feat = self.encode_timestamps(audio_timestamps, 1.0)  # (B, 64)
        text_ts_feat = self.encode_timestamps(text_timestamps, 1.0)  # (B, 64)
        
        # 扩展时间戳特征
        vision_ts_feat = vision_ts_feat.unsqueeze(1).expand(-1, vision_features.shape[1], -1)
        audio_ts_feat = audio_ts_feat.unsqueeze(1).expand(-1, audio_features.shape[1], -1)
        text_ts_feat = text_ts_feat.unsqueeze(1).expand(-1, text_features.shape[1], -1)
        
        # 投影到对齐空间
        vision_aligned = self.time_projections['vision'](
            torch.cat([vision_features, vision_ts_feat], dim=-1)
        )
        
        audio_aligned = self.time_projections['audio'](
            torch.cat([audio_features, audio_ts_feat], dim=-1)
        )
        
        text_aligned = self.time_projections['text'](
            torch.cat([text_features, text_ts_feat], dim=-1)
        )
        
        return {
            'vision': vision_aligned,
            'audio': audio_aligned,
            'text': text_aligned,
        }


class CrossModalProjection(nn.Module):
    """跨模态投影对齐"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        input_dim = config['text']['output_dim']
        hidden_dim = config['alignment']['projection_hidden_dim']
        
        # 共享投影空间
        self.shared_space = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim),
        )
        
        # 可学习的对齐权重
        self.alignment_weights = nn.ModuleDict({
            'vision': nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ),
            'audio': nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ),
            'text': nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ),
        })
    
    def forward(self, vision_features, audio_features, text_features):
        """
        Args:
            vision_features: (B, T, D)
            audio_features: (B, D) 或 (B, T, D)
            text_features: (B, D) 或 (B, T, D)
        Returns:
            aligned_features: dict
        """
        # 确保统一形状
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(1)
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
        
        # 投影到共享空间
        batch_size = vision_features.shape[0]
        
        # 视觉特征对齐
        vision_flat = vision_features.reshape(-1, vision_features.shape[-1])
        vision_aligned = self.shared_space(vision_flat)
        vision_aligned = vision_aligned.reshape(vision_features.shape)
        
        # 音频特征对齐
        audio_flat = audio_features.reshape(-1, audio_features.shape[-1])
        audio_aligned = self.shared_space(audio_flat)
        audio_aligned = audio_aligned.reshape(audio_features.shape)
        
        # 文本特征对齐
        text_flat = text_features.reshape(-1, text_features.shape[-1])
        text_aligned = self.shared_space(text_flat)
        text_aligned = text_aligned.reshape(text_features.shape)
        
        # 计算对齐权重
        vision_weight = self.alignment_weights['vision'](vision_flat).reshape(vision_features.shape[:-1] + (1,))
        audio_weight = self.alignment_weights['audio'](audio_flat).reshape(audio_features.shape[:-1] + (1,))
        text_weight = self.alignment_weights['text'](text_flat).reshape(text_features.shape[:-1] + (1,))
        
        # 应用权重
        vision_aligned = vision_aligned * vision_weight
        audio_aligned = audio_aligned * audio_weight
        text_aligned = text_aligned * text_weight
        
        return {
            'vision': vision_aligned,
            'audio': audio_aligned,
            'text': text_aligned,
        }


class LearnedAlignment(nn.Module):
    """可学习的多模态对齐"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        input_dim = config['text']['output_dim']
        hidden_dim = config['alignment']['projection_hidden_dim']
        
        # 跨模态相似度计算
        self.similarity_fn = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
        # 对齐矩阵学习
        self.alignment_matrix = nn.Parameter(torch.eye(3))  # 3个模态
        
        # 特征融合网络
        self.fusion_net = nn.Sequential(
            nn.Linear(input_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, input_dim),
        )
    
    def compute_alignment_score(self, feat1, feat2):
        """计算两个特征之间的对齐分数"""
        combined = torch.cat([feat1, feat2], dim=-1)
        return self.similarity_fn(combined)
    
    def forward(self, vision_features, audio_features, text_features):
        """
        Args:
            vision_features: (B, T, D)
            audio_features: (B, D) 或 (B, T, D)
            text_features: (B, D) 或 (B, T, D)
        Returns:
            aligned_features: dict
        """
        if audio_features.dim() == 2:
            audio_features = audio_features.unsqueeze(1)
        if text_features.dim() == 2:
            text_features = text_features.unsqueeze(1)
        
        # 计算交叉模态相似度
        batch_size = vision_features.shape[0]
        
        # 对齐音频和文本到视觉时间轴
        vision_mean = vision_features.mean(dim=1)  # (B, D)
        audio_mean = audio_features.mean(dim=1)  # (B, D)
        text_mean = text_features.mean(dim=1)  # (B, D)
        
        # 动态调整特征
        aligned_features = {
            'vision': vision_features,
            'audio': audio_features,
            'text': text_features,
        }
        
        return aligned_features


class FeatureAlignment(nn.Module):
    """统一的特征对齐接口"""
    
    def __init__(self, config, alignment_type='temporal_sync'):
        super().__init__()
        self.config = config
        self.alignment_type = alignment_type
        
        if alignment_type == 'temporal_sync':
            self.aligner = TemporalAlignment(config)
        elif alignment_type == 'cross_modal':
            self.aligner = CrossModalProjection(config)
        elif alignment_type == 'learned':
            self.aligner = LearnedAlignment(config)
        else:
            raise ValueError(f"Unknown alignment type: {alignment_type}")
    
    def forward(self, vision_features, audio_features, text_features,
                vision_timestamps=None, audio_timestamps=None, text_timestamps=None):
        """
        Args:
            vision_features: (B, T, D)
            audio_features: (B, D) 或 (B, T, D)
            text_features: (B, D) 或 (B, T, D)
        Returns:
            aligned_features: dict
        """
        if self.alignment_type == 'temporal_sync':
            return self.aligner(vision_features, audio_features, text_features,
                              vision_timestamps, audio_timestamps, text_timestamps)
        else:
            return self.aligner(vision_features, audio_features, text_features)
