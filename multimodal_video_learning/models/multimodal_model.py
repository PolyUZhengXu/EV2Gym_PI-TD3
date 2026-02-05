"""完整的多模态视频理解模型"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import (
    VisionFeatureExtractor,
    AudioFeatureExtractor,
    TextFeatureExtractor,
    FeatureAlignment,
    DynamicMultimodalFusion,
    TemporalSemanticModule
)


class MultimodalVideoUnderstandingModel(nn.Module):
    """完整的多模态视频理解模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 特征提取器
        self.vision_extractor = VisionFeatureExtractor(
            config,
            model_type='vit'
        )
        
        self.audio_extractor = AudioFeatureExtractor(
            config,
            model_type='whisper'
        )
        
        self.text_extractor = TextFeatureExtractor(
            config,
            model_type='bert'
        )
        
        # 特征对齐
        alignment_type = config['alignment']['method']
        self.feature_alignment = FeatureAlignment(config, alignment_type)
        
        # 动态融合
        fusion_type = config['fusion']['type']
        self.dynamic_fusion = DynamicMultimodalFusion(config, fusion_type)
        
        # 时序建模
        temporal_model = config['temporal']['model']
        self.temporal_module = TemporalSemanticModule(config, temporal_model)
        
        # 分类头
        input_dim = config['text']['output_dim']
        num_classes = config['dataset']['num_classes']
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # 损失函数
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=config['training']['label_smoothing']
        )
    
    def forward(self, frames, audio_waveform, texts, labels=None):
        """
        Args:
            frames: (B, T, C, H, W) 视频帧
            audio_waveform: (B, audio_length) 音频波形
            texts: List[str] 字幕或转录文本
            labels: (B,) 类标签
        Returns:
            outputs: dict 包含logits和loss
        """
        # 提取各模态特征
        vision_features = self.vision_extractor(frames)  # (B, T, D)
        # Detach audio before passing to extractor to avoid requires_grad issues in Whisper processor
        audio_input = audio_waveform.detach() if isinstance(audio_waveform, torch.Tensor) and audio_waveform.requires_grad else audio_waveform
        audio_features = self.audio_extractor(audio_input)  # (B, D)
        text_features = self.text_extractor(texts)  # (B, D)
        
        # 特征对齐
        aligned_features = self.feature_alignment(
            vision_features, audio_features, text_features
        )
        
        # 融合（逐时间步）
        batch_size = frames.shape[0]
        time_steps = frames.shape[1]
        
        fused_features = []
        for t in range(time_steps):
            v_feat = aligned_features['vision'][:, t, :] if aligned_features['vision'].dim() == 3 else aligned_features['vision']
            a_feat = aligned_features['audio'][:, t, :] if aligned_features['audio'].dim() == 3 else aligned_features['audio']
            t_feat = aligned_features['text'][:, t, :] if aligned_features['text'].dim() == 3 else aligned_features['text']
            
            fused = self.dynamic_fusion(v_feat, a_feat, t_feat)
            fused_features.append(fused)
        
        fused_sequence = torch.stack(fused_features, dim=1)  # (B, T, D)
        
        # 时序建模
        temporal_output = self.temporal_module(fused_sequence)  # (B, T, D)
        
        # 取最后一个时间步或进行平均池化
        pooled_output = torch.mean(temporal_output, dim=1)  # (B, D)
        
        # 分类
        logits = self.classifier(pooled_output)  # (B, num_classes)
        
        outputs = {
            'logits': logits,
            'features': pooled_output,
            'temporal_features': temporal_output
        }
        
        # 计算损失
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            outputs['loss'] = loss
        
        return outputs
    
    def extract_features(self, frames, audio_waveform, texts):
        """提取融合后的特征表示"""
        # 提取各模态特征
        vision_features = self.vision_extractor(frames)
        # Detach audio before passing to extractor to avoid requires_grad issues in Whisper processor
        audio_input = audio_waveform.detach() if isinstance(audio_waveform, torch.Tensor) and audio_waveform.requires_grad else audio_waveform
        audio_features = self.audio_extractor(audio_input)
        text_features = self.text_extractor(texts)
        
        # 特征对齐
        aligned_features = self.feature_alignment(
            vision_features, audio_features, text_features
        )
        
        # 融合
        batch_size = frames.shape[0]
        time_steps = frames.shape[1]
        
        fused_features = []
        for t in range(time_steps):
            v_feat = aligned_features['vision'][:, t, :] if aligned_features['vision'].dim() == 3 else aligned_features['vision']
            a_feat = aligned_features['audio'][:, t, :] if aligned_features['audio'].dim() == 3 else aligned_features['audio']
            t_feat = aligned_features['text'][:, t, :] if aligned_features['text'].dim() == 3 else aligned_features['text']
            
            fused = self.dynamic_fusion(v_feat, a_feat, t_feat)
            fused_features.append(fused)
        
        fused_sequence = torch.stack(fused_features, dim=1)
        
        # 时序建模
        temporal_output = self.temporal_module(fused_sequence)
        
        return temporal_output
    
    def freeze_extractors(self):
        """冻结特征提取器"""
        self.vision_extractor.freeze_backbone()
        # 音频和文本提取器已在模块内冻结
    
    def unfreeze_extractors(self):
        """解冻特征提取器"""
        self.vision_extractor.unfreeze_backbone()
    
    def get_learnable_params(self):
        """获取可训练的参数"""
        return filter(lambda p: p.requires_grad, self.parameters())


class MultimodalVideoClassifier(nn.Module):
    """多模态视频分类器 - 简化版"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 特征提取器
        self.vision_extractor = VisionFeatureExtractor(config, model_type='vit')
        self.audio_extractor = AudioFeatureExtractor(config, model_type='whisper')
        self.text_extractor = TextFeatureExtractor(config, model_type='bert')
        
        # 融合模块
        input_dim = config['text']['output_dim']
        
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # 时序建模
        self.temporal = nn.LSTM(
            input_size=256,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # 分类头
        num_classes = config['dataset']['num_classes']
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=config['training']['label_smoothing']
        )
    
    def forward(self, frames, audio_waveform, texts, labels=None):
        """
        Args:
            frames: (B, T, C, H, W)
            audio_waveform: (B, audio_length)
            texts: List[str]
            labels: (B,)
        Returns:
            outputs: dict
        """
        # Extract features
        vision_feat = self.vision_extractor(frames)  # (B, T, D)
        audio_feat = self.audio_extractor(audio_waveform).unsqueeze(1)  # (B, 1, D)
        text_feat = self.text_extractor(texts).unsqueeze(1)  # (B, 1, D)

        # Debug: print shapes to diagnose broadcasting/shape issues
        try:
            print(f"[DEBUG] vision_feat.shape: {vision_feat.shape}")
            print(f"[DEBUG] audio_feat.shape (before expand): {audio_feat.shape}")
            print(f"[DEBUG] text_feat.shape (before expand): {text_feat.shape}")
        except Exception:
            print("[DEBUG] Failed to print initial feature shapes")
        
        # 处理维度
        batch_size, time_steps, feat_dim = vision_feat.shape
        
        # Fusion (expand audio and text to time dimension)
        try:
            audio_feat = audio_feat.expand(-1, time_steps, -1)
            text_feat = text_feat.expand(-1, time_steps, -1)
        except Exception as e:
            print(f"[DEBUG] Error during expand: {e}")
            print(f"[DEBUG] shapes before expand - vision: {vision_feat.shape}, audio: {audio_feat.shape}, text: {text_feat.shape}")
            raise

        # Debug: shapes after expand
        try:
            print(f"[DEBUG] audio_feat.shape (after expand): {audio_feat.shape}")
            print(f"[DEBUG] text_feat.shape (after expand): {text_feat.shape}")
        except Exception:
            print("[DEBUG] Failed to print shapes after expand")
        
        print(f"[DEBUG] shapes before cat: vision={vision_feat.shape}, audio={audio_feat.shape}, text={text_feat.shape}")
        combined = torch.cat([vision_feat, audio_feat, text_feat], dim=-1)  # (B, T, D*3)
        fused = self.fusion(combined.view(-1, feat_dim * 3))  # (B*T, 256)
        fused = fused.view(batch_size, time_steps, -1)  # (B, T, 256)
        
        # 时序建模
        lstm_out, (h_n, c_n) = self.temporal(fused)  # (B, T, 512)
        
        # 平均池化
        pooled = torch.mean(lstm_out, dim=1)  # (B, 512)
        
        # 分类
        logits = self.classifier(pooled)  # (B, num_classes)
        
        outputs = {
            'logits': logits,
            'features': pooled
        }
        
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            outputs['loss'] = loss
        
        return outputs
