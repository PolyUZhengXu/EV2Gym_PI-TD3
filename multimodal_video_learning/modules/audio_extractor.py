"""音频特征提取器 - Whisper和MFCC"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np
from transformers import WhisperProcessor, WhisperModel
import torch.fft as fft


class WhisperAudioExtractor(nn.Module):
    """基于Whisper的音频特征提取器"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config['audio']['output_dim']
        
        # 使用轻量级的Whisper-base模型
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.whisper = WhisperModel.from_pretrained("openai/whisper-base")
        
        # 冻结预训练层以节省内存和计算
        for param in self.whisper.parameters():
            param.requires_grad = config['audio'].get('freeze_pretrained', True)
        
        # 获取Whisper的输出维度
        whisper_output_dim = 512
        
        # 投影层将Whisper输出投影到目标维度
        self.projection = nn.Sequential(
            nn.Linear(whisper_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.output_dim)
        )
        
        self.sr = config['audio']['sr']
        
    def forward(self, audio_waveform, sample_rate=None):
        """
        Args:
            audio_waveform: (B, audio_length) 或 (B, T, audio_length)
            sample_rate: 采样率
        Returns:
            features: (B, output_dim) 或 (B, T, output_dim)
        """
        if sample_rate is None:
            sample_rate = self.sr
        
        # Minimal robust forward implementation for debugging.
        # If a proper Whisper-based pipeline is desired, replace this
        # with the processor -> whisper -> projection logic.
        if isinstance(audio_waveform, torch.Tensor):
            batch_size = audio_waveform.shape[0]
            device = audio_waveform.device
        elif isinstance(audio_waveform, list):
            batch_size = len(audio_waveform)
            device = torch.device('cpu')
        else:
            # Fallback: treat as single example
            batch_size = 1
            device = torch.device('cpu')

        # Return a zero tensor with the expected output dimension so
        # downstream fusion code can run during debugging.
        features = torch.zeros(batch_size, self.output_dim, device=device, dtype=torch.float32)
        return features


class MFCCAudioExtractor(nn.Module):
    """基于MFCC的音频特征提取器"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config['audio']['output_dim']
        
        self.n_mfcc = config['audio']['n_mfcc']
        self.n_fft = config['audio']['n_fft']
        self.hop_length = config['audio']['hop_length']
        self.sr = config['audio']['sr']
        
        # MFCC特征处理网络
        mfcc_input_dim = self.n_mfcc
        
        self.mfcc_net = nn.Sequential(
            nn.Conv1d(mfcc_input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.1),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        
        self.projection = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.output_dim)
        )
    
    def extract_mfcc(self, audio_waveform):
        """提取MFCC特征"""
        if isinstance(audio_waveform, torch.Tensor):
            audio_np = audio_waveform.detach().cpu().numpy()
        else:
            audio_np = audio_waveform
        
        # 使用librosa提取MFCC
        mfccs = librosa.feature.mfcc(
            y=audio_np,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        return torch.from_numpy(mfccs).float()
    
    def forward(self, audio_waveform):
        """
        Args:
            audio_waveform: (B, audio_length) 或 np.array
        Returns:
            features: (B, output_dim)
        """
        batch_size = 1 if not isinstance(audio_waveform, list) else len(audio_waveform)
        
        # 提取MFCC特征
        if isinstance(audio_waveform, list):
            mfcc_features = []
            for audio in audio_waveform:
                mfcc = self.extract_mfcc(audio)  # (n_mfcc, time)
                mfcc_features.append(mfcc)
            mfcc_features = torch.stack(mfcc_features)  # (B, n_mfcc, time)
        else:
            mfcc_features = self.extract_mfcc(audio_waveform)  # (n_mfcc, time)
            mfcc_features = mfcc_features.unsqueeze(0)  # (1, n_mfcc, time)
        
        # 处理MFCC特征
        mfcc_features = mfcc_features.to(next(self.parameters()).device)
        net_features = self.mfcc_net(mfcc_features)  # (B, 256, 1)
        net_features = net_features.squeeze(-1)  # (B, 256)
        
        # 投影到目标维度
        features = self.projection(net_features)  # (B, output_dim)
        
        return features


class HybridAudioExtractor(nn.Module):
    """混合音频特征提取器 - 结合Whisper和MFCC"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config['audio']['output_dim']
        
        self.whisper_extractor = WhisperAudioExtractor(config)
        self.mfcc_extractor = MFCCAudioExtractor(config)
        
        # 特征融合层
        self.fusion = nn.Sequential(
            nn.Linear(self.output_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.output_dim)
        )
    
    def forward(self, audio_waveform, sample_rate=None):
        """
        Args:
            audio_waveform: 音频波形数据
            sample_rate: 采样率
        Returns:
            features: (B, output_dim)
        """
        # 提取Whisper特征
        whisper_features = self.whisper_extractor(audio_waveform, sample_rate)
        
        # 提取MFCC特征
        mfcc_features = self.mfcc_extractor(audio_waveform)
        
        # 融合两种特征
        combined = torch.cat([whisper_features, mfcc_features], dim=-1)
        fused = self.fusion(combined)
        
        return fused


class AudioFeatureExtractor(nn.Module):
    """统一的音频特征提取器接口"""
    
    def __init__(self, config, model_type='whisper'):
        super().__init__()
        self.config = config
        self.model_type = model_type
        
        if model_type == 'whisper':
            self.extractor = WhisperAudioExtractor(config)
        elif model_type == 'mfcc':
            self.extractor = MFCCAudioExtractor(config)
        elif model_type == 'hybrid':
            self.extractor = HybridAudioExtractor(config)
        else:
            raise ValueError(f"Unknown audio model type: {model_type}")
    
    def forward(self, audio_waveform, sample_rate=None):
        """
        Args:
            audio_waveform: 音频波形数据
            sample_rate: 采样率
        Returns:
            features: (B, output_dim)
        """
        return self.extractor(audio_waveform, sample_rate)
