"""时序语义建模模块 - Transformer和LSTM"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_length, d_model)
        """
        return x + self.pe[:, :x.size(1), :].to(x.device)


class TransformerTemporalModule(nn.Module):
    """基于Transformer的时序建模"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        input_dim = config['text']['output_dim']
        hidden_dim = config['temporal']['hidden_dim']
        num_heads = config['temporal']['num_heads']
        num_layers = config['temporal']['num_layers']
        dropout = config['temporal']['attention_dropout']
        
        # 位置编码
        self.positional_encoding = PositionalEncoding(input_dim, max_len=1000)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=config['temporal']['ff_hidden_dim'],
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, T, D) 融合后的特征序列
            mask: (B, T) 注意力掩码
        Returns:
            output: (B, T, D)
        """
        # 添加位置编码
        x = self.positional_encoding(x)
        
        # 转换掩码
        if mask is not None:
            # mask: (B, T) -> (B, 1, 1, T) 用于MultiheadAttention
            mask = mask.unsqueeze(1).unsqueeze(1)
        
        # Transformer处理
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # 输出投影
        output = self.output_proj(output)
        
        return output


class LSTMTemporalModule(nn.Module):
    """基于LSTM的时序建模"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        input_dim = config['text']['output_dim']
        lstm_hidden_dim = config['temporal']['lstm_hidden_dim']
        num_layers = config['temporal']['num_layers']
        bidirectional = config['temporal']['bidirectional']
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        lstm_output_dim = lstm_hidden_dim * (2 if bidirectional else 1)
        
        # 输出投影回原维度
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_output_dim, config['temporal']['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config['temporal']['hidden_dim'], input_dim)
        )
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: (B, T, D)
            lengths: (B,) 序列长度
        Returns:
            output: (B, T, D)
        """
        # 打包序列（如果提供了长度）
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, (h_n, c_n) = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        else:
            output, (h_n, c_n) = self.lstm(x)
        
        # 投影到输出维度
        output = self.output_proj(output)
        
        return output


class TransformerLSTMModule(nn.Module):
    """混合Transformer-LSTM时序建模"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        input_dim = config['text']['output_dim']
        
        # Transformer层用于全局上下文
        self.transformer = TransformerTemporalModule(config)
        
        # LSTM层用于局部时序依赖
        lstm_config = config.copy()
        lstm_config['temporal']['num_layers'] = 1
        self.lstm = LSTMTemporalModule(lstm_config)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(input_dim * 2, config['temporal']['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config['temporal']['hidden_dim'], input_dim)
        )
    
    def forward(self, x, mask=None, lengths=None):
        """
        Args:
            x: (B, T, D)
            mask: (B, T)
            lengths: (B,)
        Returns:
            output: (B, T, D)
        """
        # Transformer处理
        trans_out = self.transformer(x, mask)
        
        # LSTM处理
        lstm_out = self.lstm(x, lengths)
        
        # 融合
        combined = torch.cat([trans_out, lstm_out], dim=-1)  # (B, T, D*2)
        output = self.fusion(combined)  # (B, T, D)
        
        return output


class AttentionAugmentedLSTM(nn.Module):
    """注意力增强的LSTM"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        input_dim = config['text']['output_dim']
        hidden_dim = config['temporal']['lstm_hidden_dim']
        
        self.lstm = nn.LSTMCell(input_dim, hidden_dim)
        self.num_layers = config['temporal']['num_layers']
        
        # 自注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, config['temporal']['hidden_dim']),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config['temporal']['hidden_dim'], input_dim)
        )
    
    def forward(self, x, lengths=None):
        """
        Args:
            x: (B, T, D)
            lengths: (B,)
        Returns:
            output: (B, T, D)
        """
        batch_size, seq_len, input_dim = x.shape
        hidden_dim = self.config['temporal']['lstm_hidden_dim']
        
        h = torch.zeros(batch_size, hidden_dim, device=x.device)
        c = torch.zeros(batch_size, hidden_dim, device=x.device)
        
        outputs = []
        attention_states = []
        
        for t in range(seq_len):
            h, c = self.lstm(x[:, t, :], (h, c))
            
            # 应用注意力到之前的隐藏状态
            if len(attention_states) > 0:
                prev_states = torch.stack(attention_states, dim=1)  # (B, T-1, hidden_dim)
                attn_output, _ = self.attention(
                    h.unsqueeze(1),
                    prev_states,
                    prev_states
                )
                h = h + attn_output.squeeze(1)
            
            attention_states.append(h)
            outputs.append(h)
        
        # 堆栈输出
        output = torch.stack(outputs, dim=1)  # (B, T, hidden_dim)
        
        # 投影到原维度
        output = self.output_proj(output)
        
        return output


class TemporalSemanticModule(nn.Module):
    """统一的时序语义建模接口"""
    
    def __init__(self, config, temporal_model='transformer_lstm'):
        super().__init__()
        self.config = config
        self.temporal_model = temporal_model
        
        if temporal_model == 'transformer':
            self.model = TransformerTemporalModule(config)
        elif temporal_model == 'lstm':
            self.model = LSTMTemporalModule(config)
        elif temporal_model == 'transformer_lstm':
            self.model = TransformerLSTMModule(config)
        elif temporal_model == 'attention_lstm':
            self.model = AttentionAugmentedLSTM(config)
        else:
            raise ValueError(f"Unknown temporal model: {temporal_model}")
    
    def forward(self, x, mask=None, lengths=None):
        """
        Args:
            x: (B, T, D) 融合后的特征序列
            mask: (B, T) 可选的注意力掩码
            lengths: (B,) 可选的序列长度
        Returns:
            output: (B, T, D)
        """
        if self.temporal_model == 'transformer_lstm':
            return self.model(x, mask, lengths)
        elif self.temporal_model == 'attention_lstm':
            return self.model(x, lengths)
        else:
            return self.model(x, mask)
