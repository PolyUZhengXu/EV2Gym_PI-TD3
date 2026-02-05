"""文本特征提取器 - BERT和语言模型"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, BertModel, RobertaModel
import numpy as np


class BERTTextExtractor(nn.Module):
    """基于BERT的文本特征提取器"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config['text']['output_dim']
        
        # 使用预训练的BERT模型
        model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # 冻结预训练参数以节省内存
        for param in self.bert.parameters():
            param.requires_grad = False
        
        bert_output_dim = 768
        
        # 投影层
        self.projection = nn.Sequential(
            nn.Linear(bert_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.output_dim)
        )
        
        self.max_seq_length = config['text']['max_seq_length']
    
    def forward(self, texts):
        """
        Args:
            texts: List[str] 文本列表或单个文本字符串
        Returns:
            features: (B, output_dim)
        """
        # 将单个文本转换为列表
        if isinstance(texts, str):
            texts = [texts]
        
        # 分词并编码
        encoded = self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 移至相同设备
        for key in encoded:
            encoded[key] = encoded[key].to(next(self.parameters()).device)
        
        # 获取BERT输出
        with torch.no_grad():
            outputs = self.bert(**encoded)
        
        # 使用[CLS]令牌的输出作为句子表示
        cls_output = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        
        # 投影到目标维度
        features = self.projection(cls_output)  # (B, output_dim)
        
        return features


class RoBERTaTextExtractor(nn.Module):
    """基于RoBERTa的文本特征提取器"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config['text']['output_dim']
        
        # 使用RoBERTa模型
        model_name = "roberta-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.roberta = AutoModel.from_pretrained(model_name)
        
        # 冻结预训练参数
        for param in self.roberta.parameters():
            param.requires_grad = False
        
        roberta_output_dim = 768
        
        self.projection = nn.Sequential(
            nn.Linear(roberta_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.output_dim)
        )
        
        self.max_seq_length = config['text']['max_seq_length']
    
    def forward(self, texts):
        """
        Args:
            texts: List[str]
        Returns:
            features: (B, output_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        encoded = self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        for key in encoded:
            encoded[key] = encoded[key].to(next(self.parameters()).device)
        
        with torch.no_grad():
            outputs = self.roberta(**encoded)
        
        cls_output = outputs.last_hidden_state[:, 0, :]
        features = self.projection(cls_output)
        
        return features


class MultiHeadTextExtractor(nn.Module):
    """多头文本特征提取器 - 从多个隐藏层聚合特征"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config['text']['output_dim']
        
        model_name = "bert-base-uncased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # 使用多个隐藏层的特征
        self.num_layers = 4
        bert_output_dim = 768
        
        # 每层的投影
        self.layer_projections = nn.ModuleList([
            nn.Linear(bert_output_dim, 256)
            for _ in range(self.num_layers)
        ])
        
        self.fusion = nn.Sequential(
            nn.Linear(256 * self.num_layers, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, self.output_dim)
        )
        
        self.max_seq_length = config['text']['max_seq_length']
    
    def forward(self, texts):
        """
        Args:
            texts: List[str]
        Returns:
            features: (B, output_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        encoded = self.tokenizer(
            texts,
            max_length=self.max_seq_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        for key in encoded:
            encoded[key] = encoded[key].to(next(self.parameters()).device)
        
        with torch.no_grad():
            outputs = self.bert(**encoded)
        
        # 从最后4层提取[CLS]特征
        hidden_states = outputs.hidden_states
        layer_features = []
        
        for i in range(self.num_layers):
            layer_idx = -(self.num_layers - i)
            cls = hidden_states[layer_idx][:, 0, :]  # (B, 768)
            projected = self.layer_projections[i](cls)  # (B, 256)
            layer_features.append(projected)
        
        # 拼接所有层的特征
        combined = torch.cat(layer_features, dim=-1)  # (B, 256*num_layers)
        
        # 融合
        features = self.fusion(combined)  # (B, output_dim)
        
        return features


class TextFeatureExtractor(nn.Module):
    """统一的文本特征提取器接口"""
    
    def __init__(self, config, model_type='bert'):
        super().__init__()
        self.config = config
        self.model_type = model_type
        
        if model_type == 'bert':
            self.extractor = BERTTextExtractor(config)
        elif model_type == 'roberta':
            self.extractor = RoBERTaTextExtractor(config)
        elif model_type == 'multi_head':
            self.extractor = MultiHeadTextExtractor(config)
        else:
            raise ValueError(f"Unknown text model type: {model_type}")
    
    def forward(self, texts):
        """
        Args:
            texts: List[str] 或 str
        Returns:
            features: (B, output_dim)
        """
        return self.extractor(texts)
    
    def batch_forward(self, texts_list):
        """
        处理多个文本批次
        Args:
            texts_list: List[List[str]]
        Returns:
            features_list: List[(B, output_dim)]
        """
        features_list = []
        for texts in texts_list:
            features = self.forward(texts)
            features_list.append(features)
        return features_list
