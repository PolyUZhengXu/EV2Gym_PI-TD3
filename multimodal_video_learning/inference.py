"""推理和评估脚本"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import argparse
import os
import logging
from pathlib import Path
from tqdm import tqdm
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from models.multimodal_model import MultimodalVideoClassifier
from data.dataset_loader import create_dataloader, collate_fn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Evaluator:
    """评估器"""
    
    def __init__(self, config, model_path, device='cuda'):
        self.config = config
        self.device = device
        
        # 创建模型
        self.model = MultimodalVideoClassifier(config).to(device)
        
        # 加载权重（如果提供了路径）
        if model_path is not None and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from {model_path}")
        elif model_path is not None:
            logger.warning(f"Model file not found: {model_path}")
        
        self.model.eval()
    
    def evaluate(self, test_loader):
        """评估模型"""
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(test_loader, desc='Evaluating')
        
        with torch.no_grad():
            for batch in pbar:
                frames = batch['frames'].to(self.device)
                audio = batch['audio'].to(self.device)
                texts = batch['texts']
                labels = batch['labels'].to(self.device)
                
                try:
                    outputs = self.model(frames, audio, texts)
                    logits = outputs['logits']
                    
                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(logits, dim=1)
                    
                    all_preds.extend(preds.detach().cpu().numpy())
                    all_labels.extend(labels.detach().cpu().numpy())
                    all_probs.extend(probs.detach().cpu().numpy())
                
                except Exception as e:
                    logger.error(f"Error: {e}")
                    continue
        
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='weighted', zero_division=0),
            'recall': recall_score(all_labels, all_preds, average='weighted', zero_division=0),
            'f1': f1_score(all_labels, all_preds, average='weighted', zero_division=0),
        }
        
        return metrics, all_preds, all_labels, all_probs
    
    def print_metrics(self, metrics):
        """打印指标"""
        logger.info("="*50)
        logger.info("Evaluation Metrics")
        logger.info("="*50)
        
        for key, value in metrics.items():
            logger.info(f"{key:15s}: {value:.4f}")
        
        logger.info("="*50)
    
    def save_results(self, metrics, predictions, labels, output_dir='./results'):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存指标
        metrics_file = os.path.join(output_dir, 'metrics.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # 保存预测结果
        results = {
            'predictions': predictions.tolist(),
            'labels': labels.tolist(),
        }
        
        results_file = os.path.join(output_dir, 'predictions.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        
        logger.info(f"Results saved to {output_dir}")


class Predictor:
    """单样本预测器"""
    
    def __init__(self, config, model_path, device='cuda'):
        self.config = config
        self.device = device
        
        # 创建模型
        self.model = MultimodalVideoClassifier(config).to(device)
        
        # 加载权重（如果提供了路径）
        if model_path is not None and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded model from {model_path}")
        
        self.model.eval()
    
    def predict(self, frames, audio, text):
        """单样本预测"""
        # 准备输入
        frames = frames.unsqueeze(0).to(self.device)  # (1, T, C, H, W)
        audio = torch.from_numpy(audio).unsqueeze(0).float().to(self.device)  # (1, audio_len)
        texts = [text]  # List[str]
        
        with torch.no_grad():
            outputs = self.model(frames, audio, texts)
            logits = outputs['logits']
            probs = F.softmax(logits, dim=1)
        
        pred = torch.argmax(logits, dim=1).item()
        confidence = torch.max(probs, dim=1)[0].item()
        
        return pred, confidence, probs[0].detach().cpu().numpy()


class FeatureExtractor:
    """特征提取器"""
    
    def __init__(self, config, model_path, device='cuda'):
        self.config = config
        self.device = device
        
        # 创建模型
        self.model = MultimodalVideoClassifier(config).to(device)
        
        # 加载权重（如果提供了路径）
        if model_path is not None and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
    
    def extract_features(self, frames, audio, text):
        """提取融合特征"""
        # Accept either (T, C, H, W) or (B, T, C, H, W).
        if isinstance(frames, torch.Tensor):
            if frames.dim() == 4:
                # single sample (T, C, H, W) -> add batch
                frames = frames.unsqueeze(0).to(self.device)
            elif frames.dim() == 5:
                # already batched (B, T, C, H, W)
                frames = frames.to(self.device)
            else:
                raise ValueError(f"Unexpected frames tensor shape: {frames.shape}")
        else:
            # If frames provided as numpy array or list, convert
            frames = torch.from_numpy(np.array(frames)).to(self.device)

        # Audio: accept 1D array/tensor (audio_len) or batched (B, audio_len)
        if isinstance(audio, torch.Tensor):
            if audio.dim() == 1:
                audio = audio.unsqueeze(0).float().to(self.device)
            elif audio.dim() == 2:
                audio = audio.float().to(self.device)
            else:
                raise ValueError(f"Unexpected audio tensor shape: {audio.shape}")
        else:
            audio = torch.from_numpy(np.array(audio)).unsqueeze(0).float().to(self.device)

        texts = [text]
        
        with torch.no_grad():
            outputs = self.model(frames, audio, texts)
            features = outputs['features']  # (1, D)
        
        return features[0].detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description='Evaluate multimodal video model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--data-root', type=str, default='./data')
    parser.add_argument('--output', type=str, default='./results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # 加载配置
    import yaml
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['data_root'] = args.data_root
    
    # 创建测试数据加载器
    test_loader = create_dataloader(config, split='test', shuffle=False)
    
    # 创建评估器
    evaluator = Evaluator(config, args.model, args.device)
    
    # 评估
    metrics, predictions, labels, probs = evaluator.evaluate(test_loader)
    
    # 打印结果
    evaluator.print_metrics(metrics)
    
    # 保存结果
    evaluator.save_results(metrics, predictions, labels, args.output)


if __name__ == '__main__':
    main()
