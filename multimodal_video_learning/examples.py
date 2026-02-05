"""Example usage script"""

# Force UTF-8 output to avoid Unicode/terminal rendering issues on Windows consoles.
import os
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

# Ensure package root is on sys.path so imports work when running this
# script directly (avoids "attempted relative import beyond top-level package").
import sys
from pathlib import Path
pkg_root = Path(__file__).resolve().parent
# Ensure the project root (parent of the package dir) is on sys.path so
# absolute package imports like `import multimodal_video_learning.modules`
# work when running this file directly.
project_root = pkg_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import yaml
import numpy as np
from pathlib import Path
import logging

from multimodal_video_learning.models.multimodal_model import MultimodalVideoClassifier
from multimodal_video_learning.data.dataset_loader import CustomVideoDataset, collate_fn
from multimodal_video_learning.inference import Predictor, FeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config():
    """Locate and load a YAML config file from common locations.

    Tries (in order): `configs/config.yaml`, `config/config.yaml`,
    `config_files/*.(yaml|yml)` (first match). Raises FileNotFoundError
    if none found.
    """
    candidates = [
        os.path.join(str(project_root), 'configs', 'config.yaml'),
        os.path.join(str(project_root), 'config', 'config.yaml'),
        os.path.join(str(project_root), 'config_files', 'config.yaml'),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, 'r') as f:
                return yaml.safe_load(f)

    # Fallback: pick the first YAML file in config_files/
    fallback_dir = os.path.join(str(project_root), 'config_files')
    if os.path.isdir(fallback_dir):
        for name in sorted(os.listdir(fallback_dir)):
            if name.endswith('.yaml') or name.endswith('.yml'):
                with open(os.path.join(fallback_dir, name), 'r') as f:
                    return yaml.safe_load(f)

    raise FileNotFoundError("Could not find a config file in 'configs' or 'config_files'.")


def _deep_update(orig, new):
    """Recursively update dict `orig` with `new` without losing nested defaults."""
    for k, v in new.items():
        if isinstance(v, dict) and k in orig and isinstance(orig[k], dict):
            _deep_update(orig[k], v)
        else:
            orig[k] = v
    return orig


def load_config_with_defaults():
    """Load a config (via `load_config`) and merge with minimal defaults for examples."""
    try:
        cfg = load_config()
    except FileNotFoundError:
        cfg = {}

    # Minimal defaults to let examples run locally
    defaults = {
        'vision': {'output_dim': 768},
        'audio': {
            'output_dim': 768,
            'sr': 16000,
            'n_mfcc': 40,
            'n_fft': 1024,
            'hop_length': 512
        },
        'text': {'output_dim': 768, 'max_seq_length': 32},
        'dataset': {'num_frames': 8, 'num_classes': 101},
        'training': {'label_smoothing': 0.0},
        'alignment': {'method': 'temporal'},
        'fusion': {'type': 'default'},
        'temporal': {'model': 'lstm'}
    }

    merged = _deep_update(defaults, cfg if isinstance(cfg, dict) else {})
    return merged


def example_1_basic_inference():
    """Example 1: Basic Inference"""
    logger.info("="*60)
    logger.info("Example 1: Basic Inference")
    logger.info("="*60)
    
    # Load config (merged with defaults)
    config = load_config_with_defaults()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    model = MultimodalVideoClassifier(config).to(device)
    
    # 创建虚拟数据
    batch_size = 2
    num_frames = config['dataset']['num_frames']
    
    frames = torch.randn(batch_size, num_frames, 3, 224, 224).to(device)
    audio = torch.randn(batch_size, 16000).to(device)  # 1秒16kHz音频
    texts = ['video action 1', 'video action 2']
    labels = torch.tensor([0, 1]).to(device)
    
    # 前向传播
    outputs = model(frames, audio, texts, labels)
    
    logger.info(f"Logits shape: {outputs['logits'].shape}")
    logger.info(f"Loss: {outputs['loss'].item():.4f}")
    logger.info(f"Features shape: {outputs['features'].shape}")


def example_2_feature_extraction():
    """Example 2: Feature Extraction"""
    logger.info("="*60)
    logger.info("Example 2: Feature Extraction")
    logger.info("="*60)
    
    config = load_config_with_defaults()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建虚拟数据
    frames = torch.randn(1, 8, 3, 224, 224)
    audio = np.random.randn(16000)
    text = 'action description'
    
    # 提取特征
    feature_extractor = FeatureExtractor(config, None, device)
    features = feature_extractor.extract_features(frames, audio, text)
    
    logger.info(f"Extracted features shape: {features.shape}")
    logger.info(f"Features (first 10): {features[:10]}")


def example_3_prediction():
    """Example 3: Single Sample Prediction"""
    logger.info("="*60)
    logger.info("Example 3: Single Sample Prediction")
    logger.info("="*60)
    
    config = load_config_with_defaults()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建虚拟数据
    frames = torch.randn(8, 3, 224, 224)
    audio = np.random.randn(16000)
    text = 'video action'
    
    # 预测
    predictor = Predictor(config, None, device)
    pred_class, confidence, probabilities = predictor.predict(frames, audio, text)
    
    logger.info(f"Predicted class: {pred_class}")
    logger.info(f"Confidence: {confidence:.4f}")
    logger.info(f"Class probabilities: {probabilities}")


def example_4_dataset_loading():
    """Example 4: Dataset Loading"""
    logger.info("="*60)
    logger.info("Example 4: Dataset Loading")
    logger.info("="*60)
    
    # 创建自定义视频数据集
    video_dir = './sample_videos'
    
    # 确保目录存在
    Path(video_dir).mkdir(exist_ok=True)
    
    try:
        dataset = CustomVideoDataset(video_dir, num_frames=8)
        logger.info(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"Sample frames shape: {sample['frames'].shape}")
            logger.info(f"Sample audio shape: {sample['audio'].shape}")
            logger.info(f"Sample text: {sample['text']}")
    except Exception as e:
        logger.warning(f"Could not load dataset: {e}")


def example_5_model_save_load():
    """Example 5: Model Save and Load"""
    logger.info("="*60)
    logger.info("Example 5: Model Save and Load")
    logger.info("="*60)
    
    config = load_config_with_defaults()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建模型
    model = MultimodalVideoClassifier(config).to(device)
    
    # 保存模型
    checkpoint_path = 'checkpoints/example_model.pt'
    Path('checkpoints').mkdir(exist_ok=True)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
    }
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Model saved to {checkpoint_path}")
    
    # 加载模型
    new_model = MultimodalVideoClassifier(config).to(device)
    loaded_checkpoint = torch.load(checkpoint_path, map_location=device)
    new_model.load_state_dict(loaded_checkpoint['model_state_dict'])
    logger.info(f"Model loaded from {checkpoint_path}")


def example_6_batch_processing():
    """Example 6: Batch Processing"""
    logger.info("="*60)
    logger.info("Example 6: Batch Processing")
    logger.info("="*60)
    
    config = load_config_with_defaults()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultimodalVideoClassifier(config).to(device)
    model.eval()
    
    batch_size = 4
    num_frames = config['dataset']['num_frames']
    
    with torch.no_grad():
        for i in range(2):
            frames = torch.randn(batch_size, num_frames, 3, 224, 224).to(device)
            audio = torch.randn(batch_size, 16000).to(device)
            texts = [f'action {j}' for j in range(batch_size)]
            
            outputs = model(frames, audio, texts)
            logger.info(f"Batch {i+1} - Logits shape: {outputs['logits'].shape}")


def example_7_gradient_analysis():
    """Example 7: Gradient Analysis"""
    logger.info("="*60)
    logger.info("Example 7: Gradient Analysis")
    logger.info("="*60)
    
    config = load_config_with_defaults()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultimodalVideoClassifier(config).to(device)
    model.train()
    
    # Create input tensors
    frames = torch.randn(2, 8, 3, 224, 224, device=device, requires_grad=True)
    audio = torch.randn(2, 16000, device=device)  # Audio: no grad needed (processor calls .numpy() internally)
    texts = ['action 1', 'action 2']
    labels = torch.tensor([0, 1], device=device)
    
    # Forward pass - compute loss
    outputs = model(frames, audio, texts, labels)
    loss = outputs['loss']
    
    # Backward pass - compute gradients
    loss.backward()
    
    logger.info(f"Loss: {loss.item():.4f}")
    logger.info(f"Frames gradient shape: {frames.grad.shape}")
    logger.info(f"Frames gradient mean: {frames.grad.mean().item():.6f}")
    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")


def main():
    """Run all examples"""
    logger.info("\n" + "="*60)
    logger.info("Multimodal Video Understanding Examples")
    logger.info("="*60 + "\n")
    
    try:
        example_1_basic_inference()
    except Exception as e:
        logger.exception("Example 1 failed")
    
    try:
        example_2_feature_extraction()
    except Exception as e:
        logger.exception("Example 2 failed")
    
    try:
        example_3_prediction()
    except Exception as e:
        logger.exception("Example 3 failed")
    
    try:
        example_4_dataset_loading()
    except Exception as e:
        logger.exception("Example 4 failed")
    
    try:
        example_5_model_save_load()
    except Exception as e:
        logger.exception("Example 5 failed")
    
    try:
        example_6_batch_processing()
    except Exception as e:
        logger.exception("Example 6 failed")
    
    try:
        example_7_gradient_analysis()
    except Exception as e:
        logger.exception("Example 7 failed")
    
    logger.info("\n" + "="*60)
    logger.info("All examples completed!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
