"""快速启动脚本 - 一键运行完整流程"""

import os
import sys
import torch
import yaml
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_environment():
    """设置环境"""
    logger.info("="*60)
    logger.info("Setting up environment...")
    logger.info("="*60)
    
    # 检查GPU
    if torch.cuda.is_available():
        logger.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"  CUDA Version: {torch.version.cuda}")
    else:
        logger.warning("⚠ GPU not available, using CPU (slow)")
    
    # 创建必要的目录
    dirs = ['data', 'checkpoints', 'logs', 'results']
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        logger.info(f"✓ Created directory: {dir_name}")


def check_dependencies():
    """检查依赖"""
    logger.info("="*60)
    logger.info("Checking dependencies...")
    logger.info("="*60)
    
    required_packages = {
        'torch': 'torch',
        'torchvision': 'torchvision',
        'transformers': 'transformers',
        'librosa': 'librosa',
        'yaml': 'pyyaml',
        'sklearn': 'scikit-learn',
    }
    
    missing = []
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            logger.info(f"✓ {package_name}")
        except ImportError:
            logger.error(f"✗ {package_name} (missing)")
            missing.append(package_name)
    
    if missing:
        logger.error(f"\nMissing packages: {', '.join(missing)}")
        logger.error("Install with: pip install " + " ".join(missing))
        return False
    
    return True


def run_examples():
    """运行示例"""
    logger.info("="*60)
    logger.info("Running examples...")
    logger.info("="*60)
    
    try:
        from examples import main as run_examples_main
        run_examples_main()
        return True
    except Exception as e:
        logger.error(f"Examples failed: {e}")
        return False


def download_sample_data():
    """下载样本数据"""
    logger.info("="*60)
    logger.info("Setting up sample data...")
    logger.info("="*60)
    
    # 创建样本视频目录
    sample_dir = Path('data/sample_videos')
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"✓ Sample data directory: {sample_dir}")
    logger.info("  Note: Please add sample videos to this directory")
    logger.info("  Supported formats: .mp4, .avi, .flv, .mov, .mkv")


def print_quick_start():
    """打印快速启动指南"""
    logger.info("="*60)
    logger.info("Quick Start Guide")
    logger.info("="*60)
    
    guide = """
1. 基本推理 (无需数据):
   python examples.py

2. 训练模型 (需要数据):
   python train.py --config configs/config.yaml

3. 评估模型:
   python inference.py --config configs/config.yaml \\
                       --model checkpoints/best_model.pt

4. 自定义推理:
   from inference import Predictor
   predictor = Predictor(config, 'model.pt', device='cuda')
   prediction = predictor.predict(frames, audio, text)

5. 特征提取:
   from inference import FeatureExtractor
   extractor = FeatureExtractor(config, 'model.pt', device='cuda')
   features = extractor.extract_features(frames, audio, text)

详见 README.md 获取更多信息
"""
    
    logger.info(guide)


def main():
    parser = argparse.ArgumentParser(description='Quick start for multimodal video learning')
    parser.add_argument('--setup-only', action='store_true', help='Only setup environment')
    parser.add_argument('--check-deps', action='store_true', help='Only check dependencies')
    parser.add_argument('--run-examples', action='store_true', help='Run examples only')
    parser.add_argument('--skip-examples', action='store_true', help='Skip running examples')
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*60)
    logger.info("Multimodal Video Learning Framework - Quick Start")
    logger.info("="*60 + "\n")
    
    # 设置环境
    setup_environment()
    
    # 检查依赖
    if not check_dependencies():
        logger.error("Please install missing dependencies")
        sys.exit(1)
    
    if args.check_deps:
        logger.info("✓ All dependencies OK!")
        return
    
    # 下载样本数据
    download_sample_data()
    
    if args.setup_only:
        logger.info("✓ Setup complete!")
        return
    
    # 运行示例
    if args.run_examples or not args.skip_examples:
        if not run_examples():
            logger.warning("Examples encountered some errors")
    
    # 打印快速启动指南
    print_quick_start()
    
    logger.info("\n" + "="*60)
    logger.info("✓ Setup complete! Ready to train and inference")
    logger.info("="*60 + "\n")


if __name__ == '__main__':
    main()
