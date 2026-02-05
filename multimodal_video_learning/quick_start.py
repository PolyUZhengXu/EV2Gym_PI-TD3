#!/usr/bin/env python
"""
快速启动训练的脚本
一键生成数据集、配置并开始训练
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """主函数"""
    
    project_root = Path(__file__).parent
    
    logger.info("="*60)
    logger.info("Multimodal Video Learning - Quick Start")
    logger.info("="*60)
    
    # 步骤 1：检查环境
    logger.info("\n[1/4] Checking environment...")
    try:
        import torch
        logger.info(f"✓ PyTorch {torch.__version__} found")
    except ImportError:
        logger.error("✗ PyTorch not found. Please install it first.")
        sys.exit(1)
    
    # 步骤 2：生成数据集
    logger.info("\n[2/4] Setting up dataset...")
    
    sys.path.insert(0, str(project_root))
    
    try:
        from download_dataset import (
            setup_ucf101_lightweight,
            generate_sample_videos,
            create_train_test_split
        )
        
        setup_ucf101_lightweight()
        generate_sample_videos()
        create_train_test_split()
        
        logger.info("✓ Dataset ready!")
        
    except Exception as e:
        logger.error(f"✗ Failed to setup dataset: {e}")
        sys.exit(1)
    
    # 步骤 3：验证配置
    logger.info("\n[3/4] Verifying configuration...")
    config_path = project_root / 'configs' / 'config.yaml'
    
    if config_path.exists():
        logger.info(f"✓ Configuration file found: {config_path}")
    else:
        logger.error(f"✗ Configuration file not found: {config_path}")
        sys.exit(1)
    
    # 步骤 4：准备训练
    logger.info("\n[4/4] Preparing training...")
    logger.info("✓ All preparations complete!")
    
    logger.info("\n" + "="*60)
    logger.info("Now you can start training with:")
    logger.info("\n  python train.py --config configs/config.yaml --data-root ./data")
    logger.info("\nOr use this shortcut from project root:")
    logger.info("  python quick_start.py --train")
    logger.info("="*60)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick start for multimodal video learning')
    parser.add_argument('--train', action='store_true', help='Start training after setup')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # 运行初始化
    main()
    
    # 如果指定了 --train，则开始训练
    if args.train:
        logger.info("\nStarting training...")
        os.system(f'python train.py --config configs/config.yaml --data-root ./data --device {args.device}')
