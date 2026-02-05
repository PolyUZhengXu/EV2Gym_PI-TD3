"""Initialization script - one-shot setup and verification"""

import os
import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_directory_structure():
    """创建目录结构"""
    dirs = [
        'data/videos',
        'data/annotations',
        'checkpoints',
        'logs',
        'results',
        'sample_videos',
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Created: {dir_path}")


def create_sample_config():
        """Create a sample configuration file"""
        sample_config = """# Multimodal Video Learning - sample config
# This is a sample configuration file. Modify as needed.

dataset:
    name: "UCF101"
    num_frames: 8
    frame_size: [224, 224]
    num_classes: 101
    batch_size: 8
    num_workers: 4

model:
    vision_model: "vit"
    audio_model: "whisper"
    text_model: "bert"
  
training:
    num_epochs: 100
    learning_rate: 0.0001
    batch_size: 8
"""
    
    with open('sample_config.yaml', 'w') as f:
        f.write(sample_config)
    
    logger.info("✓ Created: sample_config.yaml")


def print_next_steps():
     """Print next steps and quick start instructions"""
     logger.info("\n" + "="*60)
     logger.info("Setup Complete! Next Steps:")
     logger.info("="*60)
    
     steps = """
1. Read the documentation:
    - README.md              # Usage guide
    - PROJECT_GUIDE.md       # Project architecture
    - COMPLETION_REPORT.md   # Project report

2. Run examples:
    python examples.py

3. Quick start script:
    python quickstart.py

4. Prepare data:
    - Download UCF101 or MSR-VTT datasets
    - Place data under the data/ directory
    - Update paths in configs/config.yaml

5. Start training:
    python train.py --config configs/config.yaml

6. Evaluate model:
    python inference.py --model checkpoints/best_model.pt

Need help? See:
    - examples.py            # Full usage examples
    - README.md              # Detailed documentation
    - PROJECT_GUIDE.md       # Project guide
"""

     logger.info(steps)


def verify_structure():
    """验证项目结构"""
    logger.info("\n" + "="*60)
    logger.info("Verifying Project Structure:")
    logger.info("="*60)
    
    required_files = [
        'configs/config.yaml',
        'modules/__init__.py',
        'modules/vision_extractor.py',
        'modules/audio_extractor.py',
        'modules/text_extractor.py',
        'modules/alignment_module.py',
        'modules/fusion_module.py',
        'modules/temporal_module.py',
        'models/__init__.py',
        'models/multimodal_model.py',
        'data/__init__.py',
        'data/dataset_loader.py',
        'train.py',
        'inference.py',
        'examples.py',
        'README.md',
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            logger.info(f"✓ {file_path}")
        else:
            logger.warning(f"✗ {file_path} (missing)")
            all_exist = False
    
    if all_exist:
        logger.info("\n✓ All files present!")
    else:
        logger.warning("\n✗ Some files are missing")
    
    return all_exist


def main():
    parser = argparse.ArgumentParser(description='Initialize multimodal video learning project')
    parser.add_argument('--verify-only', action='store_true', help='Only verify structure')
    parser.add_argument('--create-dirs', action='store_true', help='Create directory structure')
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*60)
    logger.info("Multimodal Video Learning - Project Initialization")
    logger.info("="*60 + "\n")
    
    # 验证结构
    all_exist = verify_structure()
    
    if args.verify_only:
        return
    
    if not all_exist:
        logger.error("Project structure incomplete!")
        sys.exit(1)
    
    # 创建目录
    create_directory_structure()
    
    # 创建样本配置
    create_sample_config()
    
    # 打印后续步骤
    print_next_steps()
    
    logger.info("\n" + "="*60)
    logger.info("✓ Initialization Complete!")
    logger.info("="*60 + "\n")


if __name__ == '__main__':
    main()
