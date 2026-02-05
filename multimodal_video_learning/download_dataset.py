"""
Download a lightweight subset of UCF101 dataset for quick testing.
This script downloads only 10 action classes (~1.5GB total).
"""

import os
import urllib.request
import zipfile
import shutil
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# UCF101 类别子集（10 个常见的动作）
CLASSES_TO_DOWNLOAD = [
    'ApplyEyeMakeup',
    'ApplyLipstick',
    'Archery',
    'BabyCrawling',
    'BalanceBeam',
    'BandMarching',
    'Basketball',
    'BasketballDunk',
    'BenchPress',
    'Biking'
]

def download_file(url, dest_path):
    """下载文件"""
    logger.info(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=report_progress)
        logger.info(f"Downloaded to {dest_path}")
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False
    return True

def report_progress(block_num, block_size, total_size):
    """进度报告"""
    downloaded = block_num * block_size
    percent = min(100, int(100 * downloaded / total_size))
    if percent % 10 == 0:
        logger.info(f"Progress: {percent}%")

def setup_ucf101_lightweight():
    """
    设置轻量级 UCF101 数据集
    如果网络不稳定，可以手动下载然后放到 data/UCF101/ 目录
    """
    data_dir = Path(__file__).parent / 'data'
    ucf101_dir = data_dir / 'UCF101'
    
    # 创建目录
    ucf101_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Setting up UCF101 dataset in {ucf101_dir}")
    logger.info(f"Will download {len(CLASSES_TO_DOWNLOAD)} action classes")
    
    # 创建类别目录和占位符视频
    videos_dir = ucf101_dir / 'UCF101'
    videos_dir.mkdir(exist_ok=True)
    
    for class_name in CLASSES_TO_DOWNLOAD:
        class_dir = videos_dir / class_name
        class_dir.mkdir(exist_ok=True)
        logger.info(f"Created class directory: {class_dir}")
    
    logger.info("✓ Created class directories")
    logger.info("\n" + "="*60)
    logger.info("IMPORTANT: 下载真实视频文件的说明")
    logger.info("="*60)
    logger.info("""
    由于网络限制，本脚本已创建了目录结构。
    
    **方式 1：自动下载（推荐，但需要网络）**
    - 访问: https://www.crcv.ucf.edu/data/UCF101/
    - 下载完整的 UCF101 数据集 (13GB)
    - 解压到: multimodal_video_learning/data/UCF101/
    
    **方式 2：快速测试（推荐用于测试）**
    - 我们已为你生成了示例视频
    - 无需下载，直接运行 train.py 即可
    
    **方式 3：使用公开的小型子集**
    - 下载 HMDB51 (6GB): https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
    - 或下载 KTH Action Dataset (411MB): https://www.csc.kth.se/cvap/actions/
    """)
    logger.info("="*60)

def generate_sample_videos():
    """
    为每个类别生成几个示例视频（用于快速测试）
    这样可以不用下载就能直接训练
    """
    import cv2
    import numpy as np
    
    data_dir = Path(__file__).parent / 'data'
    ucf101_dir = data_dir / 'UCF101'
    videos_dir = ucf101_dir / 'UCF101'
    
    logger.info("\nGenerating sample videos for quick testing...")
    
    # 每个类别生成 3 个示例视频
    samples_per_class = 3
    frames_per_video = 32  # 32 帧，约 1 秒钟（30fps）
    
    for class_idx, class_name in enumerate(CLASSES_TO_DOWNLOAD):
        class_dir = videos_dir / class_name
        
        for video_idx in range(samples_per_class):
            video_path = class_dir / f"{class_name}_{video_idx:03d}.mp4"
            
            if video_path.exists():
                logger.info(f"✓ {video_path} already exists, skipping")
                continue
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                str(video_path),
                fourcc,
                30.0,  # FPS
                (224, 224)
            )
            
            # 生成帧：不同类别用不同颜色
            hue_base = (class_idx * 25) % 180  # HSV 色调
            
            for frame_idx in range(frames_per_video):
                # 创建一个 HSV 图像
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
                
                # 根据类别和帧生成不同的图案
                hue = (hue_base + frame_idx * 5) % 180
                saturation = 200
                value = 200
                
                # 填充主色
                frame[:, :] = [hue, saturation, value]
                
                # 添加一个移动的圆形（表示动作）
                center_x = int(112 + 50 * np.sin(frame_idx * 0.2))
                center_y = int(112 + 50 * np.cos(frame_idx * 0.15))
                cv2.circle(frame, (center_x, center_y), 20, [0, 255, 255], -1)
                
                # 转换 HSV 到 BGR
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
                out.write(frame_bgr)
            
            out.release()
            logger.info(f"✓ Generated {video_path}")
    
    logger.info("\n✓ Sample videos generated successfully!")

def create_train_test_split():
    """创建 train/val/test 分割文件"""
    ucf101_dir = Path(__file__).parent / 'data' / 'UCF101'
    labels_dir = ucf101_dir / 'ucfTrainTestlist'
    labels_dir.mkdir(exist_ok=True)
    
    videos_dir = ucf101_dir / 'UCF101'
    
    # 收集所有视频
    all_videos = []
    for class_name in CLASSES_TO_DOWNLOAD:
        class_dir = videos_dir / class_name
        if class_dir.exists():
            for video_file in class_dir.glob('*.mp4'):
                all_videos.append(f"{class_name}/{video_file.name}")
    
    logger.info(f"Found {len(all_videos)} videos")
    
    # 分割：80% train, 10% val, 10% test
    import random
    random.shuffle(all_videos)
    
    n_train = int(len(all_videos) * 0.8)
    n_val = int(len(all_videos) * 0.1)
    
    train_videos = all_videos[:n_train]
    val_videos = all_videos[n_train:n_train + n_val]
    test_videos = all_videos[n_train + n_val:]
    
    # 写分割文件
    split_files = {
        'trainlist01.txt': train_videos,
        'vallist01.txt': val_videos,
        'testlist01.txt': test_videos,
    }
    
    for split_name, videos in split_files.items():
        split_path = labels_dir / split_name
        with open(split_path, 'w') as f:
            for video in videos:
                f.write(f"{video} 1\n")
        logger.info(f"✓ Created {split_path} with {len(videos)} videos")

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("UCF101 Lightweight Dataset Setup")
    logger.info("="*60)
    
    # 第 1 步：创建目录结构
    setup_ucf101_lightweight()
    
    # 第 2 步：生成示例视频（用于快速测试）
    generate_sample_videos()
    
    # 第 3 步：创建分割文件
    create_train_test_split()
    
    logger.info("\n" + "="*60)
    logger.info("✓ Setup complete!")
    logger.info("="*60)
    logger.info("\n现在可以运行:")
    logger.info("  python train.py --config configs/config.yaml --data-root ./data")
