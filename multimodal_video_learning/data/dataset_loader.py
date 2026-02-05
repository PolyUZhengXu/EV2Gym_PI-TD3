"""数据加载器 - 支持UCF101和MSR-VTT等数据集"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
import os
import json
import numpy as np
import librosa
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def read_video_opencv(video_path):
    """
    使用OpenCV读取视频，避免PyAV依赖
    Returns:
        video: (T, H, W, C) in RGB format
        audio: None (音频由其他方式处理)
        info: dict with video metadata
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 转换BGR到RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        raise RuntimeError(f"No frames read from video: {video_path}")
    
    # 转换为torch tensor (T, H, W, C)
    video = np.stack(frames, axis=0)
    video = torch.from_numpy(video).float()
    
    info = {
        'fps': fps,
        'total_frames': total_frames,
        'width': width,
        'height': height
    }
    
    return video, None, info


class VideoTransforms:
    """视频转换"""
    
    def __init__(self, size=224, num_frames=8):
        self.size = size
        self.num_frames = num_frames
        
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.CenterCrop((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def __call__(self, frames):
        """
        Args:
            frames: List[PIL.Image] or torch.Tensor (T, C, H, W)
        Returns:
            transformed_frames: (num_frames, 3, size, size)
        """
        if isinstance(frames, torch.Tensor):
            # 如果是tensor，则转换为PIL
            from torchvision.transforms.functional import to_pil_image
            frames = [to_pil_image(f) for f in frames]
        
        # 采样帧
        if len(frames) > self.num_frames:
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        elif len(frames) < self.num_frames:
            # 重复帧
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
        
        # 转换每一帧
        transformed = []
        for frame in frames:
            transformed.append(self.transform(frame))
        
        return torch.stack(transformed)  # (num_frames, 3, size, size)


class UCF101Dataset(Dataset):
    """UCF101数据集"""
    
    def __init__(self, root_dir, split='train', num_frames=8, config=None):
        """
        Args:
            root_dir: UCF101数据集根目录
            split: 'train', 'val', 'test'
            num_frames: 采样的帧数
            config: 配置文件
        """
        self.root_dir = root_dir
        self.split = split
        self.num_frames = num_frames
        self.config = config or {}
        
        self.video_dir = os.path.join(root_dir, 'UCF101')
        self.labels_dir = os.path.join(root_dir, 'ucfTrainTestlist')
        
        self.videos = []
        self.labels = []
        self.class_names = []
        
        self._load_dataset()
        
        self.video_transform = VideoTransforms(size=224, num_frames=num_frames)
    
    def _load_dataset(self):
        """加载数据集"""
        # 获取类别名称
        self.class_names = sorted(os.listdir(self.video_dir))
        
        # 加载分割文件
        split_file = os.path.join(
            self.labels_dir,
            f'{self.split}list01.txt'
        )
        
        if not os.path.exists(split_file):
            logger.warning(f"Split file not found: {split_file}")
            # 自动扫描视频文件
            self._auto_load()
        else:
            with open(split_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    video_path = os.path.join(self.video_dir, parts[0])
                    if os.path.exists(video_path):
                        self.videos.append(video_path)
                        class_name = parts[0].split('/')[0]
                        self.labels.append(self.class_names.index(class_name))
    
    def _auto_load(self):
        """自动扫描并加载视频"""
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(self.video_dir, class_name)
            if os.path.isdir(class_dir):
                for video_file in os.listdir(class_dir):
                    if video_file.endswith(('.avi', '.mp4', '.flv')):
                        video_path = os.path.join(class_dir, video_file)
                        self.videos.append(video_path)
                        self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]
        
        try:
            # 使用OpenCV读取视频（避免PyAV依赖）
            video, _, info = read_video_opencv(video_path)
            
            # video shape: (T, H, W, C) - OpenCV是 HWC格式
            # 转换为 (C, T, H, W) 用于video_transform
            video = video.permute(3, 0, 1, 2)  # (C, T, H, W)
            video = video.permute(1, 0, 2, 3)  # (T, C, H, W)
            
            # 转换视频帧
            frames = self.video_transform(video)  # (num_frames, 3, 224, 224)
            
            # 创建伪音频数据（OpenCV不支持音频提取）
            sr = 16000
            audio_np = np.zeros(sr)
            
            return {
                'frames': frames,
                'audio': audio_np,
                'label': label,
                'text': self.class_names[label],  # 使用类别名作为文本
                'video_path': video_path
            }
        
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            # 返回虚拟数据
            return {
                'frames': torch.zeros(self.num_frames, 3, 224, 224),
                'audio': np.zeros(16000),
                'label': label,
                'text': self.class_names[label],
                'video_path': video_path
            }



class MSRVTTDataset(Dataset):
    """MSR-VTT数据集（小规模教学视频集合）"""
    
    def __init__(self, root_dir, split='train', num_frames=8, config=None):
        """
        Args:
            root_dir: MSR-VTT数据集根目录
            split: 'train', 'val', 'test'
            num_frames: 采样的帧数
            config: 配置文件
        """
        self.root_dir = root_dir
        self.split = split
        self.num_frames = num_frames
        self.config = config or {}
        
        self.video_dir = os.path.join(root_dir, 'videos')
        self.annotation_file = os.path.join(root_dir, 'annotation', 'MSR_VTT.json')
        
        self.videos = []
        self.captions = []
        self.labels = []
        
        self._load_dataset()
        
        self.video_transform = VideoTransforms(size=224, num_frames=num_frames)
    
    def _load_dataset(self):
        """加载数据集"""
        if not os.path.exists(self.annotation_file):
            logger.warning(f"Annotation file not found: {self.annotation_file}")
            return
        
        with open(self.annotation_file, 'r') as f:
            data = json.load(f)
        
        # 加载视频信息
        for video in data['videos']:
            video_id = video['video_id']
            video_path = os.path.join(self.video_dir, f'{video_id}.mp4')
            
            if os.path.exists(video_path):
                self.videos.append(video_path)
                
                # 获取第一个字幕作为文本
                captions = [sent['caption'] for sent in data['sentences'] 
                           if sent['video_id'] == video_id]
                
                self.captions.append(captions[0] if captions else '')
                self.labels.append(video.get('category', 0))
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_path = self.videos[idx]
        caption = self.captions[idx]
        label = self.labels[idx]
        
        try:
            # 使用OpenCV读取视频（避免PyAV依赖）
            video, _, info = read_video_opencv(video_path)
            
            # video shape: (T, H, W, C) - OpenCV是 HWC格式
            # 转换为 (T, C, H, W)
            video = video.permute(0, 3, 1, 2)  # (T, C, H, W)
            
            # 转换视频帧
            frames = self.video_transform(video)
            
            # 创建伪音频数据
            audio_np = np.zeros(16000)
            
            return {
                'frames': frames,
                'audio': audio_np,
                'label': label,
                'text': caption,
                'video_path': video_path
            }
        
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            return {
                'frames': torch.zeros(self.num_frames, 3, 224, 224),
                'audio': np.zeros(16000),
                'label': label,
                'text': caption,
                'video_path': video_path
            }


class CustomVideoDataset(Dataset):
    """自定义视频数据集"""
    
    def __init__(self, video_dir, num_frames=8):
        """
        Args:
            video_dir: 包含视频的目录
            num_frames: 采样的帧数
        """
        self.video_dir = video_dir
        self.num_frames = num_frames
        
        self.videos = []
        self.labels = []
        
        # 收集所有视频文件
        video_extensions = ('.mp4', '.avi', '.flv', '.mov', '.mkv')
        for filename in os.listdir(video_dir):
            if filename.lower().endswith(video_extensions):
                self.videos.append(os.path.join(video_dir, filename))
                # 使用文件名作为标签
                label = os.path.splitext(filename)[0]
                self.labels.append(label)
        
        self.video_transform = VideoTransforms(size=224, num_frames=num_frames)
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]
        
        try:
            # 使用OpenCV读取视频（避免PyAV依赖）
            video, _, info = read_video_opencv(video_path)
            
            # video shape: (T, H, W, C)
            # 转换为 (T, C, H, W)
            video = video.permute(0, 3, 1, 2)  # (T, C, H, W)
            
            frames = self.video_transform(video)
            
            # 创建伪音频数据
            audio_np = np.zeros(16000)
            
            return {
                'frames': frames,
                'audio': audio_np,
                'label': label,
                'text': label,
                'video_path': video_path
            }
        
        except Exception as e:
            logger.error(f"Error loading video {video_path}: {e}")
            return {
                'frames': torch.zeros(self.num_frames, 3, 224, 224),
                'audio': np.zeros(16000),
                'label': label,
                'text': label,
                'video_path': video_path
            }


def collate_fn(batch):
    """自定义批处理函数"""
    frames_list = [item['frames'] for item in batch]
    audio_list = [item['audio'] for item in batch]
    texts_list = [item['text'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # 堆栈帧
    frames = torch.stack(frames_list)  # (B, T, 3, 224, 224)
    
    # 处理音频
    max_len = max(len(a) for a in audio_list)
    audio_padded = np.zeros((len(audio_list), max_len))
    for i, a in enumerate(audio_list):
        audio_padded[i, :len(a)] = a
    audio = torch.from_numpy(audio_padded).float()
    
    # 处理标签
    if isinstance(labels[0], int):
        labels = torch.tensor(labels)
    
    return {
        'frames': frames,
        'audio': audio,
        'texts': texts_list,
        'labels': labels
    }


def create_dataloader(config, split='train', shuffle=True):
    """创建数据加载器"""
    dataset_name = config['dataset']['name']
    
    # 获取数据根目录
    data_root = config.get('data_root', './data')
    if not os.path.isabs(data_root):
        # 如果是相对路径，转换为绝对路径（相对于 dataset_loader.py 所在的目录）
        base_dir = Path(__file__).parent.parent
        data_root = os.path.join(base_dir, data_root)
    
    if dataset_name == 'UCF101':
        ucf101_root = os.path.join(data_root, 'UCF101')
        dataset = UCF101Dataset(
            root_dir=ucf101_root,
            split=split,
            num_frames=config['dataset']['num_frames'],
            config=config
        )
    elif dataset_name == 'MSR-VTT':
        msrvtt_root = os.path.join(data_root, 'MSR-VTT')
        dataset = MSRVTTDataset(
            root_dir=msrvtt_root,
            split=split,
            num_frames=config['dataset']['num_frames'],
            config=config
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['dataset']['batch_size'],
        shuffle=shuffle,
        num_workers=config['dataset']['num_workers'],
        collate_fn=collate_fn
    )
    
    return dataloader
