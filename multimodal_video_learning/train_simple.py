"""
Simple training script that works from any directory
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
import os
import sys
import logging
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import numpy as np

# Ensure proper imports regardless of where script is run from
current_file = Path(__file__).resolve()
if current_file.parent.name == 'multimodal_video_learning':
    project_root = current_file.parent
else:
    # Find multimodal_video_learning directory
    for parent in current_file.parents:
        if (parent / 'multimodal_video_learning').exists():
            project_root = parent / 'multimodal_video_learning'
            break
    else:
        project_root = current_file.parent

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Now import from local modules
from models.multimodal_model import MultimodalVideoClassifier
from data.dataset_loader import create_dataloader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """训练器类"""
    
    def __init__(self, config, device='cuda'):
        self.config = config
        self.device = device
        
        # 创建模型
        self.model = MultimodalVideoClassifier(config).to(device)
        
        # 创建优化器
        self.optimizer = self._create_optimizer()
        
        # 创建学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 创建日志记录器
        log_dir = config['logging']['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, timestamp))
        
        # 检查点目录
        self.checkpoint_dir = config['logging']['checkpoint_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.best_acc = 0.0
        self.global_step = 0
        self.epoch = 0
    
    def _create_optimizer(self):
        """创建优化器"""
        optimizer_name = self.config['training']['optimizer']
        learning_rate = self.config['training']['learning_rate']
        weight_decay = self.config['training']['weight_decay']
        
        if optimizer_name == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        scheduler_name = self.config['training']['scheduler']
        num_epochs = self.config['training']['num_epochs']
        warmup_epochs = self.config['training'].get('warmup_epochs', 0)
        
        if scheduler_name == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs - warmup_epochs,
                eta_min=1e-6
            )
        elif scheduler_name == 'linear':
            return LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=num_epochs
            )
        else:
            return None
    
    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc='Training')
        
        for batch_idx, batch in enumerate(pbar):
            # 移至设备
            frames = batch['frames'].to(self.device)
            audio = batch['audio'].to(self.device)
            texts = batch['texts']
            labels = batch['labels'].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            
            try:
                outputs = self.model(frames, audio, texts, labels)
                loss = outputs['loss']
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                if self.config['training'].get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer.step()
                
                # 统计
                total_loss += loss.item()
                logits = outputs['logits']
                preds = torch.argmax(logits, dim=1)
                
                if isinstance(labels, torch.Tensor):
                    total_correct += (preds == labels).sum().item()
                    total_samples += labels.shape[0]
                
                # 记录到TensorBoard
                if self.global_step % self.config['logging']['log_interval'] == 0:
                    self.writer.add_scalar(
                        'train/loss',
                        loss.item(),
                        self.global_step
                    )
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{total_correct / (total_samples + 1e-8):.4f}'
                })
                
                self.global_step += 1
            
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(batch_idx + 1, 1)
        avg_acc = total_correct / max(total_samples, 1)
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader):
        """验证"""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(val_loader, desc='Validation')
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                frames = batch['frames'].to(self.device)
                audio = batch['audio'].to(self.device)
                texts = batch['texts']
                labels = batch['labels'].to(self.device)
                
                try:
                    outputs = self.model(frames, audio, texts, labels)
                    loss = outputs['loss']
                    
                    total_loss += loss.item()
                    
                    logits = outputs['logits']
                    preds = torch.argmax(logits, dim=1)
                    
                    if isinstance(labels, torch.Tensor):
                        total_correct += (preds == labels).sum().item()
                        total_samples += labels.shape[0]
                    
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{total_correct / (total_samples + 1e-8):.4f}'
                    })
                
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {e}")
                    continue
        
        avg_loss = total_loss / max(batch_idx + 1, 1)
        avg_acc = total_correct / max(total_samples, 1)
        
        return avg_loss, avg_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_acc': self.best_acc,
        }
        
        save_path = os.path.join(
            self.checkpoint_dir,
            f'checkpoint_epoch_{epoch}.pt'
        )
        
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
    
    def train(self, train_loader, val_loader):
        """训练循环"""
        num_epochs = self.config['training']['num_epochs']
        save_interval = self.config['logging']['save_interval']
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # 验证
            val_loss, val_acc = self.validate(val_loader)
            
            # 记录到TensorBoard
            self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
            self.writer.add_scalar('epoch/train_acc', train_acc, epoch)
            self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
            self.writer.add_scalar('epoch/val_acc', val_acc, epoch)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 学习率调度
            if self.scheduler is not None:
                self.scheduler.step()
            
            # 保存检查点
            if (epoch + 1) % save_interval == 0:
                is_best = val_acc > self.best_acc
                if is_best:
                    self.best_acc = val_acc
                self.save_checkpoint(epoch + 1, is_best)
        
        self.writer.close()
        logger.info("Training finished!")


def main():
    parser = argparse.ArgumentParser(description='Train multimodal video model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--data-root', type=str, default='./data',
                       help='Path to data directory')
    
    args = parser.parse_args()
    
    # 确保配置文件路径是正确的
    config_path = args.config
    if not os.path.isabs(config_path):
        # 如果是相对路径，则相对于项目根目录
        config_path = project_root / config_path
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    logger.info(f"Using config: {config_path}")
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 确保数据根目录是正确的
    if not os.path.isabs(args.data_root):
        data_root = project_root / args.data_root
    else:
        data_root = args.data_root
    
    config['data_root'] = str(data_root)
    
    logger.info(f"Using device: {args.device}")
    logger.info(f"Using data root: {data_root}")
    
    # 创建数据加载器
    logger.info("Creating data loaders...")
    try:
        train_loader = create_dataloader(config, split='train', shuffle=True)
        val_loader = create_dataloader(config, split='val', shuffle=False)
        logger.info(f"✓ Train loader: {len(train_loader)} batches")
        logger.info(f"✓ Val loader: {len(val_loader)} batches")
    except Exception as e:
        logger.error(f"Failed to create data loaders: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 创建训练器
    logger.info("\nCreating trainer...")
    trainer = Trainer(config, device=args.device)
    total_params = sum(p.numel() for p in trainer.model.parameters())
    logger.info(f"✓ Model created with {total_params:,} parameters")
    
    # 开始训练
    logger.info("\n" + "="*60)
    logger.info("Starting training...")
    logger.info("="*60 + "\n")
    
    try:
        trainer.train(train_loader, val_loader)
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    logger.info("\n" + "="*60)
    logger.info("Training complete!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
