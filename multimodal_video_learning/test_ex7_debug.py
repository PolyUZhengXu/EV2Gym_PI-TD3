#!/usr/bin/env python
"""Debug Example 7"""

# Force UTF-8 output and disable HF progress bars to avoid terminal
# garbled characters on Windows consoles during model weight loading.
import os
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')

import sys
import traceback
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

import torch
import yaml
from multimodal_video_learning.models.multimodal_model import MultimodalVideoClassifier

try:
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultimodalVideoClassifier(config).to(device)
    model.train()

    frames = torch.randn(2, 8, 3, 224, 224, device=device, requires_grad=True)
    audio = torch.randn(2, 16000, device=device, requires_grad=False)
    texts = ['action 1', 'action 2']
    labels = torch.tensor([0, 1], device=device)

    print(f"Input shapes:")
    print(f"  frames: {frames.shape}")
    print(f"  audio: {audio.shape}")
    print(f"  texts: {texts}")
    print(f"  labels: {labels.shape}")
    
    print(f"\nCalling model forward...")
    outputs = model(frames, audio, texts, labels)
    loss = outputs['loss']

    print(f"\nCalling backward...")
    loss.backward()

    print(f'\n✓ SUCCESS!')
    print(f'Loss: {loss.item():.4f}')
    print(f'Frames gradient shape: {frames.grad.shape}')
    print(f'Frames gradient mean: {frames.grad.mean().item():.6f}')
    
except Exception as e:
    print(f'\n✗ ERROR: {e}')
    print(f'\nFull traceback:')
    traceback.print_exc()
