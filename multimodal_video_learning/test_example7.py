#!/usr/bin/env python
"""Test Example 7 directly"""
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(current_dir))

import torch
import yaml
import logging
import traceback
from multimodal_video_learning.models.multimodal_model import MultimodalVideoClassifier

logging.basicConfig(level=logging.ERROR)

try:
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MultimodalVideoClassifier(config).to(device)
    model.train()

    frames = torch.randn(2, 8, 3, 224, 224, device=device, requires_grad=True)
    audio = torch.randn(2, 16000, device=device, requires_grad=True)
    texts = ['action 1', 'action 2']
    labels = torch.tensor([0, 1], device=device)

    outputs = model(frames, audio, texts, labels)
    loss = outputs['loss']

    loss.backward()

    print(f'Loss: {loss.item():.4f}')
    print(f'Frames gradient shape: {frames.grad.shape}')
    print(f'Frames gradient mean: {frames.grad.mean().item():.6f}')
    print(f'Audio gradient shape: {audio.grad.shape}')
    print(f'Audio gradient mean: {audio.grad.mean().item():.6f}')
    print('\n✓ SUCCESS: Example 7 passed!')
except Exception as e:
    print(f'\n✗ ERROR: {e}')
    traceback.print_exc()
