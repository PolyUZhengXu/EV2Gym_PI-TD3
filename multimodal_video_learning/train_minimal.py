#!/usr/bin/env python
"""
Minimal training test script
"""

import sys
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

print("="*60)
print("Starting minimal training test...")
print("="*60)

# Step 1: Check imports
print("\n[1] Checking imports...")
try:
    import torch
    import yaml
    print("  ✓ PyTorch and YAML OK")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Step 2: Load config
print("\n[2] Loading configuration...")
try:
    config_path = Path(__file__).parent / 'configs' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['data_root'] = str(Path(__file__).parent / 'data')
    print(f"  ✓ Config loaded")
    print(f"    - Dataset: {config['dataset']['name']}")
    print(f"    - Batch size: {config['dataset']['batch_size']}")
    print(f"    - Num workers: {config['dataset']['num_workers']}")
except Exception as e:
    print(f"  ✗ Config loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Import models
print("\n[3] Importing model modules...")
try:
    from models.multimodal_model import MultimodalVideoClassifier
    print("  ✓ Model imported")
except Exception as e:
    print(f"  ✗ Model import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Create model
print("\n[4] Creating model...")
try:
    device = 'cpu'
    model = MultimodalVideoClassifier(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model created with {total_params:,} parameters")
except Exception as e:
    print(f"  ✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Import data loader
print("\n[5] Importing data loader...")
try:
    from data.dataset_loader import create_dataloader
    print("  ✓ Data loader imported")
except Exception as e:
    print(f"  ✗ Data loader import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Create data loaders (no workers)
print("\n[6] Creating data loaders (num_workers=0)...")
try:
    train_loader = create_dataloader(config, split='train', shuffle=True)
    val_loader = create_dataloader(config, split='val', shuffle=False)
    print(f"  ✓ Train loader: {len(train_loader)} batches")
    print(f"  ✓ Val loader: {len(val_loader)} batches")
except Exception as e:
    print(f"  ✗ Data loader creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 7: Test a single batch
print("\n[7] Testing data loading (first batch)...")
try:
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            frames = batch['frames'].to(device)
            audio = batch['audio'].to(device)
            texts = batch['texts']
            labels = batch['labels'].to(device) if batch['labels'] is not None else None
            
            print(f"  ✓ Batch loaded successfully")
            print(f"    - Frames: {frames.shape}")
            print(f"    - Audio: {audio.shape}")
            print(f"    - Labels: {labels.shape if labels is not None else None}")
            
            # Forward pass
            outputs = model(frames, audio, texts, labels)
            print(f"  ✓ Model forward pass OK")
            print(f"    - Logits: {outputs['logits'].shape}")
            print(f"    - Loss: {outputs['loss'].item():.4f}")
            break
except Exception as e:
    print(f"  ✗ Batch loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ All tests passed! Ready to train.")
print("="*60)
print("\nNow ready to run full training with:")
print("  python train_simple.py --config configs/config.yaml --data-root ./data --device cpu")
print("\nOr use the simpler script:")
print("  python train_minimal.py")
