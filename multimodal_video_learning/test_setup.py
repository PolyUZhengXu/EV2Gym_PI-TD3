#!/usr/bin/env python
"""Test training script"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Step 1: Testing imports...")
try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
except Exception as e:
    print(f"  ✗ PyTorch failed: {e}")
    sys.exit(1)

try:
    import yaml
    print(f"  ✓ YAML imported")
except Exception as e:
    print(f"  ✗ YAML failed: {e}")
    sys.exit(1)

print("\nStep 2: Testing data loading...")
try:
    from data.dataset_loader import create_dataloader
    print(f"  ✓ dataset_loader imported")
except Exception as e:
    print(f"  ✗ dataset_loader failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 3: Loading config...")
try:
    config_path = project_root / 'configs' / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"  ✓ Config loaded from {config_path}")
    print(f"    - Dataset: {config['dataset']['name']}")
    print(f"    - Batch size: {config['dataset']['batch_size']}")
except Exception as e:
    print(f"  ✗ Config failed: {e}")
    sys.exit(1)

print("\nStep 4: Creating data loaders...")
try:
    config['data_root'] = str(project_root / 'data')
    train_loader = create_dataloader(config, split='train', shuffle=True)
    print(f"  ✓ Train loader created: {len(train_loader)} batches")
    
    val_loader = create_dataloader(config, split='val', shuffle=False)
    print(f"  ✓ Val loader created: {len(val_loader)} batches")
except Exception as e:
    print(f"  ✗ Data loader creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nStep 5: Testing a batch...")
try:
    for batch in train_loader:
        print(f"  ✓ Batch loaded successfully")
        print(f"    - Frames shape: {batch['frames'].shape}")
        print(f"    - Audio shape: {batch['audio'].shape}")
        print(f"    - Labels shape: {batch['labels'].shape}")
        print(f"    - Num texts: {len(batch['texts'])}")
        break
except Exception as e:
    print(f"  ✗ Batch loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✓ All tests passed! Ready to train.")
print("="*60)
print("\nRun training with:")
print("  python train_simple.py --config configs/config.yaml --data-root ./data --device cpu")
