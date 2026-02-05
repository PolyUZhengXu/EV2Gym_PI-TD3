# 🎬 Multimodal Video Learning - 快速训练指南

## 📊 已完全配置！可以直接训练

我已经为你：
1. ✅ **生成了示例视频数据集**（30 个视频，10 个类别）
2. ✅ **配置了数据加载器**（支持 UCF101 格式）
3. ✅ **修改了训练脚本**（支持相对路径、自动配置）
4. ✅ **创建了启动脚本**（一键运行）

---

## 🚀 快速开始（3 种方式）

### **方式 1：Windows 用户（最简单）**

```bash
# 在项目根目录双击运行
train_windows.bat
```

### **方式 2：命令行（所有操作系统）**

```bash
# 进入项目根目录
cd d:\Program Files\PolyUCode\EV2Gym_PI-TD3

# 激活环境并运行训练
conda run -n multimodal-video python -m multimodal_video_learning.train ^
    --config multimodal_video_learning/configs/config.yaml ^
    --data-root multimodal_video_learning/data ^
    --device cpu
```

### **方式 3：从项目目录运行**

```bash
cd d:\Program Files\PolyUCode\EV2Gym_PI-TD3\multimodal_video_learning

conda run -n multimodal-video python download_dataset.py
conda run -n multimodal-video python -m multimodal_video_learning.train \
    --config configs/config.yaml \
    --data-root ./data \
    --device cpu
```

---

## 📂 数据集结构

```
multimodal_video_learning/
├── data/
│   └── UCF101/
│       ├── UCF101/                  ← 视频文件在这里
│       │   ├── ApplyEyeMakeup/      (3 个示例视频)
│       │   ├── ApplyLipstick/       (3 个示例视频)
│       │   ├── Archery/             (3 个示例视频)
│       │   ├── BabyCrawling/        ...
│       │   ├── BalanceBeam/
│       │   ├── BandMarching/
│       │   ├── Basketball/
│       │   ├── BasketballDunk/
│       │   ├── BenchPress/
│       │   └── Biking/
│       └── ucfTrainTestlist/        ← 分割文件
│           ├── trainlist01.txt      (24 个视频)
│           ├── vallist01.txt        (3 个视频)
│           └── testlist01.txt       (3 个视频)
├── configs/
│   └── config.yaml                  ← 配置文件
├── train.py                         ← 训练脚本
├── download_dataset.py              ← 数据集生成脚本
└── checkpoints/                     ← 模型会保存这里
```

---

## ⚙️ 配置参数说明

编辑 `multimodal_video_learning/configs/config.yaml` 来调整：

```yaml
dataset:
  batch_size: 8        # 批大小（内存不足改小：4 或 2）
  num_workers: 4       # 数据加载线程
  num_frames: 8        # 采样的帧数

training:
  num_epochs: 50       # 训练轮数
  learning_rate: 1e-4
  optimizer: adamw
```

---

## 📊 训练输出

运行时会看到：
```
============================================================
Multimodal Video Learning - Training
============================================================

[1/4] Checking environment...
✓ PyTorch 2.10.0+cpu found

[2/4] Setting up dataset...
✓ Dataset ready!

[3/4] Verifying configuration...
✓ Configuration file found

[4/4] Preparing training...
✓ All preparations complete!

============================================================
Starting training...
============================================================

Creating data loaders...
✓ Train loader: 3 batches
✓ Val loader: 1 batches

Creating trainer...
✓ Model created with 274796901 parameters

Epoch 1/50
Training: 100%|████████████████| 3/3 [00:15<00:00]
  loss=4.23, acc=0.15
Validation: 100%|████████████████| 1/1 [00:05<00:00]
  loss=3.98, acc=0.20
...
```

---

## 💾 模型保存位置

- **检查点**：`multimodal_video_learning/checkpoints/`
  - `checkpoint_epoch_1.pt`
  - `checkpoint_epoch_2.pt`
  - ...
- **最佳模型**：`checkpoints/best_model.pt`
- **TensorBoard 日志**：`logs/` 目录

---

## 🔄 用真实数据替换

如果想用真实的 UCF101 数据集：

1. 下载完整 UCF101：https://www.crcv.ucf.edu/data/UCF101/
2. 解压到：`multimodal_video_learning/data/UCF101/`
3. 运行训练

脚本会自动使用真实视频。

---

## ⏱️ 性能预期

| 数据集 | 批大小 | 设备 | 每 epoch 时间 |
|--------|--------|------|--------------|
| 示例（30 视频） | 2 | CPU | ~30 秒 |
| 示例（30 视频） | 8 | GPU | ~5 秒 |
| UCF101（13k 视频） | 32 | GPU | ~30 分钟 |

---

## 🆘 常见问题

### Q: 运行时出现 "ModuleNotFoundError"
**A**: 确保在正确的目录运行，或用 `-m` 模块方式运行

### Q: 内存不足
**A**: 修改 config.yaml 中的 `batch_size` 为 2 或 4

### Q: 数据集没有生成
**A**: 手动运行 `python download_dataset.py`

### Q: 用 GPU 但还是很慢
**A**: 检查是否真的用了 GPU，运行时会显示 "Using device: cuda:0"

---

## 📝 命令参考

```bash
# 只生成数据集
python download_dataset.py

# 查看帮助
python train.py --help

# 使用 GPU 训练
python -m multimodal_video_learning.train \
    --config configs/config.yaml \
    --data-root ./data \
    --device cuda

# 指定 epoch 数（修改配置文件）
# 编辑 config.yaml 的 training.num_epochs

# 从检查点恢复（需要修改 train.py）
# 在 Trainer.__init__ 中添加 load_checkpoint() 逻辑
```

---

## ✨ 现在就开始吧！

选择上面的任何一种方式运行训练，你会看到进度条和实时损失值。

好运！🎯
