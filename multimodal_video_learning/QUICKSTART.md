# 🚀 多模态视频学习框架 - 5分钟快速启动

## 方案选择

### ✅ 推荐方案（自动化，最简单）

**只需 3 步即可完成环境配置！**

#### Windows 用户

```bash
# 1️⃣ 双击运行
setup_windows.bat

# 或在 PowerShell 中运行
.\setup_windows.ps1
```

#### Linux/Mac 用户

```bash
# 1️⃣ 运行安装脚本
chmod +x setup_linux.sh
./setup_linux.sh
```

#### 跳过本机环境，直接测试代码

```bash
# 验证环境
python check_env.py

# 运行所有示例（包含7个完整案例）
python examples.py

# 快速启动工具
python quickstart.py
```

---

## 🎯 三种安装方案

### 方案 A：**一键自动安装**（推荐）

| 步骤 | 操作 |
|-----|------|
| **1** | 双击 `setup_windows.bat` 或 `setup_windows.ps1` |
| **2** | 选择 CUDA 版本（1=CPU, 2=11.8, 3=12.1）|
| **3** | 等待自动安装完成 |

**预计时间**：5-15 分钟（取决于网络）

---

### 方案 B：**Conda 虚拟环境**

```bash
# 只需一条命令
conda env create -f environment.yml
conda activate multimodal-video

# 验证
python check_env.py
```

**特点**：环境隔离，版本管理完善

---

### 方案 C：**手动安装**

```bash
# 1. 创建虚拟环境
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# 2. 安装 PyTorch（选一个）
# CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CUDA 11.8
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. 安装依赖
pip install -r requirements.txt

# 4. 验证
python check_env.py
```

**特点**：完全控制，适合高级用户

---

## ✨ 环境激活（每次使用）

安装完成后，每次使用框架都需要激活虚拟环境：

**Windows：**
```bash
venv\Scripts\activate
```

**Linux/Mac：**
```bash
source venv/bin/activate
```

**Conda：**
```bash
conda activate multimodal-video
```

---

## 🧪 验证安装

```bash
# 完整的环境检查（推荐）
python check_env.py

# 快速检查
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'GPU: {torch.cuda.is_available()}')"
```

---

## 📚 安装后的下一步

### 1️⃣ 查看示例（推荐首先运行）

```bash
python examples.py
```

包含 7 个完整示例：
- ✅ 基本推理
- ✅ 特征提取
- ✅ 单样本预测
- ✅ 数据集加载
- ✅ 模型保存/加载
- ✅ 批处理
- ✅ 梯度分析

### 2️⃣ 快速启动工具

```bash
python quickstart.py
```

自动完成：
- ✅ 环境检查
- ✅ 依赖验证
- ✅ 示例运行
- ✅ 快速启动指南

### 3️⃣ 开始训练

```bash
# 准备数据（放到 data/ 目录）
# 然后运行训练

python train.py \
  --config configs/config.yaml \
  --device cuda \
  --data-root ./data
```

### 4️⃣ 模型推理

```bash
python inference.py \
  --config configs/config.yaml \
  --model checkpoints/best_model.pt \
  --device cuda
```

---

## 🔧 常见问题速解

| 问题 | 解决方案 |
|-----|--------|
| **pip 很慢** | `pip config set global.index-url https://pypi.tsinghua.edu.cn/simple` |
| **GPU 不可用** | 检查 `nvidia-smi` 和 `nvcc --version` |
| **显存不足** | 修改 `config.yaml` 中的 `batch_size` 或 `num_frames` |
| **模块导入失败** | 确保虚拟环境已激活 `which python` |
| **CUDA 版本不匹配** | 重新安装 PyTorch 选择正确的 CUDA 版本 |

更多问题见 [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md)

---

## 📊 环境检查清单

完成安装后，应该看到以下输出：

```
✓ 正常 (12+ 项):
  ✓ Python 3.10+
  ✓ PyTorch: 2.0+
  ✓ CUDA 可用 (或 CPU 模式)
  ✓ Transformers: 4.30+
  ✓ Librosa, OpenCV, scikit-learn...
  
总计: 12+ 项 - ✓ 正常
```

---

## 💾 依赖包概览

| 包名 | 版本 | 用途 | 必需 |
|-----|------|------|-----|
| **PyTorch** | 2.0+ | 深度学习框架 | ✅ |
| **Transformers** | 4.30+ | BERT/Whisper | ✅ |
| **NumPy** | 1.24+ | 数值计算 | ✅ |
| **Librosa** | 0.10+ | 音频处理 | ⚠️ |
| **OpenCV** | 4.8+ | 视频处理 | ⚠️ |
| **scikit-learn** | 1.3+ | 评估指标 | ⚠️ |
| **TensorBoard** | 2.13+ | 训练可视化 | ⚠️ |

✅ = 必需，⚠️ = 推荐但可选

---

## 🎮 快速命令参考

```bash
# 环境管理
python check_env.py              # 检查环境
python quickstart.py             # 快速启动
python examples.py               # 运行示例

# 模型训练
python train.py --config configs/config.yaml --device cuda

# 模型评估
python inference.py --model checkpoints/best_model.pt

# 特征提取
python -c "from inference import FeatureExtractor; ..."

# 查看帮助
python train.py --help
python inference.py --help
```

---

## 📖 详细文档

需要更多信息？

| 文档 | 内容 |
|-----|-----|
| [INSTALLATION_GUIDE.md](INSTALLATION_GUIDE.md) | 详细安装步骤和问题排查 |
| [README.md](README.md) | 项目概览和使用说明 |
| [PROJECT_GUIDE.md](PROJECT_GUIDE.md) | 项目架构和原理讲解 |
| [COMPLETION_REPORT.md](COMPLETION_REPORT.md) | 项目完成报告 |
| [configs/config.yaml](configs/config.yaml) | 配置参数说明 |

---

## ✅ 安装成功标志

✓ 能运行示例：`python examples.py`  
✓ 能进行推理：`python inference.py --model model.pt`  
✓ 环境检查通过：`python check_env.py`  
✓ 可以导入模块：`from models import MultimodalVideoClassifier`  

---

## 🎉 准备好开始了吗？

```bash
# 1. 激活虚拟环境
venv\Scripts\activate  # Windows 或
source venv/bin/activate  # Linux/Mac

# 2. 运行示例
python examples.py

# 3. 开始探索！
python quickstart.py
```

**祝您使用愉快！** 🚀

---

**上次更新**：2026年2月5日  
**项目版本**：1.0.0
