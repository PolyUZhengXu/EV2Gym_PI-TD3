# 环境配置完全指南

## 📋 目录

1. [快速开始（推荐）](#快速开始推荐)
2. [详细步骤](#详细步骤)
3. [常见问题](#常见问题)
4. [环境检查](#环境检查)

---

## 快速开始（推荐）

### Windows 系统

选择以下任意一种方法：

#### 方法 1️⃣：批处理文件（最简单）

```bash
# 双击运行或在 cmd 中执行
setup_windows.bat
```

✅ **特点**：
- 图形化菜单
- 自动选择 CUDA 版本
- 完整的进度提示

#### 方法 2️⃣：PowerShell 脚本（推荐）

```powershell
# 在 PowerShell 中执行
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\setup_windows.ps1
```

✅ **特点**：
- 彩色输出
- 更好的错误处理
- 自动激活虚拟环境

#### 方法 3️⃣：手动安装

```bash
# 1. 创建虚拟环境
python -m venv venv

# 2. 激活虚拟环境
venv\Scripts\activate

# 3. 升级 pip
python -m pip install --upgrade pip

# 4. 安装 PyTorch（选择一个）
# CPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 或 CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 或 CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. 安装其他依赖
pip install -r requirements.txt
```

### Linux/Mac 系统

```bash
# 1. 赋予执行权限
chmod +x setup_linux.sh

# 2. 运行脚本
./setup_linux.sh
```

或手动安装：

```bash
# 1. 创建虚拟环境
python3 -m venv venv

# 2. 激活虚拟环境
source venv/bin/activate

# 3. 升级 pip
python3 -m pip install --upgrade pip

# 4. 安装 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 5. 安装依赖
pip install -r requirements.txt
```

### 使用 Conda（可选）

如果已安装 Conda/Anaconda：

```bash
# 1. 创建环境
conda env create -f environment.yml

# 2. 激活环境
conda activate multimodal-video

# 3. 验证安装
python check_env.py
```

---

## 详细步骤

### 步骤 1：检查 Python

```bash
python --version
# 应该输出 Python 3.9 或更高版本
```

如果未安装，请从 [python.org](https://www.python.org) 下载并安装。

### 步骤 2：创建虚拟环境

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 步骤 3：升级 pip

```bash
python -m pip install --upgrade pip
```

### 步骤 4：选择并安装 PyTorch

查看您的系统配置：

```bash
# 检查 GPU
nvidia-smi

# 或检查 CUDA 版本
nvcc --version
```

然后选择对应版本：

| 情况 | 命令 |
|-----|------|
| **无 GPU** | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` |
| **NVIDIA GPU CUDA 11.8** | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` |
| **NVIDIA GPU CUDA 12.1** | `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` |
| **Apple Silicon (M1/M2)** | `pip install torch torchvision torchaudio` |

### 步骤 5：安装其他依赖

```bash
pip install -r requirements.txt
```

或逐个安装：

```bash
pip install transformers>=4.30.0
pip install librosa>=0.10.0
pip install scikit-learn>=1.3.0
pip install pyyaml>=6.0
pip install tensorboard>=2.13.0
pip install tqdm>=4.66.0
pip install opencv-python>=4.8.0
```

### 步骤 6：验证安装

```bash
python check_env.py
```

应该看到类似输出：

```
✓ 正常 (12 项):
  ✓ Python 3.10.11
  ✓ PyTorch: 2.0.1+cu118
  ✓ Transformers: 4.33.0
  ✓ Librosa: 0.10.0
  ...

总计: 12 项 - ✓ 正常
```

---

## 环境检查

运行环境检查脚本：

```bash
python check_env.py
```

### 检查内容

✓ **必需** (缺失会影响功能)：
- Python 3.9+
- PyTorch
- Transformers
- NumPy

⚠ **可选** (某些功能需要)：
- Librosa (音频处理)
- OpenCV (视频处理)
- scikit-learn (评估指标)
- TensorBoard (训练可视化)

✗ **错误** (需要立即修复)：
- Python 版本过低
- 关键包缺失

---

## 常见问题

### Q1：pip 安装很慢

**解决方案**：使用清华大学镜像源

```bash
pip install -i https://pypi.tsinghua.edu.cn/simple transformers
# 或全局配置
pip config set global.index-url https://pypi.tsinghua.edu.cn/simple
```

### Q2：PyTorch 安装失败

**检查**：
1. 网络连接
2. 磁盘空间足够
3. Python 版本正确

**解决**：
```bash
# 增加超时时间
pip install torch --default-timeout=1000 --index-url https://download.pytorch.org/whl/cpu
```

### Q3：找不到虚拟环境

**解决**：

Windows：
```bash
python -m venv venv
venv\Scripts\activate
```

Linux/Mac：
```bash
python3 -m venv venv
source venv/bin/activate
```

### Q4：显示 "command not found: activate"

**解决**：
```bash
# 确保在正确的目录
cd 项目目录

# 使用正确的路径
source ./venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate     # Windows
```

### Q5：模块导入失败

```bash
# 检查虚拟环境是否激活
which python  # Linux/Mac，应该显示 .../venv/bin/python
where python  # Windows，应该显示 .../venv/Scripts/python.exe

# 重新安装失败的包
pip install --force-reinstall librosa
```

### Q6：GPU 不可用

```bash
# 检查
python -c "import torch; print(torch.cuda.is_available())"

# 如果返回 False，检查：
nvidia-smi  # 查看 GPU
nvcc --version  # 查看 CUDA 版本
```

### Q7：显存不足

**解决**：

1. 减少 batch size：
```yaml
# 在 config.yaml 中
dataset:
  batch_size: 4  # 改小
```

2. 减少帧数：
```yaml
dataset:
  num_frames: 4  # 从 8 改为 4
```

3. 使用 CPU：
```bash
python train.py --device cpu
```

---

## 环境变量配置（可选）

### Windows

```batch
# 设置 PYTHONPATH
set PYTHONPATH=%cd%;%PYTHONPATH%

# 或编辑系统环境变量
# 计算机 > 属性 > 高级系统设置 > 环境变量
```

### Linux/Mac

```bash
export PYTHONPATH="$PWD:$PYTHONPATH"

# 或添加到 ~/.bashrc 或 ~/.zshrc
echo 'export PYTHONPATH="$PWD:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```

---

## 验证完整安装

```bash
# 运行完整的环境检查
python check_env.py

# 运行快速启动脚本
python quickstart.py

# 运行示例
python examples.py

# 运行训练（需要数据）
python train.py --config configs/config.yaml --help
```

---

## 获取帮助

遇到问题？

1. **查看日志**：
```bash
cat logs/latest.log  # Linux/Mac
type logs\latest.log  # Windows
```

2. **运行诊断**：
```bash
python check_env.py
```

3. **检查依赖**：
```bash
pip list | grep torch
pip list | grep transformer
```

4. **重新安装**：
```bash
pip install --upgrade --force-reinstall -r requirements.txt
```

---

## 下一步

✅ 环境配置完成后：

1. 查看项目结构：`tree .` 或 `ls -la`

2. 运行示例：
```bash
python examples.py
```

3. 快速测试：
```bash
python quickstart.py
```

4. 准备数据并开始训练：
```bash
python train.py --config configs/config.yaml --data-root ./data
```

---

**准备好了！开始探索多模态视频学习框架吧！** 🚀
