#!/bin/bash
# 多模态视频学习框架 - Linux/Mac 自动安装脚本
# 使用方法: chmod +x setup_linux.sh && ./setup_linux.sh

set -e  # 遇到错误立即退出

echo ""
echo "=========================================="
echo "  多模态视频学习框架 - 环境自动配置"
echo "=========================================="
echo ""

# 检查 Python
echo "[检查] Python 环境..."
if ! command -v python3 &> /dev/null; then
    echo "[✗] Python3 未安装"
    echo "请先安装 Python 3.9 或更高版本"
    exit 1
fi

python3 --version
echo "[✓] Python 已安装"
echo ""

# 检查 pip
echo "[检查] pip..."
if ! command -v pip3 &> /dev/null; then
    echo "[✗] pip3 未安装"
    exit 1
fi
echo "[✓] pip 已安装"
echo ""

# 创建虚拟环境
echo "[步骤 1] 创建虚拟环境..."
if [ -d "venv" ]; then
    echo "[信息] venv 已存在，跳过创建"
else
    python3 -m venv venv
    echo "[✓] 虚拟环境创建成功"
fi
echo ""

# 激活虚拟环境
echo "[步骤 2] 激活虚拟环境..."
source venv/bin/activate
echo "[✓] 虚拟环境已激活"
echo ""

# 升级 pip
echo "[步骤 3] 升级 pip..."
python3 -m pip install --upgrade pip -q
echo "[✓] pip 升级成功"
echo ""

# 选择 CUDA 版本
echo "[步骤 4] 安装 PyTorch..."
echo ""
echo "选择您的 CUDA 版本："
echo "  1) CPU 版本（推荐无 GPU 用户）"
echo "  2) CUDA 11.8 版本（NVIDIA GPU）"
echo "  3) CUDA 12.1 版本（最新 NVIDIA GPU）"
echo ""
read -p "请输入选择 (1-3, 默认为1): " cuda_choice

case ${cuda_choice:-1} in
    1)
        echo "[选择] CPU 版本"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
        ;;
    2)
        echo "[选择] CUDA 11.8 版本"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
        ;;
    3)
        echo "[选择] CUDA 12.1 版本"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q
        ;;
    *)
        echo "[默认] CPU 版本"
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu -q
        ;;
esac

echo "[✓] PyTorch 安装成功"
echo ""

# 安装其他依赖
echo "[步骤 5] 安装其他依赖包..."
echo "这可能需要几分钟时间..."

pip install \
    transformers>=4.30.0 \
    librosa>=0.10.0 \
    scikit-learn>=1.3.0 \
    pyyaml>=6.0 \
    tensorboard>=2.13.0 \
    tqdm>=4.66.0 \
    opencv-python>=4.8.0 \
    Pillow>=10.0.0 \
    numpy>=1.24.0 \
    scipy>=1.11.0 \
    -q

echo "[✓] 所有依赖包安装成功"
echo ""

# 验证安装
echo "[步骤 6] 验证安装..."
echo ""

python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}')" && echo "[✓] PyTorch" || echo "[⚠] PyTorch 检查失败"

python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" && echo "[✓] Transformers" || echo "[⚠] Transformers 检查失败"

python3 -c "import librosa; print('Librosa: OK')" && echo "[✓] Librosa" || echo "[⚠] Librosa 检查失败"

python3 -c "import cv2; print(f'OpenCV: {cv2.__version__}')" && echo "[✓] OpenCV" || echo "[⚠] OpenCV 检查失败"

echo ""
echo "=========================================="
echo "[✓] 环境配置完成！"
echo "=========================================="
echo ""

echo "下一步操作："
echo ""
echo "1. 进入项目目录："
echo "   cd multimodal_video_learning"
echo ""
echo "2. 快速测试（可选）："
echo "   python quickstart.py"
echo ""
echo "3. 查看示例："
echo "   python examples.py"
echo ""
echo "4. 开始训练："
echo "   python train.py --config configs/config.yaml"
echo ""
echo "5. 模型评估："
echo "   python inference.py --model checkpoints/best_model.pt"
echo ""
echo "=========================================="
echo ""
