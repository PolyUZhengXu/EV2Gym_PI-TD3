# Multimodal Video Framework - PowerShell auto-install script
# 使用方法: 在 PowerShell 中运行
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# .\setup_windows.ps1

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Multimodal Video Framework - Auto Setup" -ForegroundColor Cyan
# 原: Write-Host "  多模态视频学习框架 - 环境自动配置" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# 函数：检查命令是否存在
function Test-CommandExists {
    param($command)
    $null = Get-Command $command -ErrorAction SilentlyContinue
    return $?
}

# 检查 Python
Write-Host "[CHECK] Python environment..." -ForegroundColor Yellow
# 原: Write-Host "[检查] Python 环境..." -ForegroundColor Yellow
if (Test-CommandExists python) {
    $pythonVersion = python --version 2>&1
    Write-Host "[✓] Python installed: $pythonVersion" -ForegroundColor Green
# 原: Write-Host "[✓] Python 已安装: $pythonVersion" -ForegroundColor Green
} else {
    Write-Host "[✗] Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.9 or newer: https://www.python.org" -ForegroundColor Red
    # 原: Write-Host "请先安装 Python 3.9 或更高版本：https://www.python.org" -ForegroundColor Red
    exit 1
}

# 检查 pip
Write-Host "[CHECK] pip..." -ForegroundColor Yellow
# 原: Write-Host "[检查] pip..." -ForegroundColor Yellow
if (Test-CommandExists pip) {
    Write-Host "[✓] pip detected" -ForegroundColor Green
# 原: Write-Host "[✓] pip 已安装" -ForegroundColor Green
} else {
    Write-Host "[✗] pip 未安装" -ForegroundColor Red
    exit 1
}

Write-Host ""

# 创建虚拟环境
Write-Host "[Step 1] Creating virtual environment..." -ForegroundColor Yellow
# 原: Write-Host "[步骤 1] 创建虚拟环境..." -ForegroundColor Yellow
if (Test-Path "venv") {
    Write-Host "[INFO] venv exists, skipping creation" -ForegroundColor Cyan
# 原: Write-Host "[信息] venv 已存在，跳过创建" -ForegroundColor Cyan
} else {
    Write-Host "创建虚拟环境中..." -ForegroundColor Gray
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[✗] Failed to create virtual environment" -ForegroundColor Red
    # 原: Write-Host "[✗] 虚拟环境创建失败" -ForegroundColor Red
        exit 1
    }
    Write-Host "[✓] Virtual environment created" -ForegroundColor Green
# 原: Write-Host "[✓] 虚拟环境创建成功" -ForegroundColor Green
}

Write-Host ""

# 激活虚拟环境
Write-Host "[Step 2] Activating virtual environment..." -ForegroundColor Yellow
# 原: Write-Host "[步骤 2] 激活虚拟环境..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[✗] Failed to activate virtual environment" -ForegroundColor Red
# 原: Write-Host "[✗] 虚拟环境激活失败" -ForegroundColor Red
    exit 1
}
Write-Host "[✓] Virtual environment activated" -ForegroundColor Green
# 原: Write-Host "[✓] 虚拟环境已激活" -ForegroundColor Green

Write-Host ""

# 升级 pip
Write-Host "[Step 3] Upgrading pip..." -ForegroundColor Yellow
# 原: Write-Host "[步骤 3] 升级 pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip -q
if ($LASTEXITCODE -eq 0) {
    Write-Host "[✓] pip upgraded" -ForegroundColor Green
    # 原: Write-Host "[✓] pip 升级成功" -ForegroundColor Green
} else {
    Write-Host "[WARNING] pip upgrade may have failed, continuing..." -ForegroundColor Yellow
    # 原: Write-Host "[警告] pip 升级可能失败，继续安装..." -ForegroundColor Yellow
}

Write-Host ""

# 选择 CUDA 版本
Write-Host "[Step 4] Installing PyTorch..." -ForegroundColor Yellow
# 原: Write-Host "[步骤 4] 安装 PyTorch..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Select your CUDA option:" -ForegroundColor Cyan
# 原: Write-Host "选择您的 CUDA 版本：" -ForegroundColor Cyan
Write-Host "  1) CPU (recommended for no-GPU systems)" -ForegroundColor Gray
# 原: Write-Host "  1) CPU 版本（推荐无 GPU 用户，最快安装）" -ForegroundColor Gray
Write-Host "  2) CUDA 11.8 (NVIDIA GPU)" -ForegroundColor Gray
# 原: Write-Host "  2) CUDA 11.8 版本（NVIDIA GPU）" -ForegroundColor Gray
Write-Host "  3) CUDA 12.1 (newer NVIDIA GPU)" -ForegroundColor Gray
# 原: Write-Host "  3) CUDA 12.1 版本（最新 NVIDIA GPU）" -ForegroundColor Gray
Write-Host ""

$cudaChoice = Read-Host "Enter choice (1-3, default 1)"
# 原: $cudaChoice = Read-Host "请输入选择 (1-3, 默认为1)"
if ([string]::IsNullOrEmpty($cudaChoice)) {
    $cudaChoice = "1"
}

$torchUrl = ""
switch ($cudaChoice) {
    "1" {
        Write-Host "[CHOICE] CPU" -ForegroundColor Cyan
    # 原: Write-Host "[选择] CPU 版本" -ForegroundColor Cyan
        $torchUrl = "https://download.pytorch.org/whl/cpu"
    }
    "2" {
        Write-Host "[CHOICE] CUDA 11.8" -ForegroundColor Cyan
    # 原: Write-Host "[选择] CUDA 11.8 版本" -ForegroundColor Cyan
        $torchUrl = "https://download.pytorch.org/whl/cu118"
    }
    "3" {
        Write-Host "[CHOICE] CUDA 12.1" -ForegroundColor Cyan
    # 原: Write-Host "[选择] CUDA 12.1 版本" -ForegroundColor Cyan
        $torchUrl = "https://download.pytorch.org/whl/cu121"
    }
    default {
        Write-Host "[默认] CPU 版本" -ForegroundColor Cyan
        $torchUrl = "https://download.pytorch.org/whl/cpu"
    }
}

Write-Host "Installing, please wait..." -ForegroundColor Gray
# 原: Write-Host "安装中，请稍候..." -ForegroundColor Gray
pip install torch torchvision torchaudio --index-url $torchUrl -q
if ($LASTEXITCODE -eq 0) {
    Write-Host "[✓] PyTorch installed" -ForegroundColor Green
} else {
    Write-Host "[WARNING] PyTorch installation may have warnings" -ForegroundColor Yellow
# 原: Write-Host "[警告] PyTorch 安装可能失败" -ForegroundColor Yellow
}

Write-Host ""

# 安装其他依赖
Write-Host "[Step 5] Installing other dependencies..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Gray
# 原: Write-Host "[步骤 5] 安装其他依赖包..." -ForegroundColor Yellow
# 原: Write-Host "这可能需要几分钟时间..." -ForegroundColor Gray

$packages = @(
    "transformers>=4.30.0",
    "librosa>=0.10.0",
    "scikit-learn>=1.3.0",
    "pyyaml>=6.0",
    "tensorboard>=2.13.0",
    "tqdm>=4.66.0",
    "opencv-python>=4.8.0",
    "Pillow>=10.0.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0"
)

foreach ($package in $packages) {
    Write-Host "Installing $package..." -ForegroundColor Gray
# 原: Write-Host "安装 $package..." -ForegroundColor Gray
    pip install $package -q
}

Write-Host "[✓] All dependencies installed" -ForegroundColor Green
# 原: Write-Host "[✓] 所有依赖包安装成功" -ForegroundColor Green

Write-Host ""

# 验证安装
Write-Host "[Step 6] Verifying installation..." -ForegroundColor Yellow
# 原: Write-Host "[步骤 6] 验证安装..." -ForegroundColor Yellow
Write-Host ""

try {
    $torchVersion = python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}')"
    Write-Host "[✓] PyTorch: $($torchVersion -split "`n")" -ForegroundColor Green
# 原: Write-Host "[✓] PyTorch: $($torchVersion -split "`n")" -ForegroundColor Green
    
    $tfVersion = python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
    Write-Host "[✓] $tfVersion" -ForegroundColor Green
# 原: Write-Host "[✓] $tfVersion" -ForegroundColor Green
    
    $librosaCheck = python -c "import librosa; print('Librosa: OK')"
    Write-Host "[✓] $librosaCheck" -ForegroundColor Green
# 原: Write-Host "[✓] $librosaCheck" -ForegroundColor Green
    
    $cvVersion = python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
    Write-Host "[✓] $cvVersion" -ForegroundColor Green
# 原: Write-Host "[✓] $cvVersion" -ForegroundColor Green
    
} catch {
    Write-Host "[警告] 某些模块验证失败" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "[✓] Environment setup complete!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "" 
Write-Host "1. Change to the project directory:" -ForegroundColor White
Write-Host "   cd multimodal_video_learning" -ForegroundColor Gray
Write-Host "" 
Write-Host "2. Quick test (optional):" -ForegroundColor White
Write-Host "   python quickstart.py" -ForegroundColor Gray
Write-Host "" 
Write-Host "3. Run examples:" -ForegroundColor White
Write-Host "   python examples.py" -ForegroundColor Gray
Write-Host "" 
Write-Host "4. Start training:" -ForegroundColor White
Write-Host "   python train.py --config configs/config.yaml" -ForegroundColor Gray
Write-Host "" 
Write-Host "5. Model evaluation:" -ForegroundColor White
Write-Host "   python inference.py --model checkpoints/best_model.pt" -ForegroundColor Gray
Write-Host "" 
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Press Enter to exit..." -ForegroundColor Yellow
Write-Host "原: 按 Enter 键退出..." -ForegroundColor DarkGray
Read-Host
