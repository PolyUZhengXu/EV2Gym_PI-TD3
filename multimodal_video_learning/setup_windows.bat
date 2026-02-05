@echo off
REM 多模态视频学习框架 - 一键环境配置脚本
REM 适用于 Windows 系统

setlocal enabledelayedexpansion
title Multimodal Video Framework - Setup
REM 原: title 多模态视频框架 - 环境配置

echo.
echo ==========================================
echo  Multimodal Video Framework - Installation Wizard
REM 原: echo  多模态视频学习框架 - 环境安装向导
echo ==========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    REM 原: echo [错误] Python 未安装或不在PATH中
    echo Please install Python 3.9 or newer
    REM 原: echo 请先安装 Python 3.9 或更高版本
    pause
    exit /b 1
)

echo [✓] Python detected
python --version

REM Check pip
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip not found
    REM 原: echo [错误] pip 未安装
    pause
    exit /b 1
)

echo [✓] pip detected
echo.

REM Create virtual environment
echo [Step 1] Create virtual environment...
REM 原: echo [步骤1] 创建虚拟环境...
if exist "venv" (
    echo [INFO] venv already exists, skipping creation
    REM 原: echo [信息] venv 已存在，跳过创建
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        REM 原: echo [错误] 虚拟环境创建失败
        pause
        exit /b 1
    )
    echo [✓] Virtual environment created
    REM 原: echo [✓] 虚拟环境创建成功
)

REM Activate virtual environment
echo.
echo [Step 2] Activating virtual environment...
REM 原: echo [步骤2] 激活虚拟环境...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    REM 原: echo [错误] 虚拟环境激活失败
    pause
    exit /b 1
)
echo [✓] Virtual environment activated
REM 原: echo [✓] 虚拟环境已激活

REM Upgrade pip
echo.
echo [Step 3] Upgrading pip...
REM 原: echo [步骤3] 升级 pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARNING] pip upgrade failed, continuing with dependency installation...
    REM 原: echo [警告] pip 升级失败，继续安装依赖...
)

REM Install PyTorch (choose CUDA)
echo.
echo [Step 4] Installing PyTorch...
REM 原: echo [步骤4] 安装 PyTorch...
echo Select your CUDA option:
REM 原: echo 选择您的CUDA版本：
echo   1) CPU (recommended for no-GPU systems)
REM 原: echo   1) CPU 版本（推荐无GPU用户）
echo   2) CUDA 11.8
REM 原: echo   2) CUDA 11.8 版本
echo   3) CUDA 12.1
REM 原: echo   3) CUDA 12.1 版本
set /p cuda_choice="Enter choice (1-3): "

if "%cuda_choice%"=="1" (
    echo Installing CPU version...
    REM 原: echo 安装 CPU 版本...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
) else if "%cuda_choice%"=="2" (
    echo Installing CUDA 11.8 version...
    REM 原: echo 安装 CUDA 11.8 版本...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
) else if "%cuda_choice%"=="3" (
    echo Installing CUDA 12.1 version...
    REM 原: echo 安装 CUDA 12.1 版本...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
) else (
    echo [default] Installing CPU version...
    REM 原: echo [默认] 安装 CPU 版本...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
)

if errorlevel 1 (
    echo [WARNING] PyTorch installation may have failed; check your network connection
    REM 原: echo [警告] PyTorch 安装可能失败，请检查网络连接
)

REM Install other dependencies
echo.
echo [Step 5] Installing other dependencies...
REM 原: echo [步骤5] 安装其他依赖包...
pip install transformers>=4.30.0 ^
    librosa>=0.10.0 ^
    scikit-learn>=1.3.0 ^
    pyyaml>=6.0 ^
    tensorboard>=2.13.0 ^
    tqdm>=4.66.0 ^
    opencv-python>=4.8.0 ^
    Pillow>=10.0.0 ^
    numpy>=1.24.0 ^
    scipy>=1.11.0

if errorlevel 1 (
    echo [WARNING] Some dependencies may have failed to install
    REM 原: echo [警告] 某些依赖安装失败
)

REM Verify installation
echo.
echo [Step 6] Verifying installation...
REM 原: echo [步骤6] 验证安装...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import librosa; print('Librosa: OK')"
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

echo.
echo ==========================================
echo [✓] Environment setup complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Change to project directory:
echo    cd multimodal_video_learning
echo.
echo 2. Quick test (optional):
echo    python quickstart.py
echo.
echo 3. Run examples:
echo    python examples.py
echo.
echo 4. Start training:
echo    python train.py --config configs/config.yaml
echo.
echo ==========================================
echo.
REM 原: echo 下一步：
REM 原: echo 1. 进入项目目录：
REM 原: echo    cd multimodal_video_learning
REM 原: echo.
REM 原: echo 2. 快速测试（可选）：
REM 原: echo    python quickstart.py
REM 原: echo.
REM 原: echo 3. 查看示例：
REM 原: echo    python examples.py
REM 原: echo.
REM 原: echo 4. 开始训练：
REM 原: echo    python train.py --config configs/config.yaml
echo.

pause
