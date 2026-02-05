"""Environment check script - verify required dependencies are installed"""

import sys
import subprocess
from importlib.util import find_spec
from pathlib import Path


class EnvironmentChecker:
    """环境检查器"""
    
    def __init__(self):
        self.results = {
            '✓': [],
            '⚠': [],
            '✗': []
        }
    
    def check_python_version(self):
        """检查 Python 版本"""
        version = sys.version_info
        min_version = (3, 9)
        
        if version >= min_version:
            self.results['✓'].append(
                f"Python {version.major}.{version.minor}.{version.micro}"
            )
        else:
            self.results['✗'].append(
                f"Python version too low: {version.major}.{version.minor}, requires 3.9+"
            )
    
    def check_module(self, module_name, display_name=None, critical=False):
        """检查模块是否安装"""
        display_name = display_name or module_name
        
        try:
            spec = find_spec(module_name)
            if spec is not None:
                # 尝试导入并获取版本
                try:
                    module = __import__(module_name)
                    version = getattr(module, '__version__', 'unknown')
                    self.results['✓'].append(f"{display_name}: {version}")
                except Exception as e:
                    self.results['✓'].append(f"{display_name}: installed")
            else:
                msg = f"{display_name}: not installed"
                if critical:
                    self.results['✗'].append(msg)
                else:
                    self.results['⚠'].append(msg)
        except Exception as e:
            msg = f"{display_name}: check failed ({str(e)[:30]})"
            if critical:
                self.results['✗'].append(msg)
            else:
                self.results['⚠'].append(msg)
    
    def check_cuda(self):
        """检查 CUDA 支持"""
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                device_name = torch.cuda.get_device_name(0)
                self.results['✓'].append(
                    f"CUDA: available ({device_count} GPU: {device_name})"
                )
            else:
                self.results['⚠'].append("CUDA: unavailable (using CPU mode)")
        except Exception as e:
            self.results['⚠'].append(f"CUDA check failed: {e}")
    
    def check_gpu_memory(self):
        """Check GPU memory"""
        try:
            import torch
            if torch.cuda.is_available():
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.results['✓'].append(
                    f"GPU memory: {total_memory:.2f} GB"
                )
        except Exception as e:
            pass
    
    def check_directories(self):
        """检查必要的目录"""
        dirs = ['configs', 'models', 'modules', 'data', 'checkpoints', 'logs']
        missing = []
        
        for dir_name in dirs:
            if not Path(dir_name).exists():
                missing.append(dir_name)
        
        if missing:
                self.results['⚠'].append(f"Missing directories: {', '.join(missing)}")
        else:
            self.results['✓'].append("All required directories exist")
    
    def run_all_checks(self):
        """运行所有检查"""
        print("\n" + "="*60)
        print("  Multimodal Video Learning Framework - Environment Check")
        print("="*60 + "\n")
        
        # Python 版本
        print("[CHECK] Python environment...")
        self.check_python_version()
        
        # 核心包
        print("[CHECK] Core dependencies...")
        critical_packages = [
            ('torch', 'PyTorch', True),
            ('torchvision', 'torchvision', True),
            ('transformers', 'Transformers', True),
            ('numpy', 'NumPy', True),
        ]
        
        for module, display, critical in critical_packages:
            self.check_module(module, display, critical)
        
        # 可选包
        print("[CHECK] Optional dependencies...")
        optional_packages = [
            ('librosa', 'Librosa'),
            ('cv2', 'OpenCV'),
            ('sklearn', 'scikit-learn'),
            ('yaml', 'PyYAML'),
            ('tensorboard', 'TensorBoard'),
            ('tqdm', 'tqdm'),
            ('PIL', 'Pillow'),
            ('scipy', 'SciPy'),
        ]
        
        for module, display in optional_packages:
            self.check_module(module, display, False)
        
        # CUDA 检查
        print("[CHECK] GPU support...")
        self.check_cuda()
        self.check_gpu_memory()
        
        # 目录检查
        print("[CHECK] Project structure...")
        self.check_directories()
        
        # 打印结果
        self.print_results()
    
    def print_results(self):
        """打印检查结果"""
        print("\n" + "="*60)
        print("  Check Results")
        print("="*60 + "\n")
        
        # 正常
        if self.results['✓']:
            print("✓ OK ({} items):".format(len(self.results['✓'])))
            for item in self.results['✓']:
                print(f"  ✓ {item}")
            print()
        
        # 警告
        if self.results['⚠']:
            print("⚠ Warnings ({} items):".format(len(self.results['⚠'])))
            for item in self.results['⚠']:
                print(f"  ⚠ {item}")
            print()
        
        # 错误
        if self.results['✗']:
            print("✗ Errors ({} items):".format(len(self.results['✗'])))
            for item in self.results['✗']:
                print(f"  ✗ {item}")
            print()
        
        # 总结
        print("="*60)
        total = sum(len(v) for v in self.results.values())
        status = "✓ OK" if not self.results['✗'] else "✗ Needs fixes"
        print(f"Total: {total} items - {status}")
        print("="*60 + "\n")
        
        # 建议
        if self.results['✗'] or self.results['⚠']:
            print("Recommendations:")
            print()
            if self.results['✗']:
                print("1. Install missing critical packages:")
                print("   pip install -r requirements.txt")
                print()
            
            if self.results['⚠']:
                print("2. Optional packages missing — some features may be unavailable")
                print("   python setup_windows.bat  (Windows)")
                print("   python setup_linux.sh     (Linux)")
                print()


def main():
    checker = EnvironmentChecker()
    checker.run_all_checks()
    
    # 如果有错误，返回非零退出码
    if checker.results['✗']:
        sys.exit(1)


if __name__ == '__main__':
    main()
