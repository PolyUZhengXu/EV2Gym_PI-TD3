#!/usr/bin/env python
"""
从项目根目录运行训练的启动脚本
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 现在导入并运行训练脚本
from multimodal_video_learning import train

if __name__ == '__main__':
    train.main()
